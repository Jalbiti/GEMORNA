#!/usr/bin/env python3
"""
Standalone reimplementation of GEMORNA's generation logic (CDS / 5'UTR / 3'UTR),
built by reverse-engineering the compiled `.gen()`/`.sampling()` methods in
src/shared/{libg2m,mod_xzr01}.so via dynamic tensor-op tracing (see
debug/trace_generation.py, and the "gemorna-compiled-generation-mechanics" memory
for how each fact below was established). No .pyx/.c source for those methods
exists anywhere in this repo -- everything here is re-derived from observed
runtime behavior, not read from source, so treat it as a best-effort mimic, not
a guaranteed-identical reimplementation.

Confirmed facts encoded here:
  - All three modes: softmax(logits) -> torch.multinomial(probs, 1) each step.
    No temperature scaling, no top-k/top-p, no beam search anywhere.
  - No KV cache: recomputes the full forward pass every step, exactly like the
    compiled code (confirmed via traced attention shapes growing every step).
  - CDS (encoder-decoder): protein is wrapped [<sos>] + residues + [<eos>] and
    encoded once; decoder autoregressively samples exactly one codon per
    residue -- loop length is len(protein_seq), not <eos>-driven. Each step's
    347-way softmax is masked down to only the synonymous-codon set for the
    current residue (config.codon_dict) before sampling -- NOT verified against
    the compiled binary directly (not visible to tensor tracing, since it's a
    plain index-set restriction), but empirically necessary: an unconstrained
    multinomial reliably degenerates into invalid tokens (including literal
    <eos>) partway through a long (100-residue) test protein, while the real
    compiled generator never does this for any protein length. Treat this
    specific detail as inferred-by-necessity, not directly confirmed.
  - 5' UTR (GPT-style decoder): output = "AGG" (hardcoded literal, confirmed via
    `strings` on the .so) + N sampled 3-mer tokens + "GCCACC" (hardcoded Kozak
    consensus literal, also confirmed via `strings`).
  - 3' UTR (GPT-style decoder): output = N sampled 3-mer tokens, no fixed
    flanks. Stray 'N' characters from sampled IUPAC-ambiguity tokens (e.g.
    'CAN') are stripped from the final string. A mid-sequence <eos> does NOT
    stop generation early -- confirmed by tracing a run where <eos> was sampled
    partway through and generation continued for 8 more steps; it's simply
    excluded from the joined output, same as <sos>.

IMPORTANT CAVEAT -- the per-length-class token count N is NOT a fixed constant.
Repeated runs of debug/trace_generation.py against the SAME --utr_length class
produced DIFFERENT token counts every time (e.g. 5' UTR "short": 11, 12, 14,
15, 16 tokens across 7 separate runs; 3' UTR "long": 35-43 across 4 runs). The
sampled *content* is highly reproducible for whatever length two runs happen to
share (consistent with a fixed torch RNG seed inside the compiled code), but
the stopping length itself varies within a class-dependent range. So it's a
per-call random draw, not a hardcoded count -- an earlier single-sample-per-
class investigation wrongly concluded it was fixed.

LENGTH_RANGES below are empirically calibrated from ~4-7 repeated runs per
class and are almost certainly NOT the exact true bounds used inside the
compiled binary -- just an approximation good enough for exploratory use.
Re-calibrate with more repeats (rerun debug/trace_generation.py many times per
class and count multinomial calls, e.g. `grep -c multinomial <full-log>`)
before relying on this for anything where the exact length distribution
matters.

Usage (run from the GEMORNA repo root, inside the `gemorna` conda env):

    python debug/mimic_generate.py --mode 5utr --ckpt_path checkpoints/gemorna_5utr.pt \
        --utr_length short

    python debug/mimic_generate.py --mode 3utr --ckpt_path checkpoints/gemorna_3utr.pt \
        --utr_length long --n_tokens 40   # override the random length draw

    python debug/mimic_generate.py --mode cds --ckpt_path checkpoints/gemorna_cds.pt \
        --protein_seq MAKGEEL
"""
import argparse
import os
import pickle
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Must run with cwd == GEMORNA repo root (same convention as src/generate.py).
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from config import (  # noqa: E402
    GEMORNA_5UTR_Config,
    GEMORNA_3UTR_Config,
    GEMORNA_CDS_Config,
    codon_dict,
    five_prime_utr_vocab,
    three_prime_utr_vocab,
)
from models.gemorna_utr import LayerNorm, DecoderBlock  # noqa: E402
# Importing gemorna_cds pulls in the compiled shared module as a side effect
# (only used there for the `max_length` constant + reference layer classes,
# not for the actual sampling loop below, which is entirely reimplemented here).
from models.gemorna_cds import Encoder, Decoder  # noqa: E402

# Empirically calibrated (see caveat above) -- (min, max) sampled 3-mer tokens.
LENGTH_RANGES = {
    "5utr": {"short": (11, 16), "medium": (21, 27), "long": (29, 35)},
    "3utr": {"short": (16, 21), "medium": (26, 32), "long": (37, 43)},
}


class MimicUTR(nn.Module):
    """GPT-style decoder-only model matching the transformer.*/lm_head.* keys
    found in checkpoints/gemorna_{5,3}utr.pt (verified via `torch.load(...).keys()`)."""

    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        _, t = idx.shape
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)


def load_utr_model(ckpt_path, config_cls, device):
    config = config_cls()
    model = MimicUTR(config)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def load_cds_model(ckpt_path, device):
    cfg = GEMORNA_CDS_Config()
    encoder = Encoder(
        input_dim=cfg.input_dim,
        hid_dim=cfg.hidden_dim,
        n_layers=cfg.num_layers,
        n_heads=cfg.num_heads,
        pf_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        cnn_kernel_size=cfg.cnn_kernel_size,
        cnn_padding=cfg.cnn_padding,
        device=device,
    )
    decoder = Decoder(
        output_dim=cfg.output_dim,
        hid_dim=cfg.hidden_dim,
        n_layers=cfg.num_layers,
        n_heads=cfg.num_heads,
        pf_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        device=device,
    )
    sd = torch.load(ckpt_path, map_location=device)
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    dec_sd = {k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")}
    encoder.load_state_dict(enc_sd)
    decoder.load_state_dict(dec_sd)
    encoder.to(device).eval()
    decoder.to(device).eval()
    return encoder, decoder


def generate_utr(model, vocab, device, mode, utr_length, n_tokens=None):
    assert mode in ("5utr", "3utr")
    inv_vocab = {i: t for t, i in vocab.items()}
    sos_id, eos_id = vocab["<sos>"], vocab["<eos>"]

    if n_tokens is None:
        lo, hi = LENGTH_RANGES[mode][utr_length]
        n_tokens = random.randint(lo, hi)

    idx = torch.tensor([[sos_id]], dtype=torch.long, device=device)
    sampled_ids = []
    with torch.no_grad():
        for _ in range(n_tokens):
            logits = model(idx)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            sampled_ids.append(next_id.item())
            idx = torch.cat([idx, next_id], dim=1)

    # <sos>/<eos> contribute nothing to the output even if sampled mid-sequence
    # (confirmed for 3' UTR; assumed symmetric for 5' UTR by architecture).
    tokens = [inv_vocab[i] for i in sampled_ids if i not in (sos_id, eos_id)]
    # vocab tokens are RNA (U); the real compiled .gen() prints DNA-style (T) --
    # confirmed by every observed real output using T, never U.
    core = "".join(tokens).replace("N", "").replace("U", "T")

    if mode == "5utr":
        return "AGG" + core + "GCCACC"
    return core


def generate_cds(encoder, decoder, protein_seq, prot_vocab, cds_vocab, device):
    protein_lower = protein_seq.lower()
    sos_p, eos_p = prot_vocab.stoi["<sos>"], prot_vocab.stoi["<eos>"]
    prot_ids = [sos_p] + [prot_vocab.stoi[a] for a in protein_lower] + [eos_p]
    prot = torch.tensor([prot_ids], dtype=torch.long, device=device)

    sos_c = cds_vocab.stoi["<sos>"]
    cds = torch.tensor([[sos_c]], dtype=torch.long, device=device)

    with torch.no_grad():
        enc_prot = encoder(prot, None)
        for residue in protein_lower:
            # Constrain each step to the synonymous-codon set for the current
            # residue (config.codon_dict) -- an unconstrained full-vocab
            # multinomial can (rarely, but compounding over a long
            # autoregressive chain) sample an invalid token like <eos>
            # mid-sequence; verified this actually happens on a 100-residue
            # test protein without this mask.
            allowed_ids = torch.tensor(
                [cds_vocab.stoi[c] for c in codon_dict[residue]], device=device
            )
            logits, _ = decoder(cds, enc_prot, None, None)
            step_logits = logits[:, -1, :]
            masked_logits = torch.full_like(step_logits, float("-inf"))
            masked_logits[:, allowed_ids] = step_logits[:, allowed_ids]
            probs = F.softmax(masked_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            cds = torch.cat([cds, next_id], dim=1)

    codon_ids = cds[0, 1:].tolist()
    return "".join(cds_vocab.itos[i] for i in codon_ids).upper()


def main():
    parser = argparse.ArgumentParser(description="Mimic GEMORNA's compiled generation logic")
    parser.add_argument("--mode", required=True, choices=["cds", "5utr", "3utr"])
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--protein_seq", default=None)
    parser.add_argument("--utr_length", default=None, choices=["short", "medium", "long"])
    parser.add_argument(
        "--n_tokens",
        type=int,
        default=None,
        help="override the random length-class draw with an exact sampled-token count (5utr/3utr only)",
    )
    parser.add_argument("--seed", type=int, default=None, help="torch.manual_seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "cds":
        if not args.protein_seq:
            raise SystemExit("--protein_seq is required for --mode cds")
        with open("vocab/prot_vocab.pkl", "rb") as f:
            prot_vocab = pickle.load(f)
        with open("vocab/cds_vocab.pkl", "rb") as f:
            cds_vocab = pickle.load(f)
        encoder, decoder = load_cds_model(args.ckpt_path, device)
        seq = generate_cds(encoder, decoder, args.protein_seq, prot_vocab, cds_vocab, device)
    else:
        if not args.utr_length and args.n_tokens is None:
            raise SystemExit("--utr_length (or --n_tokens) is required for --mode 5utr/3utr")
        config_cls = GEMORNA_5UTR_Config if args.mode == "5utr" else GEMORNA_3UTR_Config
        vocab = five_prime_utr_vocab if args.mode == "5utr" else three_prime_utr_vocab
        model = load_utr_model(args.ckpt_path, config_cls, device)
        seq = generate_utr(model, vocab, device, args.mode, args.utr_length, n_tokens=args.n_tokens)

    print(seq)


if __name__ == "__main__":
    main()
