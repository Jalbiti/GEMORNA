#!/usr/bin/env python3
"""
Dynamic tracer for GEMORNA's compiled UTR/CDS generation (shared/libg2m.so or
shared/mod_xzr01.so). The .gen()/.sampling() methods are Cython-compiled and
can't be read as source, but they still call real, interceptable PyTorch
Python API functions (F.softmax, torch.multinomial, torch.topk, tensor math,
nn.functional.embedding, ...). We install a TorchFunctionMode that intercepts
*every* torch API call made while model.gen(...) runs, so we get a full,
ordered log of the actual numeric operations without needing a disassembler.

Usage (run from the GEMORNA repo root, inside the `gemorna` conda env):

    python trace_utr_gen.py --mode 5utr --ckpt_path checkpoints/gemorna_5utr.pt \
        --utr_length short --out /tmp/utr_trace.log

    python trace_utr_gen.py --mode cds --ckpt_path checkpoints/gemorna_cds.pt \
        --protein_seq MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLSAESVGEVYIKSTETGQYLAMDTSGLLYGSQTPSEECLFLERLEENHYNTYTSKKHAEKNWFVGLKKNGSCKRGPRTHYGQKAILFLPLPVSSD \
        --out /tmp/cds_trace.log

Output:
  - Full ordered call trace (every torch op) written to --out.
  - A live, filtered stream of "interesting" ops (softmax, multinomial, topk,
    argmax, embedding, div/mul-by-scalar, cat, masked_fill, where, arange,
    zeros/full) printed to stdout as they happen, each tagged with a running
    "sample step" counter that increments on every multinomial/argmax/topk
    call (a reasonable proxy for "one generated token").
  - Special attention to nn.functional.embedding calls: print the embedding
    weight's shape. A weight shaped (3, n_embd) sitting alongside the normal
    (vocab_size, n_embd) token embedding is a strong signal of a dedicated
    short/medium/long length-class embedding table.

This intercepts Python-level torch API calls. It will NOT reveal pure-Python
control flow inside the Cython code that never touches a tensor (e.g. a plain
`if utr_len == 'short': ...` branch) unless that branch's effect shows up as a
different tensor op (different embedding index, different loop bound via
torch.arange/zeros shape, etc.) -- which is usually the case for anything that
affects the generated output.
"""
import argparse
import os
import platform
import re
import sys

# Must run with cwd == GEMORNA repo root (same convention as src/generate.py,
# which uses relative paths like ./vocab/prot_vocab.pkl).
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

if platform.system() == "Darwin":
    from shared.libg2m import *  # noqa: F401,F403  (CDS, UTR, max_length)
elif platform.system() == "Linux":
    from shared.mod_xzr01 import *  # noqa: F401,F403
else:
    raise RuntimeError("Unsupported OS")

from config import *  # noqa: F401,F403  (GEMORNA_*_Config, vocab dicts, codon_dict, ...)
from models.gemorna_cds import *  # noqa: F401,F403
from models.gemorna_utr import *  # noqa: F401,F403

INTERESTING = re.compile(
    r"softmax|multinomial|topk|argmax|embedding|div|truediv|mul\b|cat\b|"
    r"masked_fill|where|arange|zeros|full\b|scaled_dot_product_attention|"
    r"cumsum|sort\b|gather|index_select",
    re.IGNORECASE,
)


def fmt(x, max_vals=12):
    import torch

    if isinstance(x, torch.Tensor):
        extra = ""
        if x.numel() <= max_vals:
            try:
                extra = f" vals={x.detach().flatten().tolist()}"
            except Exception:
                pass
        return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype}{extra})"
    if isinstance(x, (list, tuple)):
        inner = ", ".join(fmt(i, max_vals) for i in x)
        return f"[{inner}]"
    if isinstance(x, dict):
        inner = ", ".join(f"{k}={fmt(v, max_vals)}" for k, v in x.items())
        return "{" + inner + "}"
    return repr(x)


def build_trace_mode(log_fh):
    from torch.overrides import TorchFunctionMode

    try:
        from torch.overrides import resolve_name
    except ImportError:
        resolve_name = None

    class TraceMode(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.call_count = 0
            self.sample_step = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            self.call_count += 1
            name = None
            if resolve_name is not None:
                try:
                    name = resolve_name(func)
                except Exception:
                    name = None
            if not name:
                name = getattr(func, "__qualname__", None) or getattr(
                    func, "__name__", str(func)
                )

            is_interesting = bool(INTERESTING.search(name))
            result = func(*args, **kwargs)

            if is_interesting:
                if re.search(r"multinomial|argmax|topk", name, re.IGNORECASE):
                    self.sample_step += 1

                arg_str = ", ".join(fmt(a) for a in args)
                kwarg_str = ", ".join(f"{k}={fmt(v)}" for k, v in kwargs.items())
                sig = f"{name}({arg_str}{', ' if arg_str and kwarg_str else ''}{kwarg_str})"
                ret_str = fmt(result)
                line = (
                    f"[step {self.sample_step:04d}] [{self.call_count:06d}] "
                    f"{sig} -> {ret_str}"
                )
                print(line, flush=True)
                print(line, file=log_fh, flush=True)
            else:
                arg_str = ", ".join(fmt(a, max_vals=0) for a in args)
                print(f"[{self.call_count:06d}] {name}({arg_str})", file=log_fh)

            return result

    return TraceMode()


def main():
    parser = argparse.ArgumentParser(description="Trace GEMORNA compiled generation")
    parser.add_argument("--mode", required=True, choices=["cds", "5utr", "3utr"])
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--protein_seq", default=None)
    parser.add_argument("--utr_length", default=None, choices=["short", "medium", "long", None])
    parser.add_argument("--out", default="/tmp/gemorna_trace.log")
    args = parser.parse_args()

    repo_root = os.getcwd()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.out, "w") as log_fh:
        print(f"# full trace -> {args.out}", file=sys.stderr)
        trace_mode = build_trace_mode(log_fh)

        if args.mode == "cds":
            if not args.protein_seq:
                raise SystemExit("--protein_seq is required for --mode cds")
            import pickle

            with open(os.path.join(repo_root, "vocab/prot_vocab.pkl"), "rb") as f:
                prot_vocab = pickle.load(f)
            with open(os.path.join(repo_root, "vocab/cds_vocab.pkl"), "rb") as f:
                cds_vocab = pickle.load(f)

            model_config = GEMORNA_CDS_Config()
            enc = Encoder(
                input_dim=model_config.input_dim,
                hid_dim=model_config.hidden_dim,
                n_layers=model_config.num_layers,
                n_heads=model_config.num_heads,
                pf_dim=model_config.ff_dim,
                dropout=model_config.dropout,
                cnn_kernel_size=model_config.cnn_kernel_size,
                cnn_padding=model_config.cnn_padding,
                device=device,
            )
            dec = Decoder(
                output_dim=model_config.output_dim,
                hid_dim=model_config.hidden_dim,
                n_layers=model_config.num_layers,
                n_heads=model_config.num_heads,
                pf_dim=model_config.ff_dim,
                dropout=model_config.dropout,
                device=device,
            )
            model = CDS(enc, dec, model_config.prot_pad_idx, model_config.cds_pad_idx, device)
            model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
            model.to(device)
            model.eval()

            with trace_mode:
                model.gen(args.protein_seq, prot_vocab, cds_vocab, device)

        else:
            if not args.utr_length:
                raise SystemExit("--utr_length is required for --mode 5utr/3utr")
            if args.mode == "5utr":
                model_config = GEMORNA_5UTR_Config()
                vocab = five_prime_utr_vocab
            else:
                model_config = GEMORNA_3UTR_Config()
                vocab = three_prime_utr_vocab

            model = UTR(model_config)
            model.load_state_dict(torch.load(args.ckpt_path, map_location=device)["model"])
            model.to(device)
            model.eval()

            with trace_mode:
                model.gen(args.mode, vocab, device, args.utr_length)

        print(
            f"\n# done: {trace_mode.call_count} total torch calls, "
            f"{trace_mode.sample_step} sample-step ops (multinomial/argmax/topk)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
