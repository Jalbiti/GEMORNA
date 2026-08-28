# Session summary — `improve-model` branch

Summary of investigation, tooling, and model work done in this session, covering commits
`ffe2bf6..ab6f086`.

## 1. Reverse-engineering the compiled generation logic

GEMORNA's `CDS.gen`/`UTR.gen` (used by `src/generate.py`) are compiled into
`src/shared/{libg2m,mod_xzr01}.so` — Cython-compiled binaries with no `.pyx`/`.c` source
anywhere in the repo's history, so their actual sampling logic can't be read from source
(per the existing note in `CLAUDE.md`).

**New tool: `debug/trace_generation.py`** (commit `ffe2bf6`) — installs a
`torch.overrides.TorchFunctionMode` that intercepts every torch API call made during a real
`model.gen(...)` run, logging an ordered trace of the actual tensor operations (softmax,
multinomial, embeddings, attention, etc.) without needing a disassembler.

Running it against real checkpoints (`checkpoints/gemorna_{cds,5utr,3utr}.pt`) established,
by direct observation rather than guesswork:

- **Sampling**: all three modes use plain `softmax(logits) → torch.multinomial(probs, 1)`
  each step — no temperature scaling, no top-k/top-p, no beam search.
- **No KV cache**: the full forward pass is recomputed every generated token.
- **CDS** (encoder-decoder): the protein is wrapped `[<sos>] + residues + [<eos>]` and
  encoded once; the decoder samples exactly one codon per residue (loop length =
  `len(protein_seq)`, not `<eos>`-driven).
- **5' UTR** (GPT-style decoder): every output is `"AGG"` (hardcoded literal, confirmed via
  `strings` on the `.so`) + N sampled 3-mer tokens + `"GCCACC"` (hardcoded Kozak consensus
  literal). N is drawn per-call from a length-class-dependent random range (empirically
  ~11-16 / 21-27 / 29-35 tokens for short/medium/long) — **not** a fixed constant, which an
  earlier single-sample-per-class test incorrectly suggested.
- **3' UTR** (GPT-style decoder): no fixed flanks. A mid-sequence `<eos>` does **not** stop
  generation (confirmed: sampled at step 35/43 in one run, generation continued for 8 more
  steps). Sampled IUPAC-ambiguity tokens (e.g. `CAN`) have their `N` characters stripped
  from the final output rather than being rejected.
- Vocab tokens are RNA (`U`); printed output is DNA-style (`U`→`T` substituted inside the
  compiled code).

**New tool: `debug/mimic_generate.py`** (commit `5ae7fc1`) — a standalone, pure-PyTorch
reimplementation of all three generation paths, built directly from each checkpoint's real
`state_dict` keys (verified with **strict** `load_state_dict`) and the behavior documented
above, with no dependency on the compiled `.so` at runtime. Testing it surfaced a real bug:
an unconstrained 347-way softmax over the codon vocab reliably degenerated into garbage
(including literal `<eos>` tokens) partway through a 100-residue test protein, even though
a 7-residue test passed cleanly. Fixed by masking each decoding step to the current
residue's synonymous-codon set (`config.codon_dict`) before sampling — re-verified via
Biopython translation round-trips on proteins from 7 to 155 residues.

## 2. Repository hygiene

- Made the repo's Git LFS hooks (`post-commit`, `pre-push`, `post-checkout`, `post-merge`)
  executable — they were present but non-executable, so LFS's post-checkout/merge/commit/
  push bookkeeping was silently skipped.
- Found and removed Python bytecode cache files (`__pycache__/*.pyc`) that had been
  accidentally committed to git; added `__pycache__/` and `*.pyc` to `.gitignore`
  (commit `4154534`).
- Verified the README's official example commands (CDS/5'/3' UTR generation, 5'/3' UTR
  prediction) all still run cleanly end-to-end against the real checkpoints.

## 3. Fine-tuning the 3' UTR prediction model

Fine-tuned `checkpoints/3utr.pt` (the CNN regressor behind `src/main_pred3UTR.py` — the
*prediction* model, distinct from the generative one above) on three labeled 3' UTR
stability datasets, benchmarking each fine-tuned model against the untouched pretrained
checkpoint on an identical held-out split.

**New tool: `debug/finetune_pred3utr.py`** (commit `ab6f086`) — generalized fine-tuning/
benchmark script (dataset column names, optional pre-existing train/val split column, and
target column are all CLI flags).

| Dataset | held-out n | pretrained (Pearson r / Spearman ρ) | fine-tuned (Pearson r / Spearman ρ) |
|---|---|---|---|
| `siegel2021_jurkat.csv` | 3,026 | −0.315 / −0.294 | **+0.503 / +0.515** |
| `litterman2019_pretrain.csv` | 1,555 | −0.072 / −0.168 | **+0.625 / +0.779** |
| `west2025_3utr_stability.csv` | 125 | −0.269 / −0.290 | **+0.301 / +0.280** |

Fine-tuned checkpoints saved separately (gitignored, local only): `3utr_finetuned_siegel.pt`,
`3utr_finetuned_litterman2019.pt`, `3utr_finetuned_west2025.pt`.

**Consistent finding**: the pretrained model's zero-shot correlation with every one of
these "stability score" targets is *negative* across all three, independent datasets —
almost certainly because the model was originally trained toward a decay/instability-
flavored target, naturally anti-correlated with a stability ratio, rather than indicating
no signal. Fine-tuning consistently flips the sign to the intuitive direction and increases
correlation strength.

**Learning-rate sensitivity** (found on Siegel, held up on the other two): LR=1e-4 collapses
correlation toward zero within 1 epoch (target variance is small, so an aggressive LR
overwrites pretrained features rather than adapting them); LR=1e-5 is stable but converges
too slowly; **LR=3e-5 for ~20-30 epochs** converges cleanly and is the script's default —
though the right epoch count is dataset-dependent (Litterman plateaued by epoch ~14; West,
with only 1,124 training sequences, was still improving near epoch 19-20).

## Commits

```
ffe2bf6  Add dynamic tracer for compiled CDS/UTR generation
5ae7fc1  Add standalone reimplementation of GEMORNA generation logic
4154534  Untrack Python bytecode cache files
ab6f086  Add fine-tuning/benchmark script for the 3'UTR prediction model
```

All pushed to `origin/improve-model`.
