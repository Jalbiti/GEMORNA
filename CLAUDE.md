# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

GEMORNA is an inference-only package (no training code included) for mRNA sequence design and
prediction, from the paper "Deep generative models design mRNA sequences with enhanced
translational capacity and stability" (Zhang et al., *Science*, 2025). It ships pretrained
checkpoints for two tasks:

- **Generation**: zero-shot design of CDS (from a protein sequence), and 5'/3' UTRs (by length
  class) — `src/generate.py`.
- **Prediction**: scoring an existing 5' or 3' UTR sequence — `src/main_pred5UTR.py` /
  `src/main_pred3UTR.py`.

## Environment setup

```
conda env create -f environment.yaml
conda activate gemorna
```

Key pinned deps: `python==3.10`, `torch==2.2.0`, `torchtext==0.6.0`, `numpy==1.26.4`,
`biopython==1.72`. Only `checkpoints/*.pt` is tracked via git-lfs (see `.gitattributes`) —
`git lfs install` before cloning. Vocab files (`vocab/*.pkl`) are regular git-tracked files, not
LFS.

All commands are run from the repo root with `src` as the script's own directory (scripts use
relative paths like `./vocab/prot_vocab.pkl`), e.g.:

```
python src/generate.py --mode cds --ckpt_path checkpoints/gemorna_cds.pt --protein_seq <AA_SEQ>
python src/generate.py --mode 5utr --ckpt_path checkpoints/gemorna_5utr.pt --utr_length {short,medium,long}
python src/generate.py --mode 3utr --ckpt_path checkpoints/gemorna_3utr.pt --utr_length {short,medium,long}
python src/main_pred5UTR.py --ckpt_path checkpoints/5utr.pt --sequence <5UTR_SEQ>
python src/main_pred3UTR.py --ckpt_path checkpoints/3utr.pt --sequence <3UTR_SEQ>
```

There is no test suite, linter, or build step in this repo — verification is manual, by running
the above commands against a checkpoint and checking output.

## Architecture — read this before editing `src/models/` or `src/generate.py`

The core model classes actually used at runtime (`CDS`, `UTR`, and the `max_length` constant used
by `models/gemorna_cds.py`) are **not defined in any `.py` file**. They are imported via
wildcard from a platform-specific compiled binary extension in `src/shared/`:

```python
if platform.system() == "Darwin":
    from shared.libg2m import *      # Mach-O arm64 bundle
elif platform.system() == "Linux":
    from shared.mod_xzr01 import *   # ELF x86-64 shared object
```

This import happens at the top of `src/generate.py` (and inside `src/models/gemorna_cds.py`)
*before* the local `models.*` wildcard imports, so `CDS(...)`/`UTR(...)` instances constructed
in `generate.py` and their `.gen(...)` methods resolve to the compiled module, not to the
`CDS_`/`UTR_` classes visibly defined in `src/utils/utils_cds.py` / `src/models/gemorna_utr.py`.
Those visible `_`-suffixed classes are **not used by the entry-point scripts** — treat them as
reference/partial versions only; the real generation/sampling logic (beam search, temperature
sampling, etc.) is compiled in and not visible from Python source. Practically, this means:

- You cannot trace or edit the actual `.gen()` implementation from source in this repo.
- Changes to `Encoder`/`Decoder`/`DecoderBlock`/`Attention` in `src/models/*.py` affect only the
  sub-layers that the compiled `CDS`/`UTR` wrapper classes compose — verify any change by
  actually running generation, since static reading of the `.py` files alone won't reveal the
  full forward path.
- `src/shared/helper.py` is a separate, plain-Python module (prediction-only helpers) — not part
  of the compiled-binary path.

### Task-specific pipelines

**CDS generation** (`src/generate.py`, mode `cds`):
`Encoder`/`Decoder`/`EncoderLayer`/`DecoderLayer`/`MultiHeadAttentionLayer` in
`src/models/gemorna_cds.py` implement a standard Transformer seq2seq (protein → codon sequence).
Config in `GEMORNA_CDS_Config` (`src/config.py`). Protein and CDS vocabularies are loaded from
`vocab/prot_vocab.pkl` / `vocab/cds_vocab.pkl`. `config.py` also holds `codon_dict` (all synonymous
codons per residue) and `codon_freq` (single most-frequent human codon per residue) used for
codon-level post-processing.

**5'/3' UTR generation** (`src/generate.py`, modes `5utr`/`3utr`):
`src/models/gemorna_utr.py` implements a GPT-style causal decoder-only Transformer
(`DecoderBlock`/`Attention`/`MLP`, with flash attention when available). Configs are
`GEMORNA_5UTR_Config` / `GEMORNA_3UTR_Config` in `src/config.py`. Sequences are tokenized as
**3-mers** (codon-length chunks), not single nucleotides — see the large `five_prime_utr_vocab` /
`three_prime_utr_vocab` dicts in `src/config.py` and `tokenize_seq` in `src/tokenization.py`,
which splits whitespace-separated tokens into 3-character chunks unless they're `<sos>`/`<eos>`.
Generation is conditioned on a coarse length class (`short`/`medium`/`long`) rather than an exact
length.

**5'/3' UTR prediction** (`main_pred5UTR.py` / `main_pred3UTR.py`):
Independent from the generation models — these use small CNN (`src/models/model_pred3UTR.py`,
multi-kernel-size TextCNN) and GRU (`src/models/model_pred5UTR.py`) regressors that score a raw
nucleotide sequence. Tokenization here is per-nucleotide (`src/shared/helper.py`: `vocab = {'[PAD]':
0, 'A':5, 'U':6, 'G':7, 'C':8, 'N':9}`), unlike the 3-mer tokenization used for UTR generation —
don't conflate the two vocabularies/tokenizers. 5' UTR prediction pads/truncates input to a fixed
length of 100 tokens before inference; the raw model output is de-normalized via
`scale()`/`scale_()` (hardcoded mean/std) to get the final score. 3' UTR prediction has no
fixed-length padding and no de-normalization.

### Import quirks

Scripts under `src/` use bare intra-package imports (`from config import *`, `import
models.model_pred3UTR as model`) rather than package-relative imports, and there is no
`src/__init__.py`. This only works because scripts are invoked as `python src/<script>.py`, which
puts `src/` on `sys.path`. Don't run these as `python -m src.generate` or import them from outside
`src/` without adjusting `sys.path`.
