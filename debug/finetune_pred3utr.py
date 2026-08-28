#!/usr/bin/env python3
"""
Fine-tune GEMORNA's 3' UTR PREDICTION model (checkpoints/3utr.pt, the CNN
regressor behind src/main_pred3UTR.py -- NOT the generative gemorna_3utr.pt
model) on a labeled 3' UTR stability/activity dataset, and benchmark it
against the untouched pretrained checkpoint on an identical held-out split.

Generalized from the first run of this comparison (siegel2021_jurkat.csv) to
also cover west2025_3utr_stability.csv and litterman2019_pretrain.csv --
column names and split logic differ per dataset, so both are CLI flags rather
than hardcoded.

Architecture/preprocessing exactly mirrors src/main_pred3UTR.py and
src/shared/helper.py: models.model_pred3UTR.Model with
(embed_num=10, embed_dim=256, kernel_num=200, kernel_sizes=[2,4,6,8,10],
dropout=0.1), per-nucleotide tokenization (shared.helper.tokenize, T->U mapped
to the model's RNA vocab), no fixed-length padding. Batches are built by
grouping sequences of identical tokenized length together (length-bucketing)
instead of padding, to avoid a Conv+global-max-pool model's pad embedding
leaking into the pooled features -- this degenerates to one batch shape per
dataset for litterman2019 (every sequence is exactly 70 nt).

Note on output scale: src/main_pred3UTR.py never rescales the 3' UTR model's
raw output (unlike the 5' UTR predictor's shared.helper.scale()) -- so the
pretrained model's raw predictions are on whatever scale its original
(undocumented) training target used, which may not match a given dataset's
target scale or sign at all. That makes Pearson/Spearman correlation
(scale- and sign-invariant... well, sign-invariant only up to |r|) the fair
"before vs after" comparison; MSE is reported too but is only meaningful for
the fine-tuned model itself, not as a cross-model comparison. On
siegel2021_jurkat.csv the pretrained model's zero-shot correlation was real
but sign-flipped (Pearson r=-0.32) relative to the target's convention --
worth checking for on any new dataset too, rather than assuming a negative r
means "no signal".

Learning-rate note (from the siegel2021_jurkat.csv run): 1e-4 was too
aggressive and collapsed correlation toward zero within 1 epoch (these
stability targets tend to have small variance, so a large LR overwrites the
pretrained CNN features rather than adapting them); 1e-5 was stable but still
climbing after 12 epochs (under-converged); 3e-5 for 30 epochs converged
cleanly (correlation climbed monotonically then plateaued around epoch 26-28).
Used as the default here, but re-check the per-epoch trend for each new
dataset -- optimal LR/epoch count is dataset-size- and target-variance-
dependent, not something to assume transfers exactly.

Usage (run from the GEMORNA repo root, inside the `gemorna` conda env):

    python debug/finetune_pred3utr.py --csv_path siegel2021_jurkat.csv \
        --target_col ratios_T4T0_GC_resid --out_ckpt checkpoints/3utr_finetuned_siegel.pt

    python debug/finetune_pred3utr.py --csv_path west2025_3utr_stability.csv \
        --target_col stability_score --out_ckpt checkpoints/3utr_finetuned_west2025.pt

    python debug/finetune_pred3utr.py --csv_path litterman2019_pretrain.csv \
        --target_col stability_score --split_col split \
        --out_ckpt checkpoints/3utr_finetuned_litterman2019.pt
"""
import argparse
import copy
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import models.model_pred3UTR as model_pred3utr  # noqa: E402
from shared.helper import kernel_sizes_3UTR, tokenize, validate_sequence  # noqa: E402


def build_args():
    class Args:
        pass

    args = Args()
    args.embed_num = 10
    args.embed_dim = 256
    args.kernel_num = 200
    args.kernel_sizes = kernel_sizes_3UTR
    args.dropout = 0.1
    return args


def load_pred3utr_model(ckpt_path, device):
    model = model_pred3utr.Model(build_args())
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    model.to(device)
    return model


def detect_seq_col(df, seq_col):
    if seq_col:
        return seq_col
    for candidate in ("seq", "sequence"):
        if candidate in df.columns:
            return candidate
    raise SystemExit(f"Could not auto-detect sequence column; pass --seq_col. Columns: {list(df.columns)}")


VALID_CHARS = set("ACGTUNacgtun")


def load_dataset(csv_path, seq_col, target_col, split_col):
    df = pd.read_csv(csv_path)
    seq_col = detect_seq_col(df, seq_col)
    df = df[df[target_col].notna() & df[seq_col].notna()].reset_index(drop=True)

    valid_mask = df[seq_col].apply(lambda s: set(s) <= VALID_CHARS)
    n_dropped = (~valid_mask).sum()
    if n_dropped:
        print(f"Dropping {n_dropped} row(s) with non-nucleotide characters in '{seq_col}' "
              f"(e.g. {df.loc[~valid_mask, seq_col].iloc[0][:40]!r})", flush=True)
    df = df[valid_mask].reset_index(drop=True)

    seqs, targets = [], []
    for seq, y in zip(df[seq_col], df[target_col]):
        validate_sequence(seq)
        seqs.append(tokenize(seq))
        targets.append(float(y))

    split_labels = df[split_col].tolist() if split_col else None
    return seqs, targets, split_labels


def make_length_bucketed_batches(seqs, batch_size, shuffle, seed):
    buckets = defaultdict(list)
    for i, s in enumerate(seqs):
        buckets[len(s)].append(i)
    rng = random.Random(seed)
    batches = []
    for idxs in buckets.values():
        idxs = list(idxs)
        if shuffle:
            rng.shuffle(idxs)
        for i in range(0, len(idxs), batch_size):
            batches.append(idxs[i : i + batch_size])
    if shuffle:
        rng.shuffle(batches)
    return batches


@torch.no_grad()
def evaluate(model, seqs, targets, device, batch_size=256):
    model.eval()
    preds, ys = [], []
    for batch_idxs in make_length_bucketed_batches(seqs, batch_size, shuffle=False, seed=0):
        x = torch.tensor([seqs[i] for i in batch_idxs], dtype=torch.long, device=device)
        pred = model(x).squeeze(-1).cpu().numpy()
        preds.extend(np.atleast_1d(pred).tolist())
        ys.extend(targets[i] for i in batch_idxs)
    preds, ys = np.array(preds), np.array(ys)
    mse = float(np.mean((preds - ys) ** 2))
    pearson_r = float(pearsonr(preds, ys)[0])
    spearman_r = float(spearmanr(preds, ys)[0])
    return {"mse": mse, "pearson_r": pearson_r, "spearman_r": spearman_r, "n": len(ys)}


def train(model, train_seqs, train_targets, val_seqs, val_targets, device, epochs, batch_size, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = copy.deepcopy(model.state_dict())
    best_val_mse = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        for batch_idxs in make_length_bucketed_batches(train_seqs, batch_size, shuffle=True, seed=epoch):
            x = torch.tensor([train_seqs[i] for i in batch_idxs], dtype=torch.long, device=device)
            y = torch.tensor([train_targets[i] for i in batch_idxs], dtype=torch.float32, device=device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_idxs)
            n_seen += len(batch_idxs)

        val_metrics = evaluate(model, val_seqs, val_targets, device)
        print(
            f"epoch {epoch:2d}/{epochs}  train_mse={total_loss / n_seen:.4f}  "
            f"val_mse={val_metrics['mse']:.4f}  val_pearson_r={val_metrics['pearson_r']:.4f}  "
            f"val_spearman_r={val_metrics['spearman_r']:.4f}",
            flush=True,
        )
        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--seq_col", default=None, help="auto-detected as 'seq' or 'sequence' if omitted")
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--split_col", default=None, help="if the CSV has a train/val split column, use it instead of --val_frac")
    parser.add_argument("--ckpt_path", default="checkpoints/3utr.pt", help="pretrained checkpoint to start from")
    parser.add_argument("--out_ckpt", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.csv_path} (target={args.target_col}) ...", flush=True)
    seqs, targets, split_labels = load_dataset(args.csv_path, args.seq_col, args.target_col, args.split_col)
    print(f"{len(seqs)} rows with non-null seq/target", flush=True)

    if split_labels is not None:
        train_idx = [i for i, s in enumerate(split_labels) if s == "train"]
        val_idx = [i for i, s in enumerate(split_labels) if s != "train"]
    else:
        idx = np.arange(len(seqs))
        train_idx, val_idx = train_test_split(idx, test_size=args.val_frac, random_state=args.seed)

    train_seqs = [seqs[i] for i in train_idx]
    train_targets = [targets[i] for i in train_idx]
    val_seqs = [seqs[i] for i in val_idx]
    val_targets = [targets[i] for i in val_idx]
    print(f"train={len(train_seqs)}  val={len(val_seqs)}", flush=True)

    print(f"\nLoading pretrained checkpoint from {args.ckpt_path} ...", flush=True)
    pretrained_model = load_pred3utr_model(args.ckpt_path, device)
    baseline_metrics = evaluate(pretrained_model, val_seqs, val_targets, device)

    print("\nFine-tuning ...", flush=True)
    finetune_model = load_pred3utr_model(args.ckpt_path, device)
    finetune_model = train(
        finetune_model, train_seqs, train_targets, val_seqs, val_targets, device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
    finetuned_metrics = evaluate(finetune_model, val_seqs, val_targets, device)

    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)
    torch.save(finetune_model.state_dict(), args.out_ckpt)

    print(f"\n{'=' * 72}")
    print(f"Benchmark on held-out split (n={baseline_metrics['n']}, target={args.target_col})")
    print(f"{'=' * 72}")
    print(f"{'':20s}{'MSE':>12s}{'Pearson r':>14s}{'Spearman r':>14s}")
    print(f"{'pretrained':20s}{baseline_metrics['mse']:12.4f}{baseline_metrics['pearson_r']:14.4f}{baseline_metrics['spearman_r']:14.4f}")
    print(f"{'fine-tuned':20s}{finetuned_metrics['mse']:12.4f}{finetuned_metrics['pearson_r']:14.4f}{finetuned_metrics['spearman_r']:14.4f}")
    print(f"\nFine-tuned checkpoint saved to {args.out_ckpt}")


if __name__ == "__main__":
    main()
