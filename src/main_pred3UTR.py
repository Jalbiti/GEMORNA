#! /usr/bin/env python3

import argparse
import torch
import models.model_pred3UTR as model
from shared.helper import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence",  type=str, default=None, help="UTR sequence")
    parser.add_argument("--ckpt_path",   type=str, default=None, help="Path to load pre-trained models")

    args = parser.parse_args()

    args.embed_num = 10
    args.embed_dim = 256
    args.kernel_num = 200
    args.kernel_sizes = kernel_sizes_3UTR
    args.dropout = 0.1

    return args

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.sequence:
        raise ValueError("No input UTR sequence!")
    else:
        seq = args.sequence

    validate_sequence(seq)
    tokenized_seq = tokenize(seq)
    predictor = model.Model(args).to(device)
    if not args.ckpt_path:
        raise ValueError("No ckpt_path path!")
    else:
        predictor.load_state_dict(torch.load(args.ckpt_path, map_location=device), strict=True)
    
    predictor.eval()
    with torch.no_grad():
        pred = predictor(torch.tensor([tokenized_seq], device=device)).squeeze().cpu().numpy()

    print(f"\n3' UTR sequence & Prediction")
    print(f"{seq} {pred:.2f}\n")     

if __name__ == "__main__":
    main()
     
