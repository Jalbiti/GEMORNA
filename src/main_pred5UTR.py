#! /usr/bin/env python3

import argparse
import torch
import models.model_pred5UTR as model
from shared.helper import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence",  type=str, default=None, help="UTR sequence")
    parser.add_argument("--ckpt_path",   type=str, default=None, help="Path to load pre-trained models")

    args = parser.parse_args()

    args.embed_num = 10
    args.embed_dim = 64
    args.kernel_num = 128
    args.kernel_sizes = kernel_sizes_5UTR
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
        pred = predictor(torch.tensor([tokenized_seq + [vocab["[PAD]"]] * (100 - len(tokenized_seq))], device=device)).squeeze().cpu().numpy() 

    final_prediction = scale(pred)
    print(f"\n5' UTR sequence & Prediction")
    print(f"{seq} {final_prediction:.2f}\n")

if __name__ == "__main__":
    main()
    