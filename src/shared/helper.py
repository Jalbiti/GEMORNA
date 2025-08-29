#! /usr/bin/env python3

import pandas as pd
from sklearn import preprocessing
import numpy as np


def validate_sequence(seq):
    valid_set = set('ACGUN')
    seq = seq.upper().replace("T", "U")
    if any(base not in valid_set for base in seq):
        raise ValueError(f"Sequence at index {idx} contains invalid characters: {seq}")

def scale_(z, mean, std):
    return z * std + mean

def tokenize(seq):
    seq = seq.upper().replace("T", "U")
    tokens = [vocab[c] for c in seq]

    return tokens

def scale(pred):
    final_prediction = scale_(pred, 5.19937892, 1.40592675)
    return final_prediction


vocab = {'[PAD]': 0, 'A': 5, 'U': 6, 'G': 7, 'C': 8, 'N': 9}
kernel_sizes_5UTR = [5, 10, 30, 50]
kernel_sizes_3UTR = [2, 4, 6, 8, 10]
