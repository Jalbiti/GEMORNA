#! /usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, args):

        super(Model, self).__init__()
        self.args = args    

        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, args.kernel_num, (kernel_siz, args.embed_dim)) for kernel_siz in args.kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(args.kernel_sizes) * args.kernel_num, 1)
            

    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1) 
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] 
        x = torch.cat(x, 1)
        x = self.dropout(x)  
        logit = self.fc1(x)  
        
        return logit
