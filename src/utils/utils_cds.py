import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class CDS_(nn.Module):
    def __init__(self, encoder, decoder, prot_pad_idx, cds_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prot_pad_idx = prot_pad_idx
        self.cds_pad_idx = cds_pad_idx
        self.device = device

    def make_prot_mask(self, prot):
        return (prot != self.prot_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_cds_mask(self, cds):
        cds_pad_mask = (cds != self.cds_pad_idx).unsqueeze(1).unsqueeze(2)
        cds_len = cds.shape[1]
        cds_sub_mask = torch.tril(torch.ones((cds_len, cds_len), device=self.device)).bool()
        cds_mask = cds_pad_mask & cds_sub_mask
        return cds_mask

    def forward(self, prot, cds):
        protein_mask = self.make_prot_mask(protein_input)
        cds_mask = self.make_cds_mask(cds_input)
        encoded_protein = self.encoder(protein_input, protein_mask)
        decoded_output, attn_weights = self.decoder(
            cds_input, encoded_protein, cds_mask, protein_mask
        )
        return decoded_output, attn_weights
        
