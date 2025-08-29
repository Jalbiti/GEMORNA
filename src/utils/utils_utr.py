import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from config import *
from models.gemorna_utr import *


class UTR_(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),  
            'wpe': nn.Embedding(config.block_size, config.n_embd), 
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
            'ln_f': LayerNorm(config.n_embd, bias=config.bias),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=MEAN, std=STD)
            if getattr(module, "bias", None) is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=MEAN, std=STD)

    def forward(self, input_ids, targets=None):
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        pos_indices = torch.arange(seq_len, dtype=torch.long, device=device)
        token_embeds = self.transformer['wte'](input_ids)
        position_embeds = self.transformer['wpe'](pos_indices)
        hidden_states = token_embeds + position_embeds
        hidden_states = self.transformer['drop'](hidden_states)

        for block in self.transformer['h']:
            hidden_states = block(hidden_states)
        hidden_states = self.transformer['ln_f'](hidden_states)

        logits = self.lm_head(hidden_states[:, [-1], :])

        return logits, None
