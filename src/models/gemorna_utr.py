import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from config import *

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_head
        self.embed_dim = config.n_embd
        self.dropout_prob = config.dropout
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.bias)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.bias)
        self.attention_dropout = nn.Dropout(self.dropout_prob)
        self.residual_dropout = nn.Dropout(self.dropout_prob)
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            tril = torch.tril(torch.ones(config.block_size, config.block_size))
            tril = tril.view(1, 1, config.block_size, config.block_size)
            self.register_buffer("causal_mask", tril)

    def forward(self, hidden_states):
        batch_sz, seq_len, emb_dim = hidden_states.size()
        proj = self.c_attn(hidden_states)
        q_proj, k_proj, v_proj = proj.chunk(3, dim=-1)
        head_dim = emb_dim // self.n_heads

        def shape_proj(tensor):
            return tensor.view(batch_sz, seq_len, self.n_heads, head_dim).transpose(1, 2)

        queries = shape_proj(q_proj)
        keys = shape_proj(k_proj)
        values = shape_proj(v_proj)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=0, is_causal=True)
        else:
            scaling = 1.0 / math.sqrt(head_dim)
            sim_matrix = torch.matmul(queries, keys.transpose(-2, -1)) * scaling
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            sim_matrix = sim_matrix.masked_fill(mask == 0, float('-inf'))
            attention = F.softmax(sim_matrix, dim=-1)
            attention = self.attention_dropout(attention)
            y = torch.matmul(attention, values)

        y = y.transpose(1, 2).contiguous().reshape(batch_sz, seq_len, emb_dim)
        x = self.c_proj(y)
        x = self.residual_dropout(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x1 = self.ln_1(x)
        attn_output = self.attn(x1)
        x = x + attn_output
        x2 = self.ln_2(x)
        mlp_output = self.mlp(x2)
        x = x + mlp_output
        
        return x


