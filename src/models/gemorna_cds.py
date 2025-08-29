import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
import platform
import platform
if platform.system() == "Darwin":
    from shared.libg2m import *
elif platform.system() == "Linux":
    from shared.mod_xzr01 import *
else:
    raise RuntimeError("Unsupported OS")
from utils.utils_cds import *


class Encoder(nn.Module):

    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, cnn_kernel_size, cnn_padding):

        super().__init__()
        self.scale = torch.sqrt(torch.tensor(hid_dim, dtype=torch.float32)).to(device)
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.cnn = nn.Conv1d(hid_dim, hid_dim, kernel_size=cnn_kernel_size, padding=cnn_padding)

    def forward(self, prot, prot_mask):
        batch_size, prot_len = prot.shape
        pos = torch.arange(0, prot_len, device=self.device).unsqueeze(0).expand(batch_size, prot_len)
        x = self.tok_embedding(prot) * self.scale + self.pos_embedding(pos)
        x = self.dropout(x)
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer(x, prot_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=max_length):

        super().__init__()
        self.scale = torch.sqrt(torch.tensor(hid_dim, dtype=torch.float32)).to(device)
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, cds, enc_prot, cds_mask, prot_mask):
        batch_size, cds_len = cds.shape
        pos = torch.arange(0, cds_len, device=self.device).unsqueeze(0).expand(batch_size, cds_len)
        x = self.tok_embedding(cds) * self.scale + self.pos_embedding(pos)
        x = self.dropout(x)
        for layer in self.layers:
            x, attention = layer(x, enc_prot, cds_mask, prot_mask)
        x = self.fc_out(x)
        return x, attention


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = FeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, prot, prot_mask):
        self_attention_output, _ = self.self_attention(prot, prot, prot, prot_mask)
        normed_attn = self.self_attn_layer_norm(prot + self.dropout(self_attention_output))
        feedforward_output = self.positionwise_feedforward(normed_attn)
        x = self.ff_layer_norm(normed_attn + self.dropout(feedforward_output))
        return x

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, device):
        super().__init__()
        self.hid_dim = hidden_dim
        self.n_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(device)

    def forward(self, q_input, k_input, v_input, attn_mask=None):
        batch_sz = q_input.size(0)
        q_proj = self.fc_q(q_input)
        k_proj = self.fc_k(k_input)
        v_proj = self.fc_v(v_input)

        def split_heads(tensor):
            return tensor.view(batch_sz, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        q_heads = split_heads(q_proj)
        k_heads = split_heads(k_proj)
        v_heads = split_heads(v_proj)

        attn_scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / self.scale
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        attention = torch.softmax(attn_scores, dim=-1)
        attention = self.dropout(attention)
        weighted_output = torch.matmul(attention, v_heads)
        merged = weighted_output.permute(0, 2, 1, 3).contiguous().view(batch_sz, -1, self.hid_dim)
        x = self.fc_o(merged)
        return x, attention

class FeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc_2(self.dropout(F.relu(self.fc_1(x))))


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = FeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cds, enc_prot, cds_mask, prot_mask):
        self_attn_output, _ = self.self_attention(cds, cds, cds, cds_mask)
        x = self.self_attn_layer_norm(cds + self.dropout(self_attn_output))
        enc_attn_output, attention = self.encoder_attention(x, enc_prot, enc_prot, prot_mask)
        x = self.enc_attn_layer_norm(x + self.dropout(enc_attn_output))
        ff_output = self.positionwise_feedforward(x)
        x = self.ff_layer_norm(x + self.dropout(ff_output))
        return x, attention
