# -*- coding:utf-8 -*-
"""
Author: kyochilian
Date: 2026.02

Transformer Components
This module contains the Transformer encoder and decoder implementations.
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Multi-head Self Attention mechanism"""
    
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )

        out = self.fc_out(out)
        return out


class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer"""
    
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.self_attention(x, x, x, mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(query)
        return out


class TransformerEncoder(nn.Module):
    """Transformer Encoder with positional embedding"""
    
    def __init__(self, embed_size, laten_size, num_layers, heads, forward_expansion, dropout, max_length):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [EncoderLayer(embed_size, heads, forward_expansion, dropout) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(embed_size, laten_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length, _ = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        x = self.dropout((x + self.position_embedding(positions)))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.linear(x)
        return x


class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer"""
    
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.self_attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(query)
        return out

 
class TransformerDecoder(nn.Module):
    """Transformer Decoder with positional embedding"""
    
    def __init__(self, input_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length):
        super(TransformerDecoder, self).__init__()
        self.embed_size = embed_size
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(embed_size, heads, forward_expansion, dropout) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(embed_size, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length, _ = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        x = self.dropout((x + self.position_embedding(positions)))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        x = self.linear(x)
        return x


# For backward compatibility
trans_encoder = TransformerEncoder
trans_decoder = TransformerDecoder
