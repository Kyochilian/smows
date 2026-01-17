# -*- coding:utf-8 -*-
"""
Author: kyochilian
Date: 2026.02

Main Model Module
Package initialization - exports all model components
"""

from .gcn import GraphConvolutionLayer, GCNEncoder, GCNDecoder
from .transformer import (
    SelfAttention,
    EncoderLayer,
    DecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    trans_encoder,
    trans_decoder,
)
from .fusion import q_distribution, GCNAutoencoder, Spafusion

__all__ = [
    # GCN components
    "GraphConvolutionLayer",
    "GCNEncoder",
    "GCNDecoder",
    # Transformer components
    "SelfAttention",
    "EncoderLayer",
    "DecoderLayer",
    "TransformerEncoder",
    "TransformerDecoder",
    "trans_encoder",
    "trans_decoder",
    # Fusion model
    "q_distribution",
    "GCNAutoencoder",
    "Spafusion",
]
