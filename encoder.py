# -*- coding:utf-8 -*-
"""
Author: kyochilian
Date: 2026.02

DEPRECATED: This file is kept for backward compatibility.
Please import directly from the models package instead:
    from models import GCNEncoder, GCNDecoder, TransformerEncoder, etc.
    
This module now serves as a compatibility layer that imports and re-exports
all model components from the refactored models/ package.
"""

# Import all components from the new modular structure
from models.gcn import GraphConvolutionLayer, GCNEncoder, GCNDecoder
from models.transformer import (
    SelfAttention,
    EncoderLayer, DecoderLayer,
    TransformerEncoder, TransformerDecoder,
    trans_encoder, trans_decoder
)
from models.fusion import q_distribution, GCNAutoencoder, Spafusion

# Re-export everything for backward compatibility
__all__ = [
    # GCN components
    'GraphConvolutionLayer',
    'GCNEncoder',
    'GCNDecoder',
    # Transformer components
    'SelfAttention',
    'EncoderLayer',
    'DecoderLayer',
    'TransformerEncoder',
    'TransformerDecoder',
    'trans_encoder',
    'trans_decoder',
    # Fusion model
   'q_distribution',
    'GCNAutoencoder',
    'Spafusion',
]
