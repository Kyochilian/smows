# -*- coding:utf-8 -*-
"""
SpaFusion: A multi-level fusion model for clustering spatial multi-omics data
Integrated into SpaMICS project
"""

from .encoder import GCNAutoencoder
from .high_order_matrix import process_adjacency_matrix
from .spafusion_utils import (
    setup_seed, adjacent_matrix_preprocessing, norm_adj, 
    target_distribution, distribution_loss, assignment, clustering
)
from .spafusion_processing import load_data as spafusion_load_data
from .spafusion_evaluate import eva, cluster_acc

__all__ = [
    'GCNAutoencoder',
    'process_adjacency_matrix',
    'setup_seed',
    'adjacent_matrix_preprocessing',
    'norm_adj',
    'target_distribution',
    'distribution_loss',
    'assignment',
    'clustering',
    'spafusion_load_data',
    'eva',
    'cluster_acc'
]
