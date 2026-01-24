# -*- coding:utf-8 -*-
"""
SpaFusion Configuration
All hyperparameters for SpaFusion model
"""

import argparse

parser = argparse.ArgumentParser(description="SpaFusion Model Configuration")

# Dataset settings
parser.add_argument('--name', type=str, default='Human_Lymph_Node_A1', help='Dataset name')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
parser.add_argument('--seed', type=int, default=0, help='Random seed')

# Graph construction parameters
parser.add_argument('--spatial_k', type=int, default=9, help='Number of spatial neighbors')
parser.add_argument('--adj_k', type=int, default=20, help='Number of feature neighbors')

# Loss weights
parser.add_argument('--lambda1', type=float, default=1, help='KL divergence loss weight')
parser.add_argument('--lambda2', type=float, default=0.1, help='Dense loss weight')
parser.add_argument('--weight_list', type=list, default=[1, 1, 1, 1, 1, 1], 
                    help='Weight list for reconstruction losses [ae1, ae2, ae3, ae4, x1, x2]')

# Training parameters
parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
parser.add_argument('--pretrain_epoch', type=int, default=10000, help='Pretrain epochs')
parser.add_argument('--train_epoch', type=int, default=350, help='Training epochs')
parser.add_argument('--skip_pretrain', type=bool, default=False, help='Skip pretraining phase')

# Model architecture parameters
parser.add_argument('--enc_dim1', type=int, default=256, help='First encoder dimension')
parser.add_argument('--enc_dim2', type=int, default=128, help='Second encoder dimension')
parser.add_argument('--dec_dim1', type=int, default=128, help='First decoder dimension')
parser.add_argument('--dec_dim2', type=int, default=256, help='Second decoder dimension')
parser.add_argument('--latent_dim', type=int, default=20, help='Latent dimension')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--num_heads1', type=int, default=1, help='Number of attention heads for view1')
parser.add_argument('--num_heads2', type=int, default=1, help='Number of attention heads for view2')

# Results saving
parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')

args = parser.parse_args(args=[])
