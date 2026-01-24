"""
SpaFusion Configuration and Hyperparameters
All hyperparameters can be adjusted via command line arguments
"""

import argparse

parser = argparse.ArgumentParser(
    description='SpaFusion: A multi-level fusion model for clustering spatial multi-omics data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Dataset settings
parser.add_argument('--name', type=str, default='Human_Lymph_Node_D1',
                    choices=['Human_Lymph_Node_A1', 'Human_Lymph_Node_D1'],
                    help='Dataset name')
parser.add_argument('--data_path', type=str, default='./data/10X',
                    help='Path to data directory')

# Device settings
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Device to use (cuda:0, cuda:1, cpu)')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed for reproducibility')

# Graph construction parameters
parser.add_argument('--spatial_k', type=int, default=9,
                    help='Number of neighbors for spatial graph')
parser.add_argument('--adj_k', type=int, default=20,
                    help='Number of neighbors for feature graph')

# Training parameters
parser.add_argument('--pretrain_epoch', type=int, default=10000,
                    help='Number of pretraining epochs')
parser.add_argument('--train_epoch', type=int, default=2500,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')

# Loss weight parameters
parser.add_argument('--lambda1', type=float, default=1.0,
                    help='Weight for KL divergence loss')
parser.add_argument('--lambda2', type=float, default=0.1,
                    help='Weight for dense loss')
parser.add_argument('--weight_ae1', type=float, default=1.0,
                    help='Weight for spatial adj1 reconstruction loss')
parser.add_argument('--weight_ae2', type=float, default=1.0,
                    help='Weight for feature adj1 reconstruction loss')
parser.add_argument('--weight_ae3', type=float, default=1.0,
                    help='Weight for spatial adj2 reconstruction loss')
parser.add_argument('--weight_ae4', type=float, default=1.0,
                    help='Weight for feature adj2 reconstruction loss')
parser.add_argument('--weight_x1', type=float, default=1.0,
                    help='Weight for omics1 feature reconstruction loss')
parser.add_argument('--weight_x2', type=float, default=1.0,
                    help='Weight for omics2 feature reconstruction loss')

# Model architecture parameters (advanced)
parser.add_argument('--enc_dim1', type=int, default=256,
                    help='First encoder hidden dimension')
parser.add_argument('--enc_dim2', type=int, default=128,
                    help='Second encoder hidden dimension')
parser.add_argument('--dec_dim1', type=int, default=128,
                    help='First decoder hidden dimension')
parser.add_argument('--dec_dim2', type=int, default=256,
                    help='Second decoder hidden dimension')
parser.add_argument('--latent_dim', type=int, default=20,
                    help='Latent representation dimension')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate')
parser.add_argument('--num_layers', type=int, default=2,
                    help='Number of transformer layers')
parser.add_argument('--num_heads1', type=int, default=1,
                    help='Number of attention heads for omics1')
parser.add_argument('--num_heads2', type=int, default=1,
                    help='Number of attention heads for omics2')

# Clustering settings
parser.add_argument('--n_clusters', type=int, default=None,
                    help='Number of clusters (auto-detected if None)')

# Training behavior
parser.add_argument('--num_runs', type=int, default=1,
                    help='Number of training runs')
parser.add_argument('--show', action='store_true',
                    help='Show detailed dataset information')

args = parser.parse_args()


def get_weight_list():
    """Get weight list from arguments"""
    return [
        args.weight_ae1,
        args.weight_ae2,
        args.weight_ae3,
        args.weight_ae4,
        args.weight_x1,
        args.weight_x2
    ]


def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("SpaFusion Configuration")
    print("=" * 60)
    print(f"Dataset        : {args.name}")
    print(f"Data path      : {args.data_path}")
    print(f"Device         : {args.device}")
    print(f"Seed           : {args.seed}")
    print("-" * 60)
    print("Graph Parameters:")
    print(f"  spatial_k    : {args.spatial_k}")
    print(f"  adj_k        : {args.adj_k}")
    print("-" * 60)
    print("Training Parameters:")
    print(f"  pretrain_epoch : {args.pretrain_epoch}")
    print(f"  train_epoch    : {args.train_epoch}")
    print(f"  learning rate  : {args.lr:.0e}")
    print(f"  lambda1        : {args.lambda1}")
    print(f"  lambda2        : {args.lambda2}")
    print(f"  weight_list    : {get_weight_list()}")
    print("-" * 60)
    print("Model Architecture:")
    print(f"  enc_dim1       : {args.enc_dim1}")
    print(f"  enc_dim2       : {args.enc_dim2}")
    print(f"  dec_dim1       : {args.dec_dim1}")
    print(f"  dec_dim2       : {args.dec_dim2}")
    print(f"  latent_dim     : {args.latent_dim}")
    print(f"  dropout        : {args.dropout}")
    print(f"  num_layers     : {args.num_layers}")
    print(f"  num_heads1     : {args.num_heads1}")
    print(f"  num_heads2     : {args.num_heads2}")
    print("=" * 60)
