# -*- coding:utf-8 -*-
"""
Author: kyochilian
Date: 2026.02

SpaFusion Configuration File
This file contains all configuration options for SpaFusion.
Modifying configurations here allows easy switching between datasets, models, and experimental setups.
"""

import os


class Config:
    """
    Centralized configuration for SpaFusion
    """
    
    # Dataset Configurations
    DATASETS = {
        'D1': {
            'name': 'Human_lymph_node_D1',
            'data_path': './data/',
            'label_file': 'D1_annotation_labels.csv',
            'rna_file': 'adata_RNA.h5ad',
            'protein_file': 'adata_ADT.h5ad',
            'view1': 'RNA',
            'view2': 'Protein',
            'has_labels': True,
            # Paper baseline metrics
            'paper_metrics': {
                'ARI': 0.351,
                'NMI': 0.384,
                'ACC': 0.599,
                'AMI': 0.379,
                'F1': 0.323
            }
        },
        'A1': {
            'name': 'Dataset_A1',
            'data_path': './data/',
            'label_file': 'A1_labels.csv',
            'rna_file': 'adata_RNA_A1.h5ad',
            'protein_file': 'adata_ADT_A1.h5ad',
            'view1': 'RNA',
            'view2': 'Protein',
            'has_labels': True,
            'paper_metrics': {
                'ARI': 0.319,
                'NMI': 0.399,
                'ACC': 0.629,
                'AMI': 0.396,
                'F1': 0.337
            }
        },
        # Add more datasets here as needed
    }
    
    # Model Architecture Configurations
    MODEL_CONFIGS = {
        'default': {
            'enc_dim1': 256,
            'enc_dim2': 128,
            'dec_dim1': 128,
            'dec_dim2': 256,
            'latent_dim': 20,
            'dropout': 0.1,
            'num_layers': 2,
            'num_heads1': 1,
            'num_heads2': 1,
        },
        # Add alternative model configurations for ablation studies
        'smaller': {
            'enc_dim1': 128,
            'enc_dim2': 64,
            'dec_dim1': 64,
            'dec_dim2': 128,
            'latent_dim': 10,
            'dropout': 0.1,
            'num_layers': 2,
            'num_heads1': 1,
            'num_heads2': 1,
        }
    }
    
    # Training Hyperparameters
    TRAINING = {
        'default': {
            'pretrain_epoch': 10000,
            'train_epoch': 350,
            'lr': 2e-3,
            'lambda1': 1.0,  # KL divergence weight
            'lambda2': 0.1,  # Consistency loss weight
            'weight_list': [1, 1, 1, 1, 1, 1],  # Reconstruction loss weights
            'num_runs': 10,  # Number of training runs for averaging
        },
        # Ablation study configurations
        'ablation_no_kl': {
            'pretrain_epoch': 10000,
            'train_epoch': 350,
            'lr': 2e-3,
            'lambda1': 0.0,  # Disable KL divergence
            'lambda2': 0.1,
            'weight_list': [1, 1, 1, 1, 1, 1],
            'num_runs': 10,
        },
        'ablation_no_consistency': {
            'pretrain_epoch': 10000,
            'train_epoch': 350,
            'lr': 2e-3,
            'lambda1': 1.0,
            'lambda2': 0.0,  # Disable consistency loss
            'weight_list': [1, 1, 1, 1, 1, 1],
            'num_runs': 10,
        }
    }
    
    # Graph Construction Parameters
    GRAPH_PARAMS = {
        'default': {
            'spatial_k': 9,   # Number of spatial neighbors
            'adj_k': 20,      # Number of feature neighbors
        }
    }
    
    # WandB Logging Configuration
    WANDB_CONFIG = {
        'enabled': True,
        'project_name': 'SpaFusion',
        'entity': None,  # Set to your WandB username/team if needed
    }
    
    # Device Configuration
    DEVICE = 'cuda:0' if __name__ == '__main__' else 'cpu'
    
    # Random Seed
    SEED = 0
    
    # Paths
    PRETRAIN_DIR = './pretrain'
    RESULT_DIR = './results'
    ADJ_DIR = './pre_adj'
    
    @classmethod
    def get_dataset_config(cls, dataset_name):
        """Get configuration for a specific dataset"""
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(cls.DATASETS.keys())}")
        return cls.DATASETS[dataset_name]
    
    @classmethod
    def get_model_config(cls, config_name='default'):
        """Get model architecture configuration"""
        if config_name not in cls.MODEL_CONFIGS:
            raise ValueError(f"Model config '{config_name}' not found. Available configs: {list(cls.MODEL_CONFIGS.keys())}")
        return cls.MODEL_CONFIGS[config_name]
    
    @classmethod
    def get_training_config(cls, config_name='default'):
        """Get training hyperparameters"""
        if config_name not in cls.TRAINING:
            raise ValueError(f"Training config '{config_name}' not found. Available configs: {list(cls.TRAINING.keys())}")
        return cls.TRAINING[config_name]
    
    @classmethod
    def get_graph_config(cls, config_name='default'):
        """Get graph construction parameters"""
        if config_name not in cls.GRAPH_PARAMS:
            raise ValueError(f"Graph config '{config_name}' not found. Available configs: {list(cls.GRAPH_PARAMS.keys())}")
        return cls.GRAPH_PARAMS[config_name]


def get_full_config(dataset='D1', model='default', training='default', graph='default', **overrides):
    """
    Get complete configuration by combining different config sections
    
    Args:
        dataset: Dataset name
        model: Model configuration name
        training: Training configuration name
        graph: Graph configuration name
        **overrides: Additional parameters to override
    
    Returns:
        dict: Complete configuration dictionary
    """
    config = {}
    
    # Get component configurations
    dataset_config = Config.get_dataset_config(dataset)
    model_config = Config.get_model_config(model)
    training_config = Config.get_training_config(training)
    graph_config = Config.get_graph_config(graph)
    
    # Merge all configurations
    config.update(dataset_config)
    config.update(model_config)
    config.update(training_config)
    config.update(graph_config)
    
    # Add global settings
    config['seed'] = Config.SEED
    config['device'] = Config.DEVICE
    config['pretrain_dir'] = Config.PRETRAIN_DIR
    config['result_dir'] = Config.RESULT_DIR
    config['adj_dir'] = Config.ADJ_DIR
    
    # WandB settings
    config['use_wandb'] = Config.WANDB_CONFIG['enabled']
    config['wandb_project'] = Config.WANDB_CONFIG['project_name']
    
    # Apply any overrides
    config.update(overrides)
    
    return config


if __name__ == '__main__':
    # Example usage
    config = get_full_config(dataset='D1')
    print("Configuration loaded successfully!")
    print(f"Dataset: {config['name']}")
    print(f"Model latent_dim: {config['latent_dim']}")
    print(f"Training epochs: {config['train_epoch']}")
