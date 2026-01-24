# smows

expert multi-scale network for spatial multi-omics with scGPT.
continue training.

**Author**: kyochilian  
**Email**: kyochilian@gmail.com  
**Date**: 2026.02

## Overview

smows is a deep learning framework that integrates spatial multi-omics data (e.g., RNA + Protein) using a combination of Graph Convolutional Networks (GCN) and scGPTs. The model performs multi-level fusion to enable effective clustering of spatial omics data.

## Features

- **Multi-level Fusion**: Combines GCN spatial/feature encoding with Transformer-based encoding
- **Modular Architecture**: Easy to modify models, datasets, and run ablation experiments
- **Flexible Configuration**: Centralized config system for quick parameter switching
- **WandB Integration**: Comprehensive experiment tracking and logging

## Requirements

smows is implemented in PyTorch and requires CUDA for optimal performance.

```bash
conda env create -f environment.yml
```

Or install manually:
```bash
pip install torch scanpy pandas numpy scipy sklearn munkres wandb
```

## Project Structure

```
SpaFusion/
├── config.py              # Centralized configuration
├── main.py                # Main training script
├── models/                # Modular model components
│   ├── __init__.py
│   ├── gcn.py            # GCN encoder/decoder
│   ├── transformer.py    # Transformer encoder/decoder
│   └── fusion.py         # Main SpaFusion model
├── datasets/              # Dataset utilities
│   └── __init__.py
├── processing.py          # Data preprocessing
├── utils.py               # Utility functions
├── evaluate.py            # Evaluation metrics
├── high_order_matrix.py   # High-order graph construction
└── encoder.py             # (Legacy compatibility layer)
```

## Usage

### Basic Usage

Run training on the default dataset (Human_lymph_node_D1):

```bash
python main.py
```

### Configuration Options

The configuration is centralized in `config.py`. You can:

1. **Switch datasets**: Modify the dataset name in `main.py` or create new dataset configs in `config.py`
2. **Change model architecture**: Update `MODEL_CONFIGS` in `config.py`
3. **Adjust hyperparameters**: Modify `TRAINING` configurations

### Command Line Arguments

```bash
python main.py --name D1 \
               --seed 0 \
               --spatial_k 9 \
               --adj_k 20 \
               --lambda1 1.0 \
               --lambda2 0.1 \
               --lr 0.002 \
               --pretrain_epoch 10000 \
               --train_epoch 350 \
               --use_wandb
```

### Running Ablation Studies

The config system supports easy ablation experiments:

```python
# In config.py, use predefined ablation configs:
TRAINING = {
    'ablation_no_kl': {...},          # Disable KL divergence
    'ablation_no_consistency': {...},  # Disable consistency loss
}
```

## Modular Design

### Adding a New Dataset

1. Add configuration to `config.py`:
```python
DATASETS = {
    'MyDataset': {
        'name': 'my_dataset_name',
        'data_path': './data/my_data/',
        'label_file': 'labels.csv',
        ...
    }
}
```

2. Run with: `python main.py --name MyDataset`

### Modifying the Model

All model components are in the `models/` directory:
- `models/gcn.py`: Modify GCN layers
- `models/transformer.py`: Modify Transformer layers  
- `models/fusion.py`: Modify fusion strategy

### Switching Components

Easy to swap or disable components for ablation:
- Remove Transformer: Modify `models/fusion.py` forward method
- Change GCN depth: Update layer counts in `models/gcn.py`
- Adjust fusion weights: Modify `emb_fusion()` in `models/fusion.py`

## Results

Results are saved in `./results/{dataset_name}/{timestamp}/`:
- Training logs
- Performance metrics (CSV)
- Predicted labels
- Latent embeddings
- Spatial coordinates

---

## SpaFusion Integration

This project includes a complete integration of the [SpaFusion](https://github.com/polarisChen/SpaFusion) network.

### SpaFusion Usage

#### Running SpaFusion on A1 Dataset:
```bash
python run_spafusion.py --name Human_Lymph_Node_A1
```

#### Running SpaFusion on D1 Dataset:
```bash
python run_spafusion.py --name Human_Lymph_Node_D1
```

#### Adjustable Hyperparameters:
```bash
python run_spafusion.py --name Human_Lymph_Node_D1 \
    --seed 0 \
    --spatial_k 9 \
    --adj_k 20 \
    --pretrain_epoch 5000 \
    --train_epoch 2500 \
    --lr 1e-3 \
    --lambda1 1.0 \
    --lambda2 0.1 \
    --enc_dim1 256 \
    --enc_dim2 128 \
    --latent_dim 20 \
    --dropout 0.1
```

#### All Available Options:
```bash
python run_spafusion.py --help
```

### SpaFusion Visualization

Generate visualizations for SpaFusion results:
```bash
python visualize_spafusion.py --results_dir results/SpaFusion_Human_Lymph_Node_D1_YYYYMMDD_HHMMSS
```

### SpaFusion Project Structure
```
SpaMICS/
├── spafusion/                    # SpaFusion core modules (DO NOT MODIFY)
│   ├── __init__.py
│   ├── encoder.py                # GCN + Transformer encoders
│   ├── high_order_matrix.py      # High-order graph construction
│   ├── spafusion_processing.py   # Data preprocessing
│   ├── spafusion_utils.py        # Utility functions
│   └── spafusion_evaluate.py     # Evaluation metrics
├── run_spafusion.py              # Main SpaFusion training script
├── spafusion_opt.py              # SpaFusion configuration/hyperparameters
├── visualize_spafusion.py        # SpaFusion visualization
└── pretrain/                     # Pretrained models directory
```

### SpaFusion Results Structure
```
results/SpaFusion_{dataset}_{timestamp}/
├── summary.json            # Configuration and metrics
├── training_log.txt        # Complete training log
├── pretrain_model.pth      # Pretrained model weights
├── pretrain_losses.npy     # Pretraining loss history
├── best_model.pth          # Best model weights
├── best_predictions.npy    # Best clustering predictions
├── final_model.pth         # Final model weights
├── final_predictions.npy   # Final clustering predictions
├── latent_features.npy     # Latent representations
├── train_losses.npy        # Training loss history
└── pre_adj/                # Preprocessed adjacency matrices
```

---

## Evaluation Metrics

The model reports:
- **ACC**: Clustering Accuracy
- **F1**: F1-score
- **NMI**: Normalized Mutual Information
- **ARI**: Adjusted Rand Index
- **AMI**: Adjusted Mutual Information
- **VMS**: V-Measure Score
- **FMS**: Fowlkes-Mallows Score

## Citation

If you use SpaFusion in your research, please cite:

```
[Citation details to be added]
```

## License

the rep follows MIT License.

## Contact

For questions or issues, please contact:
- **Author**: kyochilian
- **Email**: kyochilian@gmail.com
