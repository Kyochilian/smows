# smows Usage Guide

## Quick Start

### 1. Running with Default Settings

```bash
python main.py
```

This will run SpaFusion on the default dataset (Human_lymph_node_D1) with default parameters.

### 2. Customizing Parameters

```bash
python main.py --name D1 \
               --seed 0 \
               --spatial_k 9 \
               --adj_k 20 \
               --lambda1 1.0 \
               --lambda2 0.1 \
               --lr 0.002 \
               --pretrain_epoch 10000 \
               --train_epoch 350
```

### 3. Skip Pretraining (if model already trained)

```bash
python main.py --skip_pretrain
```

## Configuration System

The new modular design uses `config.py` for centralized configuration management.

### Adding a New Dataset

Edit `config.py`:

```python
DATASETS = {
    'MyDataset': {
        'name': 'my_dataset_name',
        'data_path': './data/my_data/',
        'label_file': 'my_labels.csv',
        'rna_file': 'adata_RNA.h5ad',
        'protein_file': 'adata_ADT.h5ad',
        'view1': 'RNA',
        'view2': 'Protein',
        'has_labels': True,
        'paper_metrics': {  # Optional: for comparison
            'ARI': 0.xxx,
            'NMI': 0.xxx,
            ...
        }
    }
}
```

Then run:
```bash
python main.py --name MyDataset
```

### Modifying Model Architecture

Edit the `MODEL_CONFIGS` section in `config.py`:

```python
MODEL_CONFIGS = {
    'custom': {
        'enc_dim1': 512,        # First encoder dimension
        'enc_dim2': 256,        # Second encoder dimension
        'dec_dim1': 256,        # First decoder dimension
        'dec_dim2': 512,        # Second decoder dimension
        'latent_dim': 32,       # Latent space dimension
        'dropout': 0.2,         # Dropout rate
        'num_layers': 3,        # Transformer layers
        'num_heads1': 2,        # Attention heads view 1
        'num_heads2': 2,        # Attention heads view 2
    }
}
```

### Running Ablation Experiments

Pre-configured ablation studies are available in `config.py`:

```python
# Disable KL divergence loss
TRAINING = {
    'ablation_no_kl': {
        'lambda1': 0.0,  # KL weight set to 0
        ...
    }
}
```

To implement custom ablation in code, modify `models/fusion.py`.

## Modifying the Model

### Adding/Removing Components

All model components are in the `models/` directory:

**1. Modify GCN Architecture** (`models/gcn.py`):
```python
class GCNEncoder(nn.Module):
    def __init__(self, ...):
        # Add more layers
        self.layer4 = GraphConvolutionLayer(...)
```

**2. Modify Transformer** (`models/transformer.py`):
```python
class TransformerEncoder(nn.Module):
    def __init__(self, ...):
        # Adjust number of layers, heads, etc.
```

**3. Modify Fusion Strategy** (`models/fusion.py`):
```python
def emb_fusion(self, adj, z_1, z_2, z_3):
    # Change fusion weights or strategy
    ...
```

### Disabling Components (Ablation)

To remove a component (e.g., Transformer):

Edit `models/fusion.py`:
```python
def forward(self, ...):
    # Comment out Transformer encoding
    # z13 = self.trans_encoder1(x1.unsqueeze(0), mask=None)
    # z13 = z13.squeeze(0)
    z13 = torch.zeros_like(z11)  # Use zeros instead
```

## Graph Parameters

### Spatial Neighbors (`--spatial_k`)

Controls the number of spatial neighbors in the spatial graph:
- Higher values: More connections, larger neighborhoods
- Default: 9

### Feature Neighbors (`--adj_k`)

Controls the number of feature-based neighbors:
- Higher values: More feature-similar connections
- Default: 20

## Training Parameters

### Lambda Parameters

- `--lambda1`: Weight for KL divergence loss (clustering guidance)
  - Higher: Stronger clustering constraint
  - Default: 1.0

- `--lambda2`: Weight for consistency loss (view agreement)
  - Higher: Force views to agree more
  - Default: 0.1

### Weight List

The `weight_list` parameter controls reconstruction loss weights:
```python
weight_list = [w1, w2, w3, w4, w5, w6]
# w1: spatial adj view1
# w2: feature adj view1
# w3: spatial adj view2
# w4: feature adj view2
# w5: feature reconstruction view1
# w6: feature reconstruction view2
```

## WandB Integration

### Enabling WandB Logging

```bash
python main.py --use_wandb --wandb_project MyProject
```

### Viewing Results

WandB logs include:
- Training/pretraining losses
- Evaluation metrics per run
- Final summary statistics
- Comparison with paper baselines

## Output Files

Results are saved to `./results/{dataset_name}/{timestamp}/`:

- `{dataset}_performance.csv`: Performance metrics for all runs
- `{dataset}_{run_idx}_pre_label.npy`: Predicted cluster labels
- `{dataset}_{run_idx}_laten.npy`: Latent embeddings
- `spatial_coords.npy`: Spatial coordinates for visualization
- `training_log.txt`: Complete training log

## Common Workflows

### 1. Test Different Seeds

```bash
for seed in 0 1 2 3 4; do
    python main.py --seed $seed
done
```

### 2. Hyperparameter Search

```bash
for lambda1 in 0.5 1.0 2.0; do
    for lambda2 in 0.05 0.1 0.2; do
        python main.py --lambda1 $lambda1 --lambda2 $lambda2
    done
done
```

### 3. Quick Testing (Fewer Epochs)

```bash
python main.py --pretrain_epoch 1000 --train_epoch 100
```

## Troubleshooting

### Out of Memory

- Reduce batch size (modify data dimensions)
- Reduce model dimensions in `config.py`
- Use CPU: `--device cpu`

### Import Errors

All old imports still work via `encoder.py`:
```python
# Both work:
from models import GCNEncoder
from encoder import GCNEncoder  # Backward compatible
```

### WandB Issues

If WandB is not installed:
```bash
pip install wandb
```

Disable WandB:
```bash
python main.py --no-use_wandb
```

## Advanced: Custom Experiments

Create a custom experiment script:

```python
from config import get_full_config
from models import GCNAutoencoder
# ... your custom training loop

config = get_full_config(
    dataset='D1',
    training='ablation_no_kl',
    lambda1=0.5  # Override specific params
)
```

## Questions?

Contact: kyochilian@gmail.com
