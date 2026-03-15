# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

SpaFusion is a multi-level fusion model for clustering spatial multi-omics data. It processes dual-omics inputs (RNA + Protein/ADT) through parallel GCN-Transformer pipelines, fuses them via cross-modal mechanisms, and performs deep clustering with KL-divergence guidance.

## Environment Setup

Python 3.9 with PyTorch 2.2.1+cu118. Requires CUDA.

```
conda env create -f environment.yaml
conda activate SP
```

Key dependencies: torch, scanpy, anndata, sklearn, scipy, munkres, umap-learn.

## Running

```
# Default run (Human lymph node D1 dataset, variance fusion)
& d:/SpaFusion/.venv/Scripts/python.exe d:/SIMO-main/main.py

# With MoE cross-modal fusion
& d:/SpaFusion/.venv/Scripts/python.exe d:/SIMO-main/main.py --cross_fusion moe --moe_num_experts 4 --moe_hidden_dim 64

# Key hyperparameters
& d:/SpaFusion/.venv/Scripts/python.exe d:/SIMO-main/main.py --name D1 --seed 0 --spatial_k 9 --adj_k 20 --lambda1 1 --lambda2 0.1 --lr 1e-3 --pretrain_epoch 10000 --train_epoch 2500
```

The script prompts for a result folder name at startup (interactive input required).

## Architecture

### Training Pipeline (main.py)

Two-phase training: **pretrain** (reconstruction-only, 10k epochs) then **train** (reconstruction + KL clustering loss, 2.5k epochs, run 10 times for statistics). Pretrained weights saved to `pretrain/`, results to `results/`.

### Model (encoder.py)

`GCNAutoencoder` is the main model class. For each omics view:
1. **Three parallel encoders** produce latent representations:
   - `GCNEncoder` on spatial graph (adj1/adj3)
   - `GCNEncoder` on feature graph blended with high-order motif matrix (adj2/adj4)
   - `trans_encoder` (Transformer) on raw features
2. **Intra-modal fusion** (`emb_fusion`): learned weighted combination of three embeddings + global-local attention via `alpha` parameter
3. **Cross-modal fusion** of the two view embeddings: either variance-based weighting (`var`, default) or dense MoE (`moe`)
4. **Decoders** reconstruct adjacency matrices and features; `q_distribution` produces soft cluster assignments

### MoE Fusion (defined in encoder.py, standalone copy in moe.py)

`MoEFusion`: Dense mixture-of-experts. Concatenates z1+z2, routes through all experts with softmax gating. Returns fused embedding + gate weights for diagnostics.

### Data Flow

- `processing.py`: Loads h5ad files, applies scanpy preprocessing (HVG selection, PCA), builds spatial and feature KNN graphs
- `utils.py`: Graph construction, adjacency normalization (symmetric D^{-1/2}AD^{-1/2}), KMeans initialization, target distribution, KL loss, evaluation dispatch
- `high_order_matrix.py`: Computes 3-node motif co-occurrence matrix from adjacency (cached to `pre_adj/`)
- `evaluate.py`: Clustering metrics via Hungarian matching (ACC, F1, NMI, ARI, AMI, VMS, FMS) and unsupervised metrics (silhouette, CH, DB)

### Graph Types Per View

Each omics view has three graph inputs:
- **Spatial graph**: KNN from `obsm['spatial']` coordinates
- **Feature graph**: KNN from PCA-reduced expression profiles
- **High-order matrix (Mt)**: 3-node motif counts derived from the feature graph

The spatial and feature graphs are element-wise multiplied before being fed to the encoder. Feature graph and Mt are blended via learned parameters k1, k2.

## Data

Default dataset: Human lymph node D1 in `data/`:
- `adata_RNA.h5ad` — RNA expression (scanpy AnnData)
- `adata_ADT.h5ad` — Protein/ADT expression (scanpy AnnData)
- `D1_annotation_labels.csv` — Ground truth cluster labels

## Generated Artifacts

- `pre_adj/{name}/` — Cached adjacency matrices and motif matrices (`.npy`)
- `pretrain/{name}_pre_model.pkl` — Pretrained model weights
- `results/{name}/{run_name}_{timestamp}/` — Per-experiment outputs: performance CSV, predicted labels, latent embeddings, training log, visualizations (UMAP, t-SNE, spatial plots)

## Loss Components

- **Reconstruction**: MSE on 4 adjacency matrices + 2 feature reconstructions (weighted by `weight_list`)
- **Clustering**: KL divergence between soft assignment Q and sharpened target P (`lambda1`)
- **Consistency**: MSE between fused Z and individual view embeddings z1_tilde, z2_tilde (`lambda2`)
