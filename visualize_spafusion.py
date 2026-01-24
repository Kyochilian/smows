# -*- coding:utf-8 -*-
"""
SpaFusion Visualization Script
Generates visualizations for SpaFusion training results
Usage: python visualize_spafusion.py --results_dir results/SpaFusion_Human_Lymph_Node_D1_20260124_170109
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import scanpy as sc
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FIXED_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
]


def get_color(cluster_id):
    return FIXED_COLORS[cluster_id % len(FIXED_COLORS)]


def normalize_dataset_name(data_name):
    mapping = {
        'A1': 'Human_Lymph_Node_A1',
        'D1': 'Human_Lymph_Node_D1',
    }
    return mapping.get(data_name, data_name)


def align_predictions_to_ground_truth(predictions, ground_truth):
    pred = np.asarray(predictions)
    gt = np.asarray(ground_truth)

    pred_labels = np.unique(pred)
    gt_labels = np.unique(gt)

    conf = np.zeros((len(gt_labels), len(pred_labels)), dtype=int)
    for i, gt_val in enumerate(gt_labels):
        gt_mask = gt == gt_val
        for j, pred_val in enumerate(pred_labels):
            conf[i, j] = int(np.sum(pred[gt_mask] == pred_val))

    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-conf)
        assignment = list(zip(row_ind.tolist(), col_ind.tolist()))
    except Exception:
        try:
            from munkres import Munkres
            m = Munkres()
            assignment = m.compute((-conf).tolist())
        except Exception:
            assignment = [(int(np.argmax(conf[:, j])), j) for j in range(conf.shape[1])]

    pred_to_gt = {}
    for r, c in assignment:
        if 0 <= r < len(gt_labels) and 0 <= c < len(pred_labels):
            pred_to_gt[int(pred_labels[c])] = int(gt_labels[r])

    next_label = (int(np.max(gt_labels)) + 1) if gt_labels.size else 0
    for pred_val in pred_labels:
        pred_key = int(pred_val)
        if pred_key not in pred_to_gt:
            pred_to_gt[pred_key] = next_label
            next_label += 1

    mapped = np.array([pred_to_gt[int(v)] for v in pred], dtype=int)
    return mapped, pred_to_gt


def load_spafusion_results(results_dir):
    """Load SpaFusion training results"""
    results = {}
    
    summary_path = os.path.join(results_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            results['summary'] = json.load(f)
    
    pretrain_loss_path = os.path.join(results_dir, 'pretrain_losses.npy')
    if os.path.exists(pretrain_loss_path):
        results['pretrain_losses'] = np.load(pretrain_loss_path)
    
    train_loss_path = os.path.join(results_dir, 'train_losses.npy')
    if os.path.exists(train_loss_path):
        results['train_losses'] = np.load(train_loss_path)
    
    pred_path = os.path.join(results_dir, 'final_predictions.npy')
    if os.path.exists(pred_path):
        results['predictions'] = np.load(pred_path)
    
    best_pred_path = os.path.join(results_dir, 'best_predictions.npy')
    if os.path.exists(best_pred_path):
        results['best_predictions'] = np.load(best_pred_path)
    
    latent_path = os.path.join(results_dir, 'latent_features.npy')
    if os.path.exists(latent_path):
        results['latent_features'] = np.load(latent_path)
    
    return results


def load_ground_truth(data_name):
    """Load ground truth labels"""
    data_name = normalize_dataset_name(data_name)
    
    label_paths = [
        f'./data/10X/{data_name}/label.npy',
        f'./data/MISAR/{data_name}/label.npy',
    ]
    
    for path in label_paths:
        if os.path.exists(path):
            return np.load(path)

    if 'D1' in data_name:
        csv_path = f'./data/10X/{data_name}/D1_annotation_labels.csv'
        if os.path.exists(csv_path):
            try:
                labels_df = pd.read_csv(csv_path)
                return labels_df['labels'].values
            except Exception:
                pass
    
    adata_paths = [
        (f'./data/10X/{data_name}/adata_RNA.h5ad', ['final_annot', 'cell_type', 'Combined_Clusters']),
        (f'./data/MISAR/{data_name}/adata_RNA.h5ad', ['Combined_Clusters', 'cell_type']),
    ]
    
    for adata_path, obs_keys in adata_paths:
        if os.path.exists(adata_path):
            try:
                adata = sc.read_h5ad(adata_path)
                for key in obs_keys:
                    if key in adata.obs:
                        labels = adata.obs[key]
                        if hasattr(labels, 'cat'):
                            return labels.cat.codes.values
                        else:
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            return le.fit_transform(labels)
            except:
                continue
    
    return None


def plot_spafusion_training_loss(results, save_dir):
    """Plot SpaFusion training loss curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if 'pretrain_losses' in results:
        pretrain_losses = results['pretrain_losses']
        axes[0].plot(pretrain_losses, 'b-', linewidth=1.5, alpha=0.8)
        axes[0].set_title('SpaFusion Pretrain Loss', fontsize=14)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        window = min(50, len(pretrain_losses) // 10)
        if window > 1:
            smoothed = np.convolve(pretrain_losses, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(pretrain_losses)), smoothed, 'r-', 
                        linewidth=2, label=f'Smoothed (window={window})')
            axes[0].legend()
    
    if 'train_losses' in results:
        train_losses = results['train_losses']
        axes[1].plot(train_losses, 'g-', linewidth=1.5, alpha=0.8)
        axes[1].set_title('SpaFusion Training Loss', fontsize=14)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        
        window = min(50, len(train_losses) // 10)
        if window > 1:
            smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')
            axes[1].plot(range(window-1, len(train_losses)), smoothed, 'r-', 
                        linewidth=2, label=f'Smoothed (window={window})')
            axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spafusion_training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: spafusion_training_loss.png")


def plot_combined_loss_spafusion(results, save_dir):
    """Plot combined loss curve for SpaFusion"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    offset = 0
    
    if 'pretrain_losses' in results:
        pretrain_losses = results['pretrain_losses']
        x_pretrain = np.arange(offset, offset + len(pretrain_losses))
        ax.plot(x_pretrain, pretrain_losses, color='#1f77b4', linewidth=1.5, 
               label='Pretrain Phase', alpha=0.8)
        ax.axvline(x=offset + len(pretrain_losses) - 1, color='gray', 
                  linestyle='--', alpha=0.5, label='Phase Boundary')
        offset += len(pretrain_losses)
    
    if 'train_losses' in results:
        train_losses = results['train_losses']
        x_train = np.arange(offset, offset + len(train_losses))
        ax.plot(x_train, train_losses, color='#2ca02c', linewidth=1.5, 
               label='Training Phase', alpha=0.8)
    
    ax.set_xlabel('Total Iterations', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('SpaFusion Complete Training Loss Curve', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spafusion_combined_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: spafusion_combined_loss.png")


def plot_cluster_distribution(predictions, ground_truth, save_dir):
    """Plot cluster distribution comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    pred_adjusted = predictions - predictions.min()
    
    pred_to_gt = None
    if ground_truth is not None:
        gt_adjusted = ground_truth - ground_truth.min()
        _, pred_to_gt = align_predictions_to_ground_truth(pred_adjusted, gt_adjusted)
    
    unique_pred, counts_pred = np.unique(pred_adjusted, return_counts=True)
    colors_pred = []
    for i in unique_pred:
        mapped_label = int(pred_to_gt[int(i)]) if pred_to_gt is not None else int(i)
        colors_pred.append(get_color(mapped_label))
        
    bars1 = axes[0].bar(unique_pred, counts_pred, color=colors_pred, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Cluster ID', fontsize=12)
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].set_title('SpaFusion Predicted Cluster Distribution', fontsize=14)
    axes[0].set_xticks(unique_pred)
    for bar, count in zip(bars1, counts_pred):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=8)
    
    if ground_truth is not None:
        unique_gt, counts_gt = np.unique(gt_adjusted, return_counts=True)
        colors_gt = [get_color(int(i)) for i in unique_gt]
        bars2 = axes[1].bar(unique_gt, counts_gt, color=colors_gt, edgecolor='black', linewidth=0.5)
        axes[1].set_xlabel('Cluster ID', fontsize=12)
        axes[1].set_ylabel('Number of Samples', fontsize=12)
        axes[1].set_title('Ground Truth Distribution', fontsize=14)
        axes[1].set_xticks(unique_gt)
        for bar, count in zip(bars2, counts_gt):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=8)
    else:
        axes[1].text(0.5, 0.5, 'Ground Truth Not Available', ha='center', va='center',
                    fontsize=14, transform=axes[1].transAxes)
        axes[1].set_title('Ground Truth Distribution', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spafusion_cluster_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: spafusion_cluster_distribution.png")


def plot_confusion_matrix_spafusion(predictions, ground_truth, save_dir):
    """Plot confusion matrix"""
    if ground_truth is None:
        print("Ground truth not available, skipping confusion matrix")
        return
    
    pred_adjusted = predictions - predictions.min()
    gt_adjusted = ground_truth - ground_truth.min()
    
    cm = confusion_matrix(gt_adjusted, pred_adjusted)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=np.unique(pred_adjusted),
                yticklabels=np.unique(gt_adjusted))
    
    ax.set_xlabel('Predicted Cluster', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    ax.set_title('SpaFusion Confusion Matrix: Prediction vs Ground Truth', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spafusion_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: spafusion_confusion_matrix.png")


def plot_tsne_spafusion(latent_features, predictions, ground_truth, save_dir, perplexity=30):
    """Plot t-SNE visualization for SpaFusion"""
    print("Computing t-SNE embedding (this may take a while)...")
    
    tsne = TSNE(n_components=2, perplexity=min(perplexity, latent_features.shape[0]-1), 
                random_state=42, n_iter=1000)
    embedding = tsne.fit_transform(latent_features)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    pred_adjusted = predictions - predictions.min()

    pred_to_gt = None
    if ground_truth is not None:
        gt_adjusted = ground_truth - ground_truth.min()
        _, pred_to_gt = align_predictions_to_ground_truth(pred_adjusted, gt_adjusted)
    
    unique_pred = np.unique(pred_adjusted)
    for cluster in unique_pred:
        mask = pred_adjusted == cluster
        mapped_label = int(pred_to_gt[int(cluster)]) if pred_to_gt is not None else int(cluster)
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1], 
                       c=[get_color(mapped_label)], 
                       label=(f'Pred {int(cluster)}→Class {mapped_label}' if pred_to_gt is not None else f'Cluster {int(cluster)}'), 
                       s=10, alpha=0.7)
    axes[0].set_xlabel('t-SNE 1', fontsize=12)
    axes[0].set_ylabel('t-SNE 2', fontsize=12)
    axes[0].set_title('SpaFusion t-SNE: Predicted Clusters', fontsize=14)
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=2, fontsize=8)
    
    if ground_truth is not None:
        unique_gt = np.unique(gt_adjusted)
        for cluster in unique_gt:
            mask = gt_adjusted == cluster
            axes[1].scatter(embedding[mask, 0], embedding[mask, 1], 
                           c=[get_color(int(cluster))], label=f'Class {int(cluster)}', 
                           s=10, alpha=0.7)
        axes[1].set_xlabel('t-SNE 1', fontsize=12)
        axes[1].set_ylabel('t-SNE 2', fontsize=12)
        axes[1].set_title('t-SNE: Ground Truth', fontsize=14)
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=2, fontsize=8)
    else:
        axes[1].text(0.5, 0.5, 'Ground Truth Not Available', ha='center', va='center',
                    fontsize=14, transform=axes[1].transAxes)
        axes[1].set_title('t-SNE: Ground Truth', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spafusion_tsne.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: spafusion_tsne.png")
    
    return embedding


def plot_pca_spafusion(latent_features, predictions, ground_truth, save_dir):
    """Plot PCA visualization for SpaFusion"""
    print("Computing PCA embedding...")
    
    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(latent_features)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    pred_adjusted = predictions - predictions.min()

    pred_to_gt = None
    if ground_truth is not None:
        gt_adjusted = ground_truth - ground_truth.min()
        _, pred_to_gt = align_predictions_to_ground_truth(pred_adjusted, gt_adjusted)
    
    unique_pred = np.unique(pred_adjusted)
    for cluster in unique_pred:
        mask = pred_adjusted == cluster
        mapped_label = int(pred_to_gt[int(cluster)]) if pred_to_gt is not None else int(cluster)
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1], 
                       c=[get_color(mapped_label)], 
                       label=(f'Pred {int(cluster)}→Class {mapped_label}' if pred_to_gt is not None else f'Cluster {int(cluster)}'), 
                       s=10, alpha=0.7)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    axes[0].set_title('SpaFusion PCA: Predicted Clusters', fontsize=14)
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=2, fontsize=8)
    
    if ground_truth is not None:
        unique_gt = np.unique(gt_adjusted)
        for cluster in unique_gt:
            mask = gt_adjusted == cluster
            axes[1].scatter(embedding[mask, 0], embedding[mask, 1], 
                           c=[get_color(int(cluster))], label=f'Class {int(cluster)}', 
                           s=10, alpha=0.7)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        axes[1].set_title('PCA: Ground Truth', fontsize=14)
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=2, fontsize=8)
    else:
        axes[1].text(0.5, 0.5, 'Ground Truth Not Available', ha='center', va='center',
                    fontsize=14, transform=axes[1].transAxes)
        axes[1].set_title('PCA: Ground Truth', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spafusion_pca.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: spafusion_pca.png")
    
    return embedding


def plot_spatial_distribution(predictions, ground_truth, data_name, save_dir):
    """Plot spatial distribution of clusters"""
    data_name = normalize_dataset_name(data_name)
    
    adata_path = f'./data/10X/{data_name}/adata_RNA.h5ad'
    if not os.path.exists(adata_path):
        print(f"Cannot find adata file at {adata_path}, skipping spatial plot")
        return
    
    try:
        adata = sc.read_h5ad(adata_path)
        spatial_coords = adata.obsm['spatial']
    except Exception as e:
        print(f"Error loading spatial coordinates: {e}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    pred_adjusted = predictions - predictions.min()
    
    pred_to_gt = None
    if ground_truth is not None:
        gt_adjusted = ground_truth - ground_truth.min()
        _, pred_to_gt = align_predictions_to_ground_truth(pred_adjusted, gt_adjusted)
    
    unique_pred = np.unique(pred_adjusted)
    for cluster in unique_pred:
        mask = pred_adjusted == cluster
        mapped_label = int(pred_to_gt[int(cluster)]) if pred_to_gt is not None else int(cluster)
        axes[0].scatter(spatial_coords[mask, 0], spatial_coords[mask, 1],
                       c=[get_color(mapped_label)],
                       label=f'Cluster {int(cluster)}',
                       s=3, alpha=0.7)
    axes[0].set_xlabel('Spatial X', fontsize=12)
    axes[0].set_ylabel('Spatial Y', fontsize=12)
    axes[0].set_title('SpaFusion Spatial: Predicted Clusters', fontsize=14)
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=3, fontsize=8)
    axes[0].set_aspect('equal')
    
    if ground_truth is not None:
        unique_gt = np.unique(gt_adjusted)
        for cluster in unique_gt:
            mask = gt_adjusted == cluster
            axes[1].scatter(spatial_coords[mask, 0], spatial_coords[mask, 1],
                           c=[get_color(int(cluster))],
                           label=f'Class {int(cluster)}',
                           s=3, alpha=0.7)
        axes[1].set_xlabel('Spatial X', fontsize=12)
        axes[1].set_ylabel('Spatial Y', fontsize=12)
        axes[1].set_title('Spatial: Ground Truth', fontsize=14)
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=3, fontsize=8)
        axes[1].set_aspect('equal')
    else:
        axes[1].text(0.5, 0.5, 'Ground Truth Not Available', ha='center', va='center',
                    fontsize=14, transform=axes[1].transAxes)
        axes[1].set_title('Spatial: Ground Truth', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spafusion_spatial.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: spafusion_spatial.png")


def plot_metrics_summary(results, save_dir):
    """Plot metrics summary"""
    if 'summary' not in results:
        return
    
    summary = results['summary']
    
    metrics_data = None
    if 'best_metrics' in summary:
        metrics_data = summary['best_metrics']
        title = 'SpaFusion Best Metrics'
    elif 'final_metrics' in summary:
        metrics_data = summary['final_metrics']
        title = 'SpaFusion Final Metrics'
    
    if metrics_data is None:
        return
    
    metric_names = ['ACC', 'F1', 'NMI', 'ARI', 'AMI']
    metric_values = [metrics_data.get(m, 0) for m in metric_names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231']
    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1)
    
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spafusion_metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: spafusion_metrics_summary.png")


def main():
    parser = argparse.ArgumentParser(description='SpaFusion Visualization')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to SpaFusion results directory')
    parser.add_argument('--tsne_perplexity', type=int, default=30,
                       help='Perplexity for t-SNE')
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    print("=" * 60)
    print("SpaFusion Visualization")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    
    results = load_spafusion_results(args.results_dir)
    
    if not results:
        print("Error: No results found in directory")
        return
    
    data_name = None
    if 'summary' in results:
        data_name = results['summary'].get('dataset', None)
    
    if data_name is None:
        dir_name = os.path.basename(args.results_dir)
        if 'A1' in dir_name:
            data_name = 'Human_Lymph_Node_A1'
        elif 'D1' in dir_name:
            data_name = 'Human_Lymph_Node_D1'
    
    print(f"Dataset: {data_name}")
    
    ground_truth = None
    if data_name:
        ground_truth = load_ground_truth(data_name)
        if ground_truth is not None:
            print(f"Ground truth loaded: {len(ground_truth)} samples, {len(np.unique(ground_truth))} classes")
        else:
            print("Ground truth not found")
    
    print("-" * 60)
    print("Generating visualizations...")
    
    if 'pretrain_losses' in results or 'train_losses' in results:
        plot_spafusion_training_loss(results, args.results_dir)
        plot_combined_loss_spafusion(results, args.results_dir)
    
    predictions = results.get('best_predictions', results.get('predictions', None))
    if predictions is not None:
        plot_cluster_distribution(predictions, ground_truth, args.results_dir)
        plot_confusion_matrix_spafusion(predictions, ground_truth, args.results_dir)
        
        if data_name:
            plot_spatial_distribution(predictions, ground_truth, data_name, args.results_dir)
    
    if 'latent_features' in results and predictions is not None:
        latent_features = results['latent_features']
        plot_tsne_spafusion(latent_features, predictions, ground_truth, args.results_dir, 
                           perplexity=args.tsne_perplexity)
        plot_pca_spafusion(latent_features, predictions, ground_truth, args.results_dir)
    
    plot_metrics_summary(results, args.results_dir)
    
    print("-" * 60)
    print("Visualization completed!")
    print(f"All figures saved to: {args.results_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
