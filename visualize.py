# -*- coding:utf-8 -*-
"""
可视化脚本 - 用于生成训练结果的各种图表(包含与Ground Truth对比)
使用方法: python visualize.py --results_dir results/Human_Lymph_Node_A1_20250114_153045
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
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义固定的高对比度颜色映射(最多20个聚类)
FIXED_COLORS = [
    '#e6194b',  # 红色 - 0
    '#3cb44b',  # 绿色 - 1
    '#ffe119',  # 黄色 - 2
    '#4363d8',  # 蓝色 - 3
    '#f58231',  # 橙色 - 4
    '#911eb4',  # 紫色 - 5
    '#46f0f0',  # 青色 - 6
    '#f032e6',  # 品红 - 7
    '#bcf60c',  # 柠檬绿 - 8
    '#fabebe',  # 粉色 - 9
    '#008080',  # 青绿 - 10
    '#e6beff',  # 淡紫 - 11
    '#9a6324',  # 棕色 - 12
    '#fffac8',  # 米色 - 13
    '#800000',  # 栗色 - 14
    '#aaffc3',  # 薄荷绿 - 15
    '#808000',  # 橄榄绿 - 16
    '#ffd8b1',  # 杏色 - 17
    '#000075',  # 海军蓝 - 18
    '#808080',  # 灰色 - 19
]


def get_color(cluster_id):
    """根据聚类ID获取固定颜色"""
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

    assignment = None
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


def load_results(results_dir):
    """加载训练结果"""
    results = {}
    
    # 加载summary
    summary_path = os.path.join(results_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            results['summary'] = json.load(f)
    
    # 加载训练历史
    history_path = os.path.join(results_dir, 'training_history.npy')
    if os.path.exists(history_path):
        results['history'] = np.load(history_path, allow_pickle=True).item()
    
    # 加载预测结果
    pred_path = os.path.join(results_dir, 'final_predictions.npy')
    if os.path.exists(pred_path):
        results['predictions'] = np.load(pred_path)
    
    # 加载相似度矩阵
    sim_path = os.path.join(results_dir, 'similarity_matrix.npy')
    if os.path.exists(sim_path):
        results['similarity'] = np.load(sim_path)
    
    # 加载谱矩阵
    spectral_path = os.path.join(results_dir, 'spectral_matrix.npy')
    if os.path.exists(spectral_path):
        results['spectral'] = np.load(spectral_path)
    
    return results


def load_ground_truth(data_name):
    """加载Ground Truth标签"""
    data_name = normalize_dataset_name(data_name)
    # 尝试从不同路径加载
    label_paths = [
        f'./data/10X/{data_name}/label.npy',
        f'./data/MISAR/{data_name}/label.npy',
    ]
    
    for path in label_paths:
        if os.path.exists(path):
            return np.load(path)

    csv_label_paths = [
        f'./data/10X/{data_name}/D1_annotation_labels.csv',
    ]
    for path in csv_label_paths:
        if os.path.exists(path):
            try:
                labels = np.loadtxt(path, delimiter=',', skiprows=1)
                return labels.astype(int)
            except Exception:
                continue
    
    # 尝试从h5ad文件加载
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


def plot_training_loss(history, save_dir):
    """绘制训练损失曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    stages = ['pretrain_stage0', 'pretrain_stage1', 'train_stage2']
    colors = ['b', 'g', 'r']
    titles = ['Pretrain Stage 0 Loss', 'Pretrain Stage 1 Loss', 'Training Stage 2 Loss']
    
    for ax, stage, color, title in zip(axes, stages, colors, titles):
        if stage in history and len(history[stage]) > 0:
            ax.plot(history[stage], f'{color}-', linewidth=1.5)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: training_loss.png")


def plot_combined_loss(history, save_dir):
    """绘制合并的损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    offset = 0
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    labels = ['Pretrain Stage 0', 'Pretrain Stage 1', 'Training Stage 2']
    
    for i, (stage, color, label) in enumerate(zip(
        ['pretrain_stage0', 'pretrain_stage1', 'train_stage2'], colors, labels)):
        if stage in history and len(history[stage]) > 0:
            x = np.arange(offset, offset + len(history[stage]))
            ax.plot(x, history[stage], color=color, linewidth=1.5, label=label)
            if i < 2:
                ax.axvline(x=offset + len(history[stage]) - 1, color='gray', 
                          linestyle='--', alpha=0.5)
            offset += len(history[stage])
    
    ax.set_xlabel('Total Iterations', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Complete Training Loss Curve', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: combined_loss.png")


def plot_cluster_distribution_comparison(predictions, ground_truth, save_dir):
    """绘制聚类分布对比柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 统一标签从0开始
    pred_adjusted = predictions - predictions.min()
    
    pred_to_gt = None
    if ground_truth is not None:
        gt_adjusted = ground_truth - ground_truth.min()
        _, pred_to_gt = align_predictions_to_ground_truth(pred_adjusted, gt_adjusted)
    
    # 预测结果分布
    unique_pred, counts_pred = np.unique(pred_adjusted, return_counts=True)
    colors_pred = []
    for i in unique_pred:
        mapped_label = int(pred_to_gt[int(i)]) if pred_to_gt is not None else int(i)
        colors_pred.append(get_color(mapped_label))
        
    bars1 = axes[0].bar(unique_pred, counts_pred, color=colors_pred, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Cluster ID', fontsize=12)
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].set_title('Predicted Cluster Distribution', fontsize=14)
    axes[0].set_xticks(unique_pred)
    for bar, count in zip(bars1, counts_pred):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=8)
    
    # Ground Truth分布
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
    plt.savefig(os.path.join(save_dir, 'cluster_distribution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: cluster_distribution_comparison.png")



def plot_confusion_matrix(predictions, ground_truth, save_dir):
    """绘制混淆矩阵"""
    if ground_truth is None:
        print("Ground truth not available, skipping confusion matrix")
        return
    
    # 调整预测标签从0开始
    pred_adjusted = predictions - predictions.min()
    gt_adjusted = ground_truth - ground_truth.min()
    
    cm = confusion_matrix(gt_adjusted, pred_adjusted)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=np.unique(pred_adjusted),
                yticklabels=np.unique(gt_adjusted))
    
    ax.set_xlabel('Predicted Cluster', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    ax.set_title('Confusion Matrix: Prediction vs Ground Truth', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrix.png")


def plot_tsne_comparison(similarity, predictions, ground_truth, save_dir, perplexity=30):
    """使用t-SNE可视化聚类结果与Ground Truth对比"""
    print("Computing t-SNE embedding (this may take a while)...")
    
    tsne = TSNE(n_components=2, perplexity=min(perplexity, similarity.shape[0]-1), 
                random_state=42, n_iter=1000)
    embedding = tsne.fit_transform(similarity)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # 统一标签从0开始
    pred_adjusted = predictions - predictions.min()

    pred_aligned = None
    pred_to_gt = None
    if ground_truth is not None:
        gt_adjusted = ground_truth - ground_truth.min()
        pred_aligned, pred_to_gt = align_predictions_to_ground_truth(pred_adjusted, gt_adjusted)
    
    # 预测结果
    unique_pred = np.unique(pred_adjusted)
    for cluster in unique_pred:
        mask = pred_adjusted == cluster
        mapped_label = int(pred_to_gt[int(cluster)]) if pred_to_gt is not None else int(cluster)
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1], 
                       c=[get_color(mapped_label)], label=(f'Pred {int(cluster)}→Class {mapped_label}' if pred_to_gt is not None else f'Cluster {int(cluster)}'), 
                       s=10, alpha=0.7)
    axes[0].set_xlabel('t-SNE 1', fontsize=12)
    axes[0].set_ylabel('t-SNE 2', fontsize=12)
    axes[0].set_title('t-SNE: Predicted Clusters', fontsize=14)
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=2, fontsize=8)
    
    # Ground Truth
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
    plt.savefig(os.path.join(save_dir, 'tsne_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: tsne_comparison.png")
    
    return embedding


def plot_pca_comparison(similarity, predictions, ground_truth, save_dir):
    """使用PCA可视化聚类结果与Ground Truth对比"""
    print("Computing PCA embedding...")
    
    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(similarity)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # 统一标签从0开始
    pred_adjusted = predictions - predictions.min()

    pred_aligned = None
    pred_to_gt = None
    if ground_truth is not None:
        gt_adjusted = ground_truth - ground_truth.min()
        pred_aligned, pred_to_gt = align_predictions_to_ground_truth(pred_adjusted, gt_adjusted)
    
    # 预测结果
    unique_pred = np.unique(pred_adjusted)
    for cluster in unique_pred:
        mask = pred_adjusted == cluster
        mapped_label = int(pred_to_gt[int(cluster)]) if pred_to_gt is not None else int(cluster)
        axes[0].scatter(embedding[mask, 0], embedding[mask, 1], 
                       c=[get_color(mapped_label)], label=(f'Pred {int(cluster)}→Class {mapped_label}' if pred_to_gt is not None else f'Cluster {int(cluster)}'), 
                       s=10, alpha=0.7)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    axes[0].set_title('PCA: Predicted Clusters', fontsize=14)
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=2, fontsize=8)
    
    # Ground Truth
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
    plt.savefig(os.path.join(save_dir, 'pca_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: pca_comparison.png")
    
    return embedding


def plot_spatial_comparison(predictions, ground_truth, data_name, save_dir):
    """绘制空间聚类结果与Ground Truth对比"""
    data_name = normalize_dataset_name(data_name)
    try:
        # 尝试加载原始数据获取空间坐标
        adata_path = f'./data/10X/{data_name}/adata_RNA.h5ad'
        if not os.path.exists(adata_path):
            adata_path = f'./data/MISAR/{data_name}/adata_RNA.h5ad'
        
        if not os.path.exists(adata_path):
            print(f"Spatial data not found for {data_name}")
            return
        
        adata = sc.read_h5ad(adata_path)
        
        if 'spatial' not in adata.obsm:
            print("No spatial coordinates found")
            return
        
        spatial = adata.obsm['spatial']
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # 统一标签从0开始
        pred_adjusted = predictions - predictions.min()
        
        pred_to_gt = None
        if ground_truth is not None:
            gt_adjusted = ground_truth - ground_truth.min()
            _, pred_to_gt = align_predictions_to_ground_truth(pred_adjusted, gt_adjusted)
        
        # 预测结果
        unique_pred = np.unique(pred_adjusted)
        for cluster in unique_pred:
            mask = pred_adjusted == cluster
            mapped_label = int(pred_to_gt[int(cluster)]) if pred_to_gt is not None else int(cluster)
            axes[0].scatter(spatial[mask, 0], spatial[mask, 1], 
                           c=[get_color(mapped_label)], 
                           label=(f'Pred {int(cluster)}→Class {mapped_label}' if pred_to_gt is not None else f'Cluster {int(cluster)}'), 
                           s=5, alpha=0.8)
        axes[0].set_xlabel('Spatial X', fontsize=12)
        axes[0].set_ylabel('Spatial Y', fontsize=12)
        axes[0].set_title('Spatial: Predicted Clusters', fontsize=14)
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=3, fontsize=8)
        axes[0].set_aspect('equal')
        axes[0].invert_yaxis()
        
        # Ground Truth
        if ground_truth is not None:
            unique_gt = np.unique(gt_adjusted)
            for cluster in unique_gt:
                mask = gt_adjusted == cluster
                axes[1].scatter(spatial[mask, 0], spatial[mask, 1], 
                               c=[get_color(int(cluster))], label=f'Class {int(cluster)}', 
                               s=5, alpha=0.8)
            axes[1].set_xlabel('Spatial X', fontsize=12)
            axes[1].set_ylabel('Spatial Y', fontsize=12)
            axes[1].set_title('Spatial: Ground Truth', fontsize=14)
            axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=3, fontsize=8)
            axes[1].set_aspect('equal')
            axes[1].invert_yaxis()
        else:
            axes[1].text(0.5, 0.5, 'Ground Truth Not Available', ha='center', va='center',
                        fontsize=14, transform=axes[1].transAxes)
            axes[1].set_title('Spatial: Ground Truth', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'spatial_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: spatial_comparison.png")
        
    except Exception as e:
        print(f"Could not create spatial comparison plot: {e}")



def plot_similarity_matrix(similarity, save_dir, max_size=500):
    """绘制相似度矩阵热图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if similarity.shape[0] > max_size:
        step = similarity.shape[0] // max_size
        similarity_plot = similarity[::step, ::step]
    else:
        similarity_plot = similarity
    
    sns.heatmap(similarity_plot, cmap='viridis', ax=ax, 
                xticklabels=False, yticklabels=False)
    ax.set_title('Similarity Matrix', fontsize=14)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Samples')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'similarity_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: similarity_matrix.png")


def plot_sorted_similarity_comparison(similarity, predictions, ground_truth, save_dir, max_samples=500):
    """绘制按聚类和Ground Truth排序的相似度热图对比"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 统一标签从0开始
    pred_adjusted = predictions - predictions.min()
    
    # 按预测聚类排序
    sorted_idx_pred = np.argsort(pred_adjusted)
    sorted_sim_pred = similarity[sorted_idx_pred][:, sorted_idx_pred]
    sorted_pred = pred_adjusted[sorted_idx_pred]
    
    if sorted_sim_pred.shape[0] > max_samples:
        step = sorted_sim_pred.shape[0] // max_samples
        sorted_sim_pred_plot = sorted_sim_pred[::step, ::step]
        sorted_pred_plot = sorted_pred[::step]
    else:
        sorted_sim_pred_plot = sorted_sim_pred
        sorted_pred_plot = sorted_pred
    
    sns.heatmap(sorted_sim_pred_plot, cmap='viridis', ax=axes[0],
                xticklabels=False, yticklabels=False)
    
    # 添加聚类边界线
    unique_pred = np.unique(sorted_pred_plot)
    for cluster in unique_pred[:-1]:
        boundary = np.where(sorted_pred_plot == cluster)[0][-1] + 1
        axes[0].axhline(y=boundary, color='white', linewidth=0.5)
        axes[0].axvline(x=boundary, color='white', linewidth=0.5)
    
    axes[0].set_title('Similarity Matrix (Sorted by Prediction)', fontsize=14)
    axes[0].set_xlabel('Samples')
    axes[0].set_ylabel('Samples')
    
    # 按Ground Truth排序
    if ground_truth is not None:
        gt_adjusted = ground_truth - ground_truth.min()
        sorted_idx_gt = np.argsort(gt_adjusted)
        sorted_sim_gt = similarity[sorted_idx_gt][:, sorted_idx_gt]
        sorted_gt = gt_adjusted[sorted_idx_gt]
        
        if sorted_sim_gt.shape[0] > max_samples:
            step = sorted_sim_gt.shape[0] // max_samples
            sorted_sim_gt_plot = sorted_sim_gt[::step, ::step]
            sorted_gt_plot = sorted_gt[::step]
        else:
            sorted_sim_gt_plot = sorted_sim_gt
            sorted_gt_plot = sorted_gt
        
        sns.heatmap(sorted_sim_gt_plot, cmap='viridis', ax=axes[1],
                    xticklabels=False, yticklabels=False)
        
        unique_gt = np.unique(sorted_gt_plot)
        for cluster in unique_gt[:-1]:
            boundary = np.where(sorted_gt_plot == cluster)[0][-1] + 1
            axes[1].axhline(y=boundary, color='white', linewidth=0.5)
            axes[1].axvline(x=boundary, color='white', linewidth=0.5)
        
        axes[1].set_title('Similarity Matrix (Sorted by Ground Truth)', fontsize=14)
        axes[1].set_xlabel('Samples')
        axes[1].set_ylabel('Samples')
    else:
        axes[1].text(0.5, 0.5, 'Ground Truth Not Available', ha='center', va='center',
                    fontsize=14, transform=axes[1].transAxes)
        axes[1].set_title('Similarity Matrix (Sorted by Ground Truth)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sorted_similarity_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: sorted_similarity_comparison.png")


def plot_metrics_bar(summary, save_dir):
    """绘制评估指标柱状图"""
    if 'final_metrics' not in summary:
        print("No metrics found in summary")
        return
    
    metrics = summary['final_metrics']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    bars = ax.bar(names, values, color=colors[:len(names)], edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Clustering Evaluation Metrics (vs Ground Truth)', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: metrics_bar.png")


def plot_metrics_comparison(summary, save_dir):
    """绘制最终指标与最佳指标对比"""
    if 'final_metrics' not in summary or 'best_metrics' not in summary:
        print("No comparison metrics available")
        return
    
    final = summary['final_metrics']
    best = summary['best_metrics']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_names = ['ACC', 'F1', 'NMI', 'ARI', 'AMI']
    x = np.arange(len(metrics_names))
    width = 0.35
    
    final_values = [final.get(m, 0) for m in metrics_names]
    best_values = [best.get(m, 0) for m in metrics_names]
    
    bars1 = ax.bar(x - width/2, final_values, width, label='Final', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, best_values, width, label='Best', color='#e74c3c', edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Final vs Best Metrics (Best at epoch {best.get("epoch", "N/A")})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: metrics_comparison.png")


def plot_cluster_matching_heatmap(predictions, ground_truth, save_dir):
    """绘制聚类匹配热图(归一化)"""
    if ground_truth is None:
        print("Ground truth not available, skipping cluster matching heatmap")
        return
    
    pred_adjusted = predictions - predictions.min()
    gt_adjusted = ground_truth - ground_truth.min()
    
    n_pred = len(np.unique(pred_adjusted))
    n_gt = len(np.unique(gt_adjusted))
    
    # 计算每个预测聚类中各Ground Truth类别的比例
    matching_matrix = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        mask = pred_adjusted == i
        if mask.sum() > 0:
            for j in range(n_gt):
                matching_matrix[i, j] = (gt_adjusted[mask] == j).sum() / mask.sum()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(matching_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                xticklabels=[f'GT {i}' for i in range(n_gt)],
                yticklabels=[f'Pred {i}' for i in range(n_pred)])
    
    ax.set_xlabel('Ground Truth Class', fontsize=12)
    ax.set_ylabel('Predicted Cluster', fontsize=12)
    ax.set_title('Cluster Matching Heatmap (Row Normalized)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cluster_matching_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: cluster_matching_heatmap.png")



def create_summary_figure(results, ground_truth, save_dir):
    """创建汇总图(包含Ground Truth对比)"""
    fig = plt.figure(figsize=(20, 15))
    
    # 统一标签从0开始
    pred_adjusted = results.get('predictions', None)
    if pred_adjusted is not None:
        pred_adjusted = pred_adjusted - pred_adjusted.min()
    
    gt_adjusted = ground_truth
    if gt_adjusted is not None:
        gt_adjusted = gt_adjusted - gt_adjusted.min()
    
    # 计算对齐
    pred_to_gt = None
    if pred_adjusted is not None and gt_adjusted is not None:
        _, pred_to_gt = align_predictions_to_ground_truth(pred_adjusted, gt_adjusted)
    
    # 1. 训练损失
    ax1 = fig.add_subplot(3, 3, 1)
    if 'history' in results:
        history = results['history']
        offset = 0
        for stage, color, label in zip(
            ['pretrain_stage0', 'pretrain_stage1', 'train_stage2'],
            ['#1f77b4', '#2ca02c', '#d62728'],
            ['Stage 0', 'Stage 1', 'Stage 2']):
            if stage in history and len(history[stage]) > 0:
                x = np.arange(offset, offset + len(history[stage]))
                ax1.plot(x, history[stage], color=color, linewidth=1, label=label)
                offset += len(history[stage])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
    
    # 2. 预测聚类分布
    ax2 = fig.add_subplot(3, 3, 2)
    if pred_adjusted is not None:
        unique, counts = np.unique(pred_adjusted, return_counts=True)
        colors = []
        for i in unique:
            mapped_label = int(pred_to_gt[int(i)]) if pred_to_gt is not None else int(i)
            colors.append(get_color(mapped_label))
        ax2.bar(unique, counts, color=colors)
        ax2.set_title('Predicted Cluster Distribution')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Count')
    
    # 3. Ground Truth分布
    ax3 = fig.add_subplot(3, 3, 3)
    if gt_adjusted is not None:
        unique_gt, counts_gt = np.unique(gt_adjusted, return_counts=True)
        colors_gt = [get_color(int(i)) for i in unique_gt]
        ax3.bar(unique_gt, counts_gt, color=colors_gt)
        ax3.set_title('Ground Truth Distribution')
        ax3.set_xlabel('Class ID')
        ax3.set_ylabel('Count')
    else:
        ax3.text(0.5, 0.5, 'Ground Truth\nNot Available', ha='center', va='center', fontsize=12)
        ax3.set_title('Ground Truth Distribution')
    
    # 4. 评估指标
    ax4 = fig.add_subplot(3, 3, 4)
    if 'summary' in results and 'final_metrics' in results['summary']:
        metrics = results['summary']['final_metrics']
        names = list(metrics.keys())
        values = list(metrics.values())
        metric_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
        ax4.bar(names, values, color=metric_colors[:len(names)])
        ax4.set_title('Evaluation Metrics (vs GT)')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1.1)
        for i, v in enumerate(values):
            ax4.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
    
    # 5. PCA - 预测
    ax5 = fig.add_subplot(3, 3, 5)
    if 'similarity' in results and pred_adjusted is not None:
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(results['similarity'])
        unique_clusters = np.unique(pred_adjusted)
        for cluster in unique_clusters:
            mask = pred_adjusted == cluster
            mapped_label = int(pred_to_gt[int(cluster)]) if pred_to_gt is not None else int(cluster)
            ax5.scatter(embedding[mask, 0], embedding[mask, 1], 
                       c=[get_color(mapped_label)], s=3, alpha=0.7)
        ax5.set_title('PCA: Predicted')
        ax5.set_xlabel('PC1')
        ax5.set_ylabel('PC2')
    
    # 6. PCA - Ground Truth
    ax6 = fig.add_subplot(3, 3, 6)
    if 'similarity' in results and gt_adjusted is not None:
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(results['similarity'])
        unique_gt = np.unique(gt_adjusted)
        for cluster in unique_gt:
            mask = gt_adjusted == cluster
            ax6.scatter(embedding[mask, 0], embedding[mask, 1], 
                       c=[get_color(int(cluster))], s=3, alpha=0.7)
        ax6.set_title('PCA: Ground Truth')
        ax6.set_xlabel('PC1')
        ax6.set_ylabel('PC2')
    else:
        ax6.text(0.5, 0.5, 'Ground Truth\nNot Available', ha='center', va='center', fontsize=12)
        ax6.set_title('PCA: Ground Truth')
    
    # 7. 相似度矩阵
    ax7 = fig.add_subplot(3, 3, 7)
    if 'similarity' in results:
        sim = results['similarity']
        if sim.shape[0] > 300:
            step = sim.shape[0] // 300
            sim = sim[::step, ::step]
        ax7.imshow(sim, cmap='viridis', aspect='auto')
        ax7.set_title('Similarity Matrix')
        ax7.set_xticks([])
        ax7.set_yticks([])
    
    # 8. 混淆矩阵(简化版)
    ax8 = fig.add_subplot(3, 3, 8)
    if pred_adjusted is not None and gt_adjusted is not None:
        cm = confusion_matrix(gt_adjusted, pred_adjusted)
        if cm.shape[0] <= 15:
            sns.heatmap(cm, cmap='Blues', ax=ax8, annot=True, fmt='d', 
                       annot_kws={'size': 6}, cbar=False)
        else:
            ax8.imshow(cm, cmap='Blues', aspect='auto')
        ax8.set_title('Confusion Matrix')
        ax8.set_xlabel('Predicted')
        ax8.set_ylabel('Ground Truth')
    else:
        ax8.text(0.5, 0.5, 'Ground Truth\nNot Available', ha='center', va='center', fontsize=12)
        ax8.set_title('Confusion Matrix')
    
    # 9. 配置信息
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    if 'summary' in results:
        summary = results['summary']
        info_text = f"""
Dataset: {summary.get('dataset', 'N/A')}
Clusters: {summary.get('n_clusters', 'N/A')}
Seed: {summary.get('seed', 'N/A')}
Pretrain Epochs: {summary.get('pretrain_epoch', 'N/A')}
Train Epochs: {summary.get('train_epoch', 'N/A')}
Lambda 1: {summary.get('lambda_1', 'N/A')}
Lambda 2: {summary.get('lambda_2', 'N/A')}
Lambda 3: {summary.get('lambda_3', 'N/A')}
"""
        if 'final_metrics' in summary:
            info_text += f"""
--- Final Metrics ---
ACC: {summary['final_metrics'].get('ACC', 'N/A'):.4f}
NMI: {summary['final_metrics'].get('NMI', 'N/A'):.4f}
ARI: {summary['final_metrics'].get('ARI', 'N/A'):.4f}
"""
        ax9.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', transform=ax9.transAxes)
        ax9.set_title('Configuration & Results')
    
    plt.suptitle('SpaMICS Training Summary (with Ground Truth Comparison)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_figure.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: summary_figure.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize SpaMICS training results with Ground Truth comparison')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to results directory')
    parser.add_argument('--skip_tsne', action='store_true',
                        help='Skip t-SNE visualization (can be slow)')
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    # 创建可视化输出文件夹
    vis_dir = os.path.join(args.results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Visualizations will be saved to: {vis_dir}")
    
    # 加载结果
    print("\nLoading results...")
    results = load_results(args.results_dir)
    
    # 获取数据集名称
    data_name = results.get('summary', {}).get('dataset', '')
    
    # 加载Ground Truth
    print("Loading ground truth...")
    ground_truth = load_ground_truth(data_name)
    if ground_truth is not None:
        print(f"Ground truth loaded: {len(np.unique(ground_truth))} classes, {len(ground_truth)} samples")
    else:
        print("Warning: Ground truth not found")
    
    # 生成各种可视化
    print("\nGenerating visualizations...")
    
    if 'history' in results:
        plot_training_loss(results['history'], vis_dir)
        plot_combined_loss(results['history'], vis_dir)
    
    if 'similarity' in results:
        plot_similarity_matrix(results['similarity'], vis_dir)
    
    if 'predictions' in results:
        plot_cluster_distribution_comparison(results['predictions'], ground_truth, vis_dir)
        plot_confusion_matrix(results['predictions'], ground_truth, vis_dir)
        plot_cluster_matching_heatmap(results['predictions'], ground_truth, vis_dir)
    
    if 'summary' in results:
        plot_metrics_bar(results['summary'], vis_dir)
        plot_metrics_comparison(results['summary'], vis_dir)
    
    if 'similarity' in results and 'predictions' in results:
        plot_pca_comparison(results['similarity'], results['predictions'], ground_truth, vis_dir)
        plot_sorted_similarity_comparison(results['similarity'], results['predictions'], ground_truth, vis_dir)
        
        if not args.skip_tsne:
            plot_tsne_comparison(results['similarity'], results['predictions'], ground_truth, vis_dir)
    
    if 'predictions' in results and data_name:
        plot_spatial_comparison(results['predictions'], ground_truth, data_name, vis_dir)
    
    # 创建汇总图
    create_summary_figure(results, ground_truth, vis_dir)
    
    print(f"\nAll visualizations saved to: {vis_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
