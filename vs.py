#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, silhouette_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import umap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ========= 命令行参数解析 =========
parser = argparse.ArgumentParser(description='SpaFusion Visualization')
parser.add_argument('--result_dir', type=str, default=None,
                    help='Path to result directory (e.g., results/D1/20231221_231959)')
parser.add_argument('--dataset', type=str, default='D1',
                    help='Dataset name (used when result_dir not specified)')
args = parser.parse_args()

# ========= 全局配置 =========
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

# 专业配色方案（适合细胞类型可视化）
CELL_PALETTE = [
    '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
    '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85',
    '#FF7F0E', '#2CA02C', '#9467BD', '#8C564B', '#E377C2',
    '#7F7F7F', '#BCBD22', '#17BECF', '#AEC7E8', '#FFBB78'
]

# ========= 路径配置 =========
def get_result_dir():
    if args.result_dir:
        result_dir = Path(args.result_dir)
        if not result_dir.exists():
            raise FileNotFoundError(f"Specified result directory not found: {result_dir}")
        # 从路径提取数据集名
        dataset_name = result_dir.parent.name
        return result_dir, dataset_name
    
    # 查找数据集目录下的时间戳子文件夹
    base_dir = Path(f'results/{args.dataset}')
    if not base_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {base_dir}")
    
    # 查找时间戳格式的子文件夹 (YYYYMMDD_HHMMSS)
    timestamp_dirs = sorted([
        d for d in base_dir.iterdir() 
        if d.is_dir() and len(d.name) == 15 and d.name[8] == '_'
    ], reverse=True)  # 按最新时间排序
    
    if timestamp_dirs:
        # 使用最新的时间戳文件夹
        result_dir = timestamp_dirs[0]
        print(f"Using latest result directory: {result_dir}")
        return result_dir, args.dataset
    else:
        # 回退到旧结构（直接使用数据集目录）
        print(f"No timestamp subdirectory found, using: {base_dir}")
        return base_dir, args.dataset

RESULT_DIR, DATASET_NAME = get_result_dir()
DATA_DIR = Path('data')
SAVE_DIR = RESULT_DIR / 'visualizations'
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """加载所有必要数据"""
    print("=" * 50)
    print("Loading data...")
    
    # 空间坐标
    coords = np.load(RESULT_DIR / 'spatial_coords.npy')
    print(f"  Spatial coords: {coords.shape}")
    
    # Ground Truth 标签
    gt_labels = pd.read_csv(DATA_DIR / f'{DATASET_NAME}_annotation_labels.csv')['labels'].values
    n_clusters = len(np.unique(gt_labels))
    print(f"  Ground truth labels: {gt_labels.shape}, {n_clusters} clusters")
    
    # 性能指标 - 正确解析CSV格式
    # 格式: seed:0, lambda1:1, ..., ACC,F1,NMI,ARI,AMI,VMS,FMS
    perf_data = []
    with open(RESULT_DIR / f'{DATASET_NAME}_performance.csv', 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # 最后7个值是指标
            metrics = [float(x.strip()) for x in parts[-7:]]
            perf_data.append(metrics)
    
    perf = pd.DataFrame(perf_data, columns=['ACC', 'F1', 'NMI', 'ARI', 'AMI', 'VMS', 'FMS'])
    print(f"  Performance runs: {len(perf)}")
    
    # 加载所有latent和预测标签（保持与perf的run索引一致）
    latents = []
    pred_labels = []
    run_ids = []
    missing_runs = []
    for i in range(len(perf)):
        latent_path = RESULT_DIR / f'{DATASET_NAME}_{i}_laten.npy'
        label_path = RESULT_DIR / f'{DATASET_NAME}_{i}_pre_label.npy'
        if latent_path.exists() and label_path.exists():
            latents.append(np.load(latent_path))
            pred_labels.append(np.load(label_path))
            run_ids.append(i)
        else:
            missing_runs.append(i)

    if len(latents) == 0:
        raise FileNotFoundError(
            f"No latent/label npy found under {RESULT_DIR} for dataset {DATASET_NAME}."
        )

    if missing_runs:
        print(f"  Warning: missing latent/label for runs: {missing_runs[:10]}" +
              (" ..." if len(missing_runs) > 10 else ""))

    print(f"  Loaded {len(latents)} runs of latent embeddings")

    # 只在已加载的run里找最佳run（按ARI）
    perf_loaded = perf.iloc[run_ids].reset_index(drop=True)
    best_pos = int(perf_loaded['ARI'].idxmax())
    best_run_id = int(run_ids[best_pos])
    print(f"  Best run id: {best_run_id} (ARI={perf.loc[best_run_id, 'ARI']:.4f})")
    print("=" * 50)
    
    return {
        'coords': coords,
        'gt_labels': gt_labels,
        'pred_labels': pred_labels,
        'latents': latents,
        'perf': perf,
        'best_pos': best_pos,
        'best_run_id': best_run_id,
        'run_ids': run_ids,
        'n_clusters': n_clusters
    }


def align_labels(pred, gt):
    """使用匈牙利算法对齐预测标签和真实标签"""
    pred = np.asarray(pred)
    gt = np.asarray(gt)

    # 允许任意取值的标签（不要求从0开始/连续）
    gt_classes = np.unique(gt)
    pred_classes = np.unique(pred)

    # 构建混淆矩阵：行=True label, 列=Pred label
    cm = confusion_matrix(gt, pred, labels=gt_classes)
    # 上面labels只控制行顺序；列会跟gt_classes一致，不适合对齐pred。
    # 因此手工构建 (len(gt_classes) x len(pred_classes)) 的计数矩阵。
    cm = np.zeros((len(gt_classes), len(pred_classes)), dtype=int)
    gt_to_i = {c: i for i, c in enumerate(gt_classes)}
    pred_to_j = {c: j for j, c in enumerate(pred_classes)}
    for g, p in zip(gt, pred):
        cm[gt_to_i[g], pred_to_j[p]] += 1
    
    # 使用匈牙利算法找最优匹配
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # 创建映射
    # mapping: pred_class -> gt_class
    mapping = {pred_classes[col_ind[i]]: gt_classes[row_ind[i]] for i in range(len(row_ind))}
    
    # 应用映射
    aligned_pred = np.array([mapping.get(p, p) for p in pred])
    
    return aligned_pred


def plot_spatial_domains(coords, labels, title, savepath, gt_labels=None, 
                         show_legend=True, point_size=8):
    """
    绘制空间域图 - 论文标准格式
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # 如果提供了gt_labels，对齐预测标签
    if gt_labels is not None:
        labels = align_labels(labels, gt_labels)
    
    unique_labels = np.unique(labels)
    n_colors = len(unique_labels)
    colors = CELL_PALETTE[:n_colors]
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], 
                   c=[colors[i]], s=point_size, 
                   label=f'Cluster {label}', 
                   alpha=0.8, edgecolors='none')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Spatial X')
    ax.set_ylabel('Spatial Y')
    ax.set_aspect('equal')
    
    if show_legend and n_colors <= 10:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                  frameon=False, markerscale=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {savepath.name}")


def plot_spatial_comparison(coords, gt_labels, pred_labels, savepath, point_size=6):
    """
    Ground Truth vs Prediction 对比图
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 对齐标签
    aligned_pred = align_labels(pred_labels, gt_labels)
    
    unique_labels = np.unique(gt_labels)
    n_colors = len(unique_labels)
    colors = CELL_PALETTE[:n_colors]
    
    # Ground Truth
    for i, label in enumerate(unique_labels):
        mask = gt_labels == label
        axes[0].scatter(coords[mask, 0], coords[mask, 1], 
                        c=[colors[i]], s=point_size, 
                        label=f'Type {label}', alpha=0.8, edgecolors='none')
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].set_aspect('equal')
    axes[0].axis('off')
    
    # Prediction
    for i, label in enumerate(unique_labels):
        mask = aligned_pred == label
        axes[1].scatter(coords[mask, 0], coords[mask, 1], 
                        c=[colors[i]], s=point_size, 
                        label=f'Type {label}', alpha=0.8, edgecolors='none')
    axes[1].set_title('SpaFusion Prediction', fontsize=14, fontweight='bold')
    axes[1].set_aspect('equal')
    axes[1].axis('off')
    
    # 共享图例
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='center right', 
               bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=2)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {savepath.name}")


def plot_tsne(latent, labels, title, savepath, gt_labels=None, perplexity=30):
    """
    t-SNE 降维可视化 - 论文标准格式
    """
    print(f"  Computing t-SNE (perplexity={perplexity})...")
    
    # sklearn不同版本参数名略有差异：n_iter vs max_iter；部分版本不支持n_jobs
    tsne_kwargs = dict(n_components=2, perplexity=perplexity, random_state=42)
    try:
        tsne = TSNE(**tsne_kwargs, n_iter=1000, n_jobs=-1)
    except TypeError:
        tsne = TSNE(**tsne_kwargs, max_iter=1000)
    emb = tsne.fit_transform(latent)
    
    if gt_labels is not None:
        labels = align_labels(labels, gt_labels)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    unique_labels = np.unique(labels)
    n_colors = len(unique_labels)
    colors = CELL_PALETTE[:n_colors]
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(emb[mask, 0], emb[mask, 1], 
                   c=[colors[i]], s=10, label=f'Cluster {label}', 
                   alpha=0.7, edgecolors='none')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
              frameon=False, markerscale=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {savepath.name}")
    
    return emb


def plot_umap(latent, labels, title, savepath, gt_labels=None):
    """
    UMAP 降维可视化
    """
    print("  Computing UMAP...")
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, 
                        random_state=42, n_jobs=-1)
    emb = reducer.fit_transform(latent)
    
    if gt_labels is not None:
        labels = align_labels(labels, gt_labels)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    unique_labels = np.unique(labels)
    n_colors = len(unique_labels)
    colors = CELL_PALETTE[:n_colors]
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(emb[mask, 0], emb[mask, 1], 
                   c=[colors[i]], s=10, label=f'Cluster {label}', 
                   alpha=0.7, edgecolors='none')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
              frameon=False, markerscale=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {savepath.name}")
    
    return emb


def plot_confusion_matrix(gt_labels, pred_labels, savepath):
    """
    绘制混淆矩阵热力图
    """
    aligned_pred = align_labels(pred_labels, gt_labels)
    
    n_clusters = len(np.unique(gt_labels))
    cm = confusion_matrix(gt_labels, aligned_pred, labels=range(n_clusters))
    
    # 归一化（按行）
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[f'Pred {i}' for i in range(n_clusters)],
                yticklabels=[f'True {i}' for i in range(n_clusters)],
                ax=ax, cbar_kws={'label': 'Proportion'})
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {savepath.name}")


def plot_metrics_summary(perf, savepath):
    """
    性能指标汇总图 - 箱线图 + 散点
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    metrics = ['ARI', 'NMI', 'ACC', 'AMI', 'F1', 'VMS', 'FMS']
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', 
              '#F39B7F', '#8491B4', '#91D1C2']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[i]
        
        # 箱线图
        bp = ax.boxplot(perf[metric], patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.6)
        bp['medians'][0].set_color('black')
        bp['medians'][0].set_linewidth(2)
        
        # 散点（显示每次run的结果）
        x = np.ones(len(perf)) + np.random.normal(0, 0.04, len(perf))
        ax.scatter(x, perf[metric], c='black', s=30, alpha=0.7, zorder=3)
        
        # 显示均值和标准差
        mean_val = perf[metric].mean()
        std_val = perf[metric].std()
        ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        ax.set_title(f'{metric}\n{mean_val:.4f} ± {std_val:.4f}', 
                     fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_ylabel('Score')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([max(0, perf[metric].min() - 0.05), 
                     min(1, perf[metric].max() + 0.05)])
    
    # 隐藏多余的subplot
    axes[-1].axis('off')
    
    plt.suptitle(f'SpaFusion Performance Summary ({len(perf)} runs)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {savepath.name}")


def plot_metrics_radar(perf, savepath):
    """
    雷达图展示平均性能
    """
    metrics = ['ARI', 'NMI', 'ACC', 'AMI', 'F1']
    values = [perf[m].mean() for m in metrics]
    
    # 闭合雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#E64B35', markersize=8)
    ax.fill(angles, values, alpha=0.25, color='#E64B35')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1)
    
    # 添加数值标签
    for angle, value, metric in zip(angles[:-1], values[:-1], metrics):
        ax.annotate(f'{value:.3f}', xy=(angle, value), 
                    xytext=(angle, value + 0.08),
                    ha='center', fontsize=10, fontweight='bold')
    
    ax.set_title('SpaFusion Performance Radar', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {savepath.name}")


def plot_embedding_comparison(latent, pred_labels, gt_labels, savepath):
    """
    t-SNE嵌入对比：预测标签 vs Ground Truth 标签着色
    """
    print("  Computing t-SNE for comparison...")
    
    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000, n_jobs=-1)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb = tsne.fit_transform(latent)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    unique_gt = np.unique(gt_labels)
    n_colors = len(unique_gt)
    colors = CELL_PALETTE[:n_colors]
    
    # Ground Truth 着色
    for i, label in enumerate(unique_gt):
        mask = gt_labels == label
        axes[0].scatter(emb[mask, 0], emb[mask, 1], 
                        c=[colors[i]], s=8, label=f'Type {label}', 
                        alpha=0.7, edgecolors='none')
    axes[0].set_title('Colored by Ground Truth', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    axes[0].legend(loc='best', frameon=False, markerscale=2)
    
    # 预测标签着色（对齐后）
    aligned_pred = align_labels(pred_labels, gt_labels)
    for i, label in enumerate(unique_gt):
        mask = aligned_pred == label
        axes[1].scatter(emb[mask, 0], emb[mask, 1], 
                        c=[colors[i]], s=8, label=f'Cluster {label}', 
                        alpha=0.7, edgecolors='none')
    axes[1].set_title('Colored by Prediction', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].legend(loc='best', frameon=False, markerscale=2)
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {savepath.name}")


def plot_paper_figure(coords, gt_labels, pred_labels, latent, perf, savepath):
    """
    生成论文级别的组合图 (类似 Figure 2/3)
    """
    fig = plt.figure(figsize=(18, 12))
    
    # 布局: 2行3列
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    aligned_pred = align_labels(pred_labels, gt_labels)
    unique_labels = np.unique(gt_labels)
    n_colors = len(unique_labels)
    colors = CELL_PALETTE[:n_colors]
    
    # 1. Ground Truth 空间图
    ax1 = fig.add_subplot(gs[0, 0])
    for i, label in enumerate(unique_labels):
        mask = gt_labels == label
        ax1.scatter(coords[mask, 0], coords[mask, 1], 
                    c=[colors[i]], s=6, alpha=0.8, edgecolors='none')
    ax1.set_title('(A) Ground Truth', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # 2. SpaFusion 预测空间图
    ax2 = fig.add_subplot(gs[0, 1])
    for i, label in enumerate(unique_labels):
        mask = aligned_pred == label
        ax2.scatter(coords[mask, 0], coords[mask, 1], 
                    c=[colors[i]], s=6, alpha=0.8, edgecolors='none')
    ax2.set_title('(B) SpaFusion', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # 3. t-SNE
    ax3 = fig.add_subplot(gs[0, 2])
    print("  Computing t-SNE for paper figure...")
    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000, n_jobs=-1)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb = tsne.fit_transform(latent)
    for i, label in enumerate(unique_labels):
        mask = aligned_pred == label
        ax3.scatter(emb[mask, 0], emb[mask, 1], 
                    c=[colors[i]], s=6, alpha=0.7, edgecolors='none')
    ax3.set_title('(C) t-SNE Embedding', fontsize=12, fontweight='bold')
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 4. UMAP
    ax4 = fig.add_subplot(gs[1, 0])
    print("  Computing UMAP for paper figure...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=-1)
    umap_emb = reducer.fit_transform(latent)
    for i, label in enumerate(unique_labels):
        mask = aligned_pred == label
        ax4.scatter(umap_emb[mask, 0], umap_emb[mask, 1], 
                    c=[colors[i]], s=6, alpha=0.7, edgecolors='none')
    ax4.set_title('(D) UMAP Embedding', fontsize=12, fontweight='bold')
    ax4.set_xlabel('UMAP 1')
    ax4.set_ylabel('UMAP 2')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # 5. 混淆矩阵
    ax5 = fig.add_subplot(gs[1, 1])
    cm = confusion_matrix(gt_labels, aligned_pred, labels=range(n_colors))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax5,
                cbar_kws={'shrink': 0.8})
    ax5.set_title('(E) Confusion Matrix', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('True')
    
    # 6. 性能指标条形图
    ax6 = fig.add_subplot(gs[1, 2])
    metrics = ['ARI', 'NMI', 'ACC', 'AMI', 'F1']
    means = [perf[m].mean() for m in metrics]
    stds = [perf[m].std() for m in metrics]
    bar_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
    
    bars = ax6.bar(metrics, means, yerr=stds, capsize=5, 
                   color=bar_colors, alpha=0.8, edgecolor='black')
    ax6.set_ylim(0, 1)
    ax6.set_ylabel('Score')
    ax6.set_title('(F) Performance Metrics', fontsize=12, fontweight='bold')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    
    # 在条形上方添加数值
    for bar, mean, std in zip(bars, means, stds):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                 f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 添加图例
    legend_elements = [plt.scatter([], [], c=colors[i], s=50, label=f'Cluster {i}') 
                       for i in range(n_colors)]
    fig.legend(handles=legend_elements, loc='center right', 
               bbox_to_anchor=(1.02, 0.5), frameon=False, title='Cell Types')
    
    plt.suptitle(f'SpaFusion Multi-omics Spatial Clustering Results ({DATASET_NAME})', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.savefig(savepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {savepath.name}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  SpaFusion Visualization Pipeline")
    print("=" * 60)
    
    # 加载数据
    data = load_data()
    
    coords = data['coords']
    gt_labels = data['gt_labels']
    pred_labels = data['pred_labels']
    latents = data['latents']
    perf = data['perf']
    best_pos = data['best_pos']
    best_run_id = data['best_run_id']
    run_ids = data['run_ids']

    best_pred = pred_labels[best_pos]
    best_latent = latents[best_pos]
    
    print("\n[1/8] Generating spatial domain plots...")
    # 生成每个run的空间图
    for i, run_id in enumerate(run_ids):
        plot_spatial_domains(
            coords, pred_labels[i], 
            f'SpaFusion Run {run_id} (ARI={perf.loc[run_id, "ARI"]:.3f})',
            SAVE_DIR / f'spatial_run{run_id}.png',
            gt_labels=gt_labels, show_legend=False, point_size=6
        )
    
    print("\n[2/8] Generating best run spatial plot...")
    plot_spatial_domains(
        coords, best_pred,
        f'SpaFusion Best Run {best_run_id} (ARI={perf.loc[best_run_id, "ARI"]:.4f})',
        SAVE_DIR / 'spatial_best.png',
        gt_labels=gt_labels, show_legend=True, point_size=8
    )
    
    print("\n[3/8] Generating GT vs Prediction comparison...")
    plot_spatial_comparison(
        coords, gt_labels, best_pred,
        SAVE_DIR / 'spatial_comparison.png'
    )
    
    print("\n[4/8] Generating t-SNE plot...")
    plot_tsne(
        best_latent, best_pred,
        'SpaFusion Latent Space (t-SNE)',
        SAVE_DIR / 'tsne_best.png',
        gt_labels=gt_labels
    )
    
    print("\n[5/8] Generating UMAP plot...")
    plot_umap(
        best_latent, best_pred,
        'SpaFusion Latent Space (UMAP)',
        SAVE_DIR / 'umap_best.png',
        gt_labels=gt_labels
    )
    
    print("\n[6/8] Generating confusion matrix...")
    plot_confusion_matrix(
        gt_labels, best_pred,
        SAVE_DIR / 'confusion_matrix.png'
    )
    
    print("\n[7/8] Generating performance summary...")
    plot_metrics_summary(perf, SAVE_DIR / 'metrics_boxplot.png')
    plot_metrics_radar(perf, SAVE_DIR / 'metrics_radar.png')
    
    print("\n[8/8] Generating embedding comparison...")
    plot_embedding_comparison(
        best_latent, best_pred, gt_labels,
        SAVE_DIR / 'embedding_comparison.png'
    )
    
    print("\n[Bonus] Generating paper-style figure...")
    plot_paper_figure(
        coords, gt_labels, best_pred, best_latent, perf,
        SAVE_DIR / 'paper_figure.png'
    )
    
    # 打印总结
    print("\n" + "=" * 60)
    print("  Visualization Summary")
    print("=" * 60)
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Total runs: {len(perf)}")
    print(f"  Best run: {best_run_id} (ARI={perf.loc[best_run_id, 'ARI']:.4f})")
    print(f"\n  Mean ± Std:")
    for col in ['ARI', 'NMI', 'ACC', 'AMI', 'F1']:
        print(f"    {col}: {perf[col].mean():.4f} ± {perf[col].std():.4f}")
    print(f"\n  Output directory: {SAVE_DIR}")
    print(f"\n  Generated files:")
    for f in sorted(SAVE_DIR.glob('*.png')):
        print(f"    - {f.name}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()