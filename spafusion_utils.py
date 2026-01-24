# -*- coding:utf-8 -*-
"""
SpaFusion Utility Functions
Author: polaris (original)
Adapted for SpaMICS comparison experiments
"""

import torch
import numpy as np
import random
from scipy.sparse import coo_matrix
import scipy.sparse as sp
from sklearn.cluster import KMeans
from evaluation import eval
import torch.nn.functional as F
import os
import pickle


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_adj_matrices(adj, spatial_path1, spatial_path2, feature1_path, feature2_path):
    """Save adjacency matrices to specified paths."""
    with open(spatial_path1, 'wb') as f:
        pickle.dump(adj['adj_spatial_omics1'], f)

    with open(spatial_path2, 'wb') as f:
        pickle.dump(adj['adj_spatial_omics2'], f)

    with open(feature1_path, 'wb') as f:
        pickle.dump(adj['adj_feature_omics1'], f)

    with open(feature2_path, 'wb') as f:
        pickle.dump(adj['adj_feature_omics2'], f)


def load_adj_matrices(spatial_path1, spatial_path2, feature1_path, feature2_path):
    """Load adjacency matrices from specified paths."""
    with open(spatial_path1, 'rb') as f:
        spatial_adj1 = pickle.load(f)

    with open(spatial_path2, 'rb') as f:
        spatial_adj2 = pickle.load(f)

    with open(feature1_path, 'rb') as f:
        adj_feature_omics1 = pickle.load(f)

    with open(feature2_path, 'rb') as f:
        adj_feature_omics2 = pickle.load(f)

    adj = {
        'adj_spatial_omics1': spatial_adj1,
        'adj_spatial_omics2': spatial_adj2,
        'adj_feature_omics1': adj_feature_omics1,
        'adj_feature_omics2': adj_feature_omics2,
    }
    return adj


def adjacent_matrix_preprocessing(adata_omics1, adata_omics2, adj_path):
    """Preprocess adjacency matrices for spatial and feature graphs."""
    # File paths
    spatial_path1 = os.path.join(adj_path, 'adj_spatial_omics1.npy')
    spatial_path2 = os.path.join(adj_path, 'adj_spatial_omics2.npy')
    feature1_path = os.path.join(adj_path, 'adj_feature_omics1.npy')
    feature2_path = os.path.join(adj_path, 'adj_feature_omics2.npy')

    if all(os.path.exists(path) for path in [spatial_path1, spatial_path2, feature1_path, feature2_path]):
        print("Loading Adj Matrix...")
        adj_spatial_omics1 = np.load(spatial_path1)
        adj_spatial_omics2 = np.load(spatial_path2)
        adj_feature_omics1 = np.load(feature1_path)
        adj_feature_omics2 = np.load(feature2_path)
    else:
        print("Constructing Adj Matrix...")
        # construct spatial graph
        adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
        adj_spatial_omics1 = construct_graph_from_adjacent(adj_spatial_omics1)
        adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
        adj_spatial_omics2 = construct_graph_from_adjacent(adj_spatial_omics2)

        adj_spatial_omics1 = adj_spatial_omics1.toarray()
        adj_spatial_omics2 = adj_spatial_omics2.toarray()

        adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
        adj_spatial_omics1 = np.where(adj_spatial_omics1 > 1, 1, adj_spatial_omics1)
        adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
        adj_spatial_omics2 = np.where(adj_spatial_omics2 > 1, 1, adj_spatial_omics2)

        # construct feature graph
        adj_feature_omics1 = torch.FloatTensor(adata_omics1.obsm['adj_feature'].copy().toarray())
        adj_feature_omics2 = torch.FloatTensor(adata_omics2.obsm['adj_feature'].copy().toarray())

        adj_feature_omics1 = adj_feature_omics1 + adj_feature_omics1.T
        adj_feature_omics1 = np.where(adj_feature_omics1 > 1, 1, adj_feature_omics1)
        adj_feature_omics2 = adj_feature_omics2 + adj_feature_omics2.T
        adj_feature_omics2 = np.where(adj_feature_omics2 > 1, 1, adj_feature_omics2)

        # saving adj matrix
        np.save(spatial_path1, adj_spatial_omics1)
        np.save(spatial_path2, adj_spatial_omics2)
        np.save(feature1_path, adj_feature_omics1)
        np.save(feature2_path, adj_feature_omics2)

    adj = {
        'adj_spatial_omics1': adj_spatial_omics1,
        'adj_spatial_omics2': adj_spatial_omics2,
        'adj_feature_omics1': adj_feature_omics1,
        'adj_feature_omics2': adj_feature_omics2,
    }

    return adj


def construct_graph_from_adjacent(adjacent):
    """Construct sparse graph from adjacent dataframe."""
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj


def degree_power(A, k):
    """Compute degree matrix raised to power k."""
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    """Normalize adjacency matrix."""
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def standardize_coords(coords, eps=1e-8):
    coords = coords.astype(np.float32)
    mean = coords.mean(axis=0, keepdims=True)
    std = coords.std(axis=0, keepdims=True)
    return (coords - mean) / (std + eps)


def build_soft_spatial_adj(x_feat, coords, spatial_mask, tau_x=0.5, tau_s=-1.0, eps=1e-8):
    x_feat = x_feat.astype(np.float32)
    coords = coords.astype(np.float32)

    mask = (spatial_mask > 0).astype(np.float32)

    x_norm = x_feat / (np.linalg.norm(x_feat, axis=1, keepdims=True) + eps)
    sim = x_norm @ x_norm.T

    sq = np.sum(coords ** 2, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (coords @ coords.T)
    dist2 = np.maximum(dist2, 0.0)

    if tau_s is None or tau_s <= 0:
        edge_dist2 = dist2[mask > 0]
        tau_s_val = float(np.median(edge_dist2)) if edge_dist2.size else 1.0
    else:
        tau_s_val = float(tau_s)

    tau_x_val = float(tau_x) if tau_x is not None and tau_x > 0 else 0.5

    w = np.exp(sim / (tau_x_val + eps)) * np.exp(-dist2 / (tau_s_val + eps))
    w = w * mask

    return norm_adj(w)


def target_distribution(Q):
    """Compute target distribution for KL divergence."""
    weight = Q ** 2 / Q.sum(0)
    P = (weight.t() / weight.sum(1)).t()
    return P


def distribution_loss(Q, P):
    """Compute KL divergence loss for clustering guidance."""
    loss = F.kl_div((Q[0].log() + Q[1].log() + Q[2].log()) / 3, P, reduction='batchmean')
    return loss


def assignment(Q, y):
    """Assign clusters and evaluate."""
    y_pred = torch.argmax(Q, dim=1).data.cpu().numpy()
    if y is not None:
        acc, f1, nmi, ari, ami, vms, fms = eval(y, y_pred)
        return acc, f1, nmi, ari, ami, vms, fms, y_pred
    else:
        return None, None, None, None, None, None, None, y_pred


def clustering(Z, y=None, n_clusters=None):
    """Perform KMeans clustering and return cluster centers."""
    if y is not None and len(y) > 0:
        model = KMeans(n_clusters=len(np.unique(y)), n_init=10)
    else:
        if n_clusters is None:
            raise ValueError("n_clusters must be specified when y is None or empty.")
        model = KMeans(n_clusters=n_clusters, n_init=10)
    cluster_id = model.fit_predict(Z.data.cpu().numpy())

    return model.cluster_centers_
