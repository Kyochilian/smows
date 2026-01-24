# -*- coding:utf-8 -*-
"""
SpaFusion Data Adapter
Adapts SpaMICS data loading to SpaFusion format
"""

import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import scipy


def load_data_for_spafusion(name, spatial_k=9, adj_k=20):
    """
    Load and preprocess data for SpaFusion model.
    
    Args:
        name: Dataset name
        spatial_k: Number of neighbors for spatial graph
        adj_k: Number of neighbors for feature graph
        
    Returns:
        Dictionary containing all necessary data for SpaFusion
    """
    # Load data based on dataset name
    if name in ["Human_Lymph_Node_A1", "Human_Lymph_Node_D1"]:
        adata_omics1, adata_omics2, label = load_human_lymph_node(name)
        datatype = 'RNA-ADT'
    elif name in ["Human_tonsil_1", "Human_tonsil_3"]:
        adata_omics1, adata_omics2, label = load_human_tonsil(name)
        datatype = 'RNA-ADT'
    elif name in ["Mouse_Brain_E15", "Mouse_Brain_E18"]:
        adata_omics1, adata_omics2, label = load_mouse_brain(name)
        datatype = 'RNA-ATAC'
    elif name == 'Human_Breast_Cancer':
        adata_omics1, adata_omics2, label = load_human_breast_cancer(name)
        datatype = 'RNA-ADT'
    elif name == 'Human_Melanoma':
        adata_omics1, adata_omics2, label = load_human_melanoma(name)
        datatype = 'RNA-ATAC'
    else:
        raise ValueError(f"Dataset {name} not supported")
    
    # Preprocess data
    adata_omics1, adata_omics2 = preprocess_for_spafusion(
        adata_omics1, adata_omics2, datatype, n_neighbors=spatial_k, k=adj_k
    )
    
    return {
        'adata_omics1': adata_omics1,
        'adata_omics2': adata_omics2,
        'label': label,
        'datatype': datatype
    }


def preprocess_for_spafusion(adata_omics1, adata_omics2, datatype, n_neighbors=9, k=20):
    """
    Preprocess data following SpaFusion's original pipeline.
    """
    if datatype == 'RNA-ADT':
        # RNA
        print("Processing RNA...")
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=100)

        # Protein
        print("Processing Protein...")
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        
    elif datatype == 'RNA-ATAC':
        # RNA
        print("Processing RNA...")
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=100)
        
        # ATAC
        print("Processing ATAC...")
        sc.pp.filter_genes(adata_omics2, min_cells=int(adata_omics2.shape[0] * 0.06))
        if 'highly_variable' not in adata_omics2.var:
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=min(100, adata_omics2.n_vars - 1))

    # Construct spatial graphs
    print("Constructing spatial graphs...")
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = build_network(cell_position_omics1, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_omics1

    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = build_network(cell_position_omics2, n_neighbors=n_neighbors)
    adata_omics2.uns['adj_spatial'] = adj_omics2

    # Construct feature graphs
    print("Constructing feature graphs...")
    feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2, k=k)
    adata_omics1.obsm['adj_feature'] = feature_graph_omics1
    adata_omics2.obsm['adj_feature'] = feature_graph_omics2

    return adata_omics1, adata_omics2


def build_network(cell_position, n_neighbors=3):
    """Construct spatial neighbor graph according to spatial coordinates."""
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj


def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode="connectivity", metric="euclidean",
                               include_self=True):
    """Construct feature neighbor graph according to expression profiles."""
    feature_graph_omics1 = kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric,
                                            include_self=include_self)
    feature_graph_omics2 = kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric,
                                            include_self=include_self)
    return feature_graph_omics1, feature_graph_omics2


def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell using CLR normalization."""
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def pca(adata, use_reps=None, n_comps=10):
    """Dimension reduction with PCA algorithm."""
    pca_model = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca_model.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat_pca = pca_model.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca_model.fit_transform(adata.X)
    return feat_pca


# Dataset loading functions
def load_human_lymph_node(name):
    """Load human lymph node data."""
    adata_rna = sc.read_h5ad(f'./data/10X/{name}/adata_RNA.h5ad')
    adata_adt = sc.read_h5ad(f'./data/10X/{name}/adata_ADT.h5ad')
    adata_rna.var_names_make_unique()
    adata_adt.var_names_make_unique()
    
    # Try to load label.npy first, fallback to CSV if not found
    import os
    label_npy_path = f'./data/10X/{name}/label.npy'
    if os.path.exists(label_npy_path):
        label = np.load(label_npy_path)
    else:
        # Try CSV format (e.g., D1_annotation_labels.csv)
        csv_files = [f for f in os.listdir(f'./data/10X/{name}') if f.endswith('_annotation_labels.csv')]
        if csv_files:
            label_df = pd.read_csv(f'./data/10X/{name}/{csv_files[0]}')
            label = label_df['labels'].values
        else:
            raise FileNotFoundError(f"No label file found for {name}")
    
    return adata_rna, adata_adt, label


def load_mouse_brain(name):
    """Load mouse brain data."""
    adata_rna = sc.read_h5ad(f'./data/MISAR/{name}/adata_RNA.h5ad')
    adata_atac = sc.read_h5ad(f'./data/MISAR/{name}/adata_ATAC.h5ad')
    adata_rna.var_names_make_unique()
    adata_atac.var_names_make_unique()
    label = adata_rna.obs['Combined_Clusters']
    return adata_rna, adata_atac, label


def load_human_tonsil(name):
    """Load human tonsil data."""
    adata_rna = sc.read_h5ad(f'./data/10X/{name}/adata_RNA.h5ad')
    adata_adt = sc.read_h5ad(f'./data/10X/{name}/adata_ADT.h5ad')
    adata_rna.var_names_make_unique()
    adata_adt.var_names_make_unique()
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels_numeric = label_encoder.fit_transform(adata_rna.obs['final_annot'])
    return adata_rna, adata_adt, labels_numeric


def load_human_breast_cancer(name):
    """Load human breast cancer data."""
    adata_rna = sc.read_h5ad(f'./data/10X/{name}/adata_RNA.h5ad')
    adata_adt = sc.read_h5ad(f'./data/10X/{name}/adata_ADT.h5ad')
    adata_rna.var_names_make_unique()
    adata_adt.var_names_make_unique()
    label = None
    return adata_rna, adata_adt, label


def load_human_melanoma(name):
    """Load human melanoma data."""
    adata_rna = sc.read_h5ad(f'./data/MISAR/{name}/adata_RNA.h5ad')
    adata_atac = sc.read_h5ad(f'./data/MISAR/{name}/adata_ATAC.h5ad')
    adata_rna.var_names_make_unique()
    adata_atac.var_names_make_unique()
    label = adata_rna.obs['cell_type'].astype('category').cat.codes
    return adata_rna, adata_atac, label
