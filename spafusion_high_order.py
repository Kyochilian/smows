# -*- coding:utf-8 -*-
"""
SpaFusion High-Order Matrix Computation
Author: polaris (original)
Adapted for SpaMICS comparison experiments

This module computes high-order adjacency matrices based on 3-node motifs.
Includes caching mechanism to avoid recomputation.
"""

import os
import time
import numpy as np


def find_3_node_motifs(adjacency_matrix):
    """
    Find all 3-node motifs (triangles) in the adjacency matrix.
    
    Args:
        adjacency_matrix: Binary adjacency matrix
        
    Returns:
        List of tuples (i, j, k) representing triangles
    """
    motifs = []
    num_nodes = adjacency_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if adjacency_matrix[i, j]:
                for k in range(j+1, num_nodes):
                    if adjacency_matrix[i, k] and adjacency_matrix[j, k]:
                        motifs.append((i, j, k))
    return motifs


def save_matrix(matrix, filename):
    """Save matrix to file."""
    np.save(filename, matrix)


def load_matrix(filename):
    """Load matrix from file."""
    return np.load(filename)


def process_adjacency_matrix(adjacency_matrix, filename):
    """
    Process adjacency matrix to compute high-order matrix based on 3-node motifs.
    Uses caching to avoid recomputation.
    
    Args:
        adjacency_matrix: Binary adjacency matrix
        filename: Path to save/load the cached matrix
        
    Returns:
        High-order adjacency matrix Mt
    """
    if os.path.exists(filename):
        print(f"Loading cached Mt: {filename}")
        return load_matrix(filename)
    else:
        print("Computing high-order matrix (3-node motifs)...")
        start_time = time.time()
        three_node_motifs = find_3_node_motifs(adjacency_matrix=adjacency_matrix)
        end_time = time.time()
        time_ = end_time - start_time
        print("3-node motifs: {}, time: {:.2f}s".format(len(three_node_motifs), time_))
        
        num_nodes = adjacency_matrix.shape[0]
        Mt = np.zeros((num_nodes, num_nodes), dtype=int)
        for i, j, k in three_node_motifs:
            Mt[i][j] += 1
            Mt[i][k] += 1
            Mt[j][k] += 1
            Mt[j][i] += 1
            Mt[k][i] += 1
            Mt[k][j] += 1
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_matrix(Mt, filename)
        print(f"Saved Mt to: {filename}")
        return Mt
