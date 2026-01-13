# -*- coding:utf-8 -*-
"""
Author: kyochilian
Date: 2026.02

GCN (Graph Convolutional Network) Components
This module contains the GCN layer, encoder, and decoder implementations.
"""

import torch
import torch.nn as nn


class GraphConvolutionLayer(nn.Module):
    """Single Graph Convolution Layer with Tanh activation"""
    
    def __init__(self, input_dim, output_dim):
        super(GraphConvolutionLayer, self).__init__()
        self.act = nn.Tanh()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj, active=False):
        if active:
            support = self.act(torch.mm(x, self.weight))
        else:
            support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        return output


class GCNEncoder(nn.Module):
    """GCN-based encoder with 3 layers"""
    
    def __init__(self, input_dim, enc_dim1, enc_dim2, latent_dim, dropout):
        super(GCNEncoder, self).__init__()
        self.layer1 = GraphConvolutionLayer(input_dim, enc_dim1)
        self.layer2 = GraphConvolutionLayer(enc_dim1, enc_dim2)
        self.layer3 = GraphConvolutionLayer(enc_dim2, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.layer1(x, adj, active=True)
        x = self.dropout(x)
        x = self.layer2(x, adj, active=True)
        x = self.dropout(x)
        z_x = self.layer3(x, adj, active=False)
        z_adj = torch.sigmoid(torch.mm(z_x, z_x.t()))
        return z_x, z_adj


class GCNDecoder(nn.Module):
    """GCN-based decoder with 3 layers"""
    
    def __init__(self, latent_dim, dec_dim1, dec_dim2, output_dim):
        super(GCNDecoder, self).__init__()
        self.layer4 = GraphConvolutionLayer(latent_dim, dec_dim1)
        self.layer5 = GraphConvolutionLayer(dec_dim1, dec_dim2)
        self.layer6 = GraphConvolutionLayer(dec_dim2, output_dim)

    def forward(self, z_x, adj):
        x_hat = self.layer4(z_x, adj, active=True)
        x_hat = self.layer5(x_hat, adj, active=True)
        x_hat = self.layer6(x_hat, adj, active=True)
        adj_hat = torch.sigmoid(torch.mm(x_hat, x_hat.t()))
        return x_hat, adj_hat
