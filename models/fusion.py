# -*- coding:utf-8 -*-
"""
Author: kyochilian
Date: 2026.02

SpaFusion Model
This module contains the main SpaFusion fusion model that combines GCN and Transformer components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .gcn import GCNEncoder, GCNDecoder
from .transformer import TransformerEncoder, TransformerDecoder


class q_distribution(nn.Module):
    """Q-distribution for clustering"""
    
    def __init__(self, centers):
        super(q_distribution, self).__init__()
        self.cluster_centers = centers

    def forward(self, z, z1, z2):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_centers, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z1.unsqueeze(1) - self.cluster_centers, 2), 2))
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        q2 = 1.0 / (1.0 + torch.sum(torch.pow(z2.unsqueeze(1) - self.cluster_centers, 2), 2))
        q2 = (q2.t() / torch.sum(q2, 1)).t()

        return [q, q1, q2]


class GCNAutoencoder(nn.Module):
    """
    SpaFusion Main Model
    A multi-level fusion model combining GCN and Transformer for spatial multi-omics data
    """
    
    def __init__(self, input_dim1, input_dim2, enc_dim1, enc_dim2, dec_dim1, dec_dim2, latent_dim, dropout,
                 num_layers, num_heads1, num_heads2, n_clusters, n_node=None):
        super(GCNAutoencoder, self).__init__()
        
        # View 1 (e.g., RNA) Encoders
        self.encoder_view1 = GCNEncoder(
            input_dim=input_dim1,
            enc_dim1=enc_dim1,
            enc_dim2=enc_dim2,
            latent_dim=latent_dim,
            dropout=dropout
        )

        # View 2 (e.g., Protein) Encoders
        self.encoder_view2 = GCNEncoder(
            input_dim=input_dim2,
            enc_dim1=enc_dim1,
            enc_dim2=enc_dim2,
            latent_dim=latent_dim,
            dropout=dropout
        )

        # Transformer Encoder for View 1
        self.trans_encoder1 = TransformerEncoder(
            embed_size=input_dim1,
            laten_size=latent_dim,
            num_layers=num_layers,
            heads=num_heads1,
            forward_expansion=num_heads1,
            dropout=dropout,
            max_length=25000
        )

        # Transformer Encoder for View 2
        self.trans_encoder2 = TransformerEncoder(
            embed_size=input_dim2,
            laten_size=latent_dim,
            num_layers=num_layers,
            heads=num_heads1,
            forward_expansion=num_heads1,
            dropout=dropout,
            max_length=25000
        )

        # Transformer Decoder for View 1
        self.trans_decoder1 = TransformerDecoder(
            input_size=input_dim1,
            embed_size=latent_dim,
            num_layers=num_layers,
            heads=num_heads1,
            forward_expansion=num_heads1,
            dropout=dropout,
            max_length=25000
        )
        
        # Transformer Decoder for View 2
        self.trans_decoder2 = TransformerDecoder(
            input_size=input_dim2,
            embed_size=latent_dim,
            num_layers=num_layers,
            heads=num_heads2,
            forward_expansion=num_heads2,
            dropout=dropout,
            max_length=25000
        )

        # GCN Decoders
        self.decoder_view1 = GCNDecoder(
            latent_dim=latent_dim,
            dec_dim1=dec_dim1,
            dec_dim2=dec_dim2,
            output_dim=input_dim1
        )

        self.decoder_view2 = GCNDecoder(
            latent_dim=latent_dim,
            dec_dim1=dec_dim1,
            dec_dim2=dec_dim2,
            output_dim=input_dim2
        )

        # Fusion parameters - CRITICAL: Must match latent_dim!
        self.a = Parameter(nn.init.constant_(torch.zeros(n_node, latent_dim), 0.5), requires_grad=True)
        self.b = Parameter(nn.init.constant_(torch.zeros(n_node, latent_dim), 0.5), requires_grad=True)
        self.c = Parameter(nn.init.constant_(torch.zeros(n_node, latent_dim), 0.5), requires_grad=True)
        self.alpha = Parameter(torch.zeros(1))

        # Clustering parameters
        self.cluster_centers1 = Parameter(torch.Tensor(n_clusters, latent_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_centers1.data)
        self.q_distribution1 = q_distribution(self.cluster_centers1)

        # Graph weight parameters
        self.k1 = Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.k2 = Parameter(torch.FloatTensor([0.5]), requires_grad=True)

        # Latent processing
        self.latent_process = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )

    def emb_fusion(self, adj, z_1, z_2, z_3):
        """Embedding fusion with adaptive weighting"""
        total = self.a + self.b + self.c
        a_normalized = self.a / total
        b_normalized = self.b / total
        c_normalized = self.c / total

        z_i = a_normalized * z_1 + b_normalized * z_2 + c_normalized * z_3
        z_l = torch.spmm(adj, z_i)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)
        z_g = torch.mm(s, z_l)
        z_tilde = self.alpha * z_g + z_l
        return z_tilde

    def forward(self, x1, adj1, adj2, x2, adj3, adj4, Mt1, Mt2, pretrain=False):
        """
        Forward pass
        
        Args:
            x1: View 1 features (e.g., RNA)
            adj1: View 1 spatial adjacency
            adj2: View 1 feature adjacency
            x2: View 2 features (e.g., Protein)
            adj3: View 2 spatial adjacency
            adj4: View 2 feature adjacency
            Mt1: View 1 high-order graph
            Mt2: View 2 high-order graph
            pretrain: Whether in pretraining mode
        
        Returns:
            Z: Fused embedding
            z1_tilde: View 1 fused embedding
            z2_tilde: View 2 fused embedding
            a11_hat: Reconstructed adjacency matrices
            a12_hat, a21_hat, a22_hat: Reconstructed adjacency matrices
            x13_hat, x23_hat: Reconstructed features
            Q: Cluster distribution (None if pretrain=True)
        """
        # Combine feature and high-order graphs
        adj2 = self.k1 * adj2 + self.k2 * Mt1
        adj4 = self.k1 * adj4 + self.k2 * Mt2

        # View 1 encoding
        z11, z_adj1 = self.encoder_view1(x1, adj1)
        z12, z_adj2 = self.encoder_view1(x1, adj2)
        z13 = self.trans_encoder1(x1.unsqueeze(0), mask=None)
        z13 = z13.squeeze(0)

        # View 2 encoding
        z21, z_adj3 = self.encoder_view2(x2, adj3)
        z22, z_adj4 = self.encoder_view2(x2, adj4)
        z23 = self.trans_encoder2(x2.unsqueeze(0), mask=None)
        z23 = z23.squeeze(0)

        # Fusion
        z1_tilde = self.emb_fusion(adj2, z11, z12, z13)
        z2_tilde = self.emb_fusion(adj4, z21, z22, z23)

        z1_tilde = self.latent_process(z1_tilde)
        z2_tilde = self.latent_process(z2_tilde)

        # Adaptive weighting based on variance
        w1 = torch.var(z1_tilde)
        w2 = torch.var(z2_tilde)
        a1 = w1 / (w1 + w2)
        a2 = 1 - a1
        Z = torch.add(z1_tilde * a1, z2_tilde * a2)

        # Decoding
        x11_hat, adj1_hat = self.decoder_view1(z11, adj1)
        a11_hat = z_adj1 + adj1_hat
        x12_hat, adj2_hat = self.decoder_view1(z12, adj2)
        a12_hat = z_adj2 + adj2_hat
        x13_hat = self.trans_decoder1(x=z1_tilde.unsqueeze(0), enc_out=z1_tilde.unsqueeze(0), src_mask=None, trg_mask=None)
        x13_hat = x13_hat.squeeze(0)

        x21_hat, adj3_hat = self.decoder_view2(z21, adj3)
        a21_hat = z_adj3 + adj3_hat
        x22_hat, adj4_hat = self.decoder_view2(z22, adj4)
        a22_hat = z_adj4 + adj4_hat
        x23_hat = self.trans_decoder2(x=z2_tilde.unsqueeze(0), enc_out=z2_tilde.unsqueeze(0), src_mask=None, trg_mask=None)
        x23_hat = x23_hat.squeeze(0)

        # Clustering (only during training)
        if pretrain:
            Q = None
        else:
            Q = self.q_distribution1(Z, z1_tilde, z2_tilde)

        return Z, z1_tilde, z2_tilde, a11_hat, a12_hat, a21_hat, a22_hat, x13_hat, x23_hat, Q


# Alias for backward compatibility
Spafusion = GCNAutoencoder
