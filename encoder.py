# -*- coding:utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.functional import normalize
import numpy as np
from torch.nn.modules.module import Module
from torch.nn import Linear
from moe import MoEFusion


class GraphConvolutionLayer(nn.Module):
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


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )

        out = self.fc_out(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.self_attention(x, x, x, mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(query)
        return out


class trans_encoder(nn.Module):
    def __init__(self, embed_size, laten_size, num_layers, heads, forward_expansion, dropout, max_length):
        super(trans_encoder, self).__init__()
        self.embed_size = embed_size
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [EncoderLayer(embed_size, heads, forward_expansion, dropout) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(embed_size, laten_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length, _ = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        x = self.dropout((x + self.position_embedding(positions)))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.linear(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.self_attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(query)
        return out


class trans_decoder(nn.Module):
    def __init__(self, input_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length):
        super(trans_decoder, self).__init__()
        self.embed_size = embed_size
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(embed_size, heads, forward_expansion, dropout) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(embed_size, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length, _ = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        x = self.dropout((x + self.position_embedding(positions)))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        x = self.linear(x)
        return x


class q_distribution(nn.Module):
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
    def __init__(
        self,
        input_dim1,
        input_dim2,
        enc_dim1,
        enc_dim2,
        dec_dim1,
        dec_dim2,
        latent_dim,
        dropout,
        num_layers,
        num_heads1,
        num_heads2,
        n_clusters,
        n_node=None,
        cross_fusion: str = "var",
        moe_num_experts: int = 4,
        moe_hidden_dim: int = 128,
        emb_global_attn: int = 1,
        emb_attn_mask: int = 1,
        emb_attn_temp: float = 1.0,
        emb_attn_dropout: float = 0.0,
        emb_alpha_tanh: int = 1,
    ):
        super(GCNAutoencoder, self).__init__()
        self.cross_fusion = cross_fusion
        self.emb_global_attn = bool(emb_global_attn)
        self.emb_attn_mask = bool(emb_attn_mask)
        self.emb_attn_temp = max(float(emb_attn_temp), 1e-6)
        self.emb_attn_dropout = float(emb_attn_dropout)
        self.emb_alpha_tanh = bool(emb_alpha_tanh)
        self.cross_moe = MoEFusion(
            latent_dim,
            latent_dim,
            latent_dim,
            num_experts=moe_num_experts,
            hidden_dim=moe_hidden_dim,
        )
        self.encoder_view1 = GCNEncoder(
            input_dim=input_dim1,
            enc_dim1=enc_dim1,
            enc_dim2=enc_dim2,
            latent_dim=latent_dim,
            dropout=dropout
        )

        self.encoder_view2 = GCNEncoder(
            input_dim=input_dim2,
            enc_dim1=enc_dim1,
            enc_dim2=enc_dim2,
            latent_dim=latent_dim,
            dropout=dropout
        )

        self.trans_encoder1 = trans_encoder(embed_size=input_dim1,
                                            laten_size=latent_dim,
                                            num_layers=num_layers,
                                            heads=num_heads1,
                                            forward_expansion=num_heads1,
                                            dropout=dropout,
                                            max_length=25000
                                            )

        self.trans_encoder2 = trans_encoder(embed_size=input_dim2,
                                            laten_size=latent_dim,
                                            num_layers=num_layers,
                                            heads=num_heads1,
                                            forward_expansion=num_heads1,
                                            dropout=dropout,
                                            max_length=25000
                                            )

        self.trans_decoder1 = trans_decoder(input_size=input_dim1,
                                            embed_size=latent_dim,
                                            num_layers=num_layers,
                                            heads=num_heads1,
                                            forward_expansion=num_heads1,
                                            dropout=dropout,
                                            max_length=25000
                                            )
        self.trans_decoder2 = trans_decoder(input_size=input_dim2,
                                            embed_size=latent_dim,
                                            num_layers=num_layers,
                                            heads=num_heads2,
                                            forward_expansion=num_heads2,
                                            dropout=dropout,
                                            max_length=25000
                                            )

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

        self.a = Parameter(nn.init.constant_(torch.zeros(n_node, 20), 0.5), requires_grad=True)
        self.b = Parameter(nn.init.constant_(torch.zeros(n_node, 20), 0.5), requires_grad=True)
        self.c = Parameter(nn.init.constant_(torch.zeros(n_node, 20), 0.5), requires_grad=True)
        self.alpha = Parameter(torch.zeros(1))  # ZG, ZL

        self.cluster_centers1 = Parameter(torch.Tensor(n_clusters, latent_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_centers1.data)
        self.q_distribution1 = q_distribution(self.cluster_centers1)

        self.k1 = Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.k2 = Parameter(torch.FloatTensor([0.5]), requires_grad=True)

        self.latent_process = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )

    def emb_fusion(self, adj, z_1, z_2, z_3):
        total = self.a + self.b + self.c
        a_normalized = self.a / total
        b_normalized = self.b / total
        c_normalized = self.c / total

        z_i = a_normalized * z_1 + b_normalized * z_2 + c_normalized * z_3
        z_l = torch.spmm(adj, z_i)

        if not self.emb_global_attn:
            return z_l

        # Stable, locality-aware attention:
        # - Normalize features (cosine similarity)
        # - Optional adjacency mask (prevents global majority pooling)
        z_attn = F.normalize(z_l, dim=1, eps=1e-12)
        scores = torch.mm(z_attn, z_attn.t()) / self.emb_attn_temp

        if self.emb_attn_mask:
            mask = adj > 0
            mask = mask | torch.eye(mask.size(0), device=mask.device, dtype=torch.bool)
            scores = scores.masked_fill(~mask, -1e9)

        attn = F.softmax(scores, dim=1)
        if self.emb_attn_dropout and self.emb_attn_dropout > 0:
            attn = F.dropout(attn, p=self.emb_attn_dropout, training=self.training)

        z_g = torch.mm(attn, z_l)
        alpha = torch.tanh(self.alpha) if self.emb_alpha_tanh else self.alpha
        return alpha * z_g + z_l

    def forward(self, x1, adj1, adj2, x2, adj3, adj4, Mt1, Mt2, pretrain=False):
        adj2 = self.k1 * adj2 + self.k2 * Mt1
        adj4 = self.k1 * adj4 + self.k2 * Mt2

        z11, z_adj1 = self.encoder_view1(x1, adj1)
        z12, z_adj2 = self.encoder_view1(x1, adj2)
        z13 = self.trans_encoder1(x1.unsqueeze(0), mask=None)
        z13 = z13.squeeze(0)

        z21, z_adj3 = self.encoder_view2(x2, adj3)
        z22, z_adj4 = self.encoder_view2(x2, adj4)
        z23 = self.trans_encoder2(x2.unsqueeze(0), mask=None)
        z23 = z23.squeeze(0)

        z1_tilde = self.emb_fusion(adj2, z11, z12, z13)
        z2_tilde = self.emb_fusion(adj4, z21, z22, z23)

        z1_tilde = self.latent_process(z1_tilde)
        z2_tilde = self.latent_process(z2_tilde)

        w1 = torch.var(z1_tilde)
        w2 = torch.var(z2_tilde)
        a1 = w1 / (w1 + w2)
        a2 = 1 - a1
        Z_var = torch.add(z1_tilde * a1, z2_tilde * a2)

        if self.cross_fusion == "moe":
            Z, gate_weights, balance_loss = self.cross_moe(z1_tilde, z2_tilde)
        else:
            Z, gate_weights, balance_loss = Z_var, None, torch.tensor(0.0, device=x1.device)

        # decoder
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

        if pretrain:
            Q = None
        else:
            Q = self.q_distribution1(Z, z1_tilde, z2_tilde)

        return Z, z1_tilde, z2_tilde, a11_hat, a12_hat, a21_hat, a22_hat, x13_hat, x23_hat, Q, gate_weights, balance_loss
