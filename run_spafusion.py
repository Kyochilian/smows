# -*- coding:utf-8 -*-
"""
SpaFusion Runner - Integrated into SpaMICS project
Supports Human_Lymph_Node_A1 and Human_Lymph_Node_D1 datasets
Results are stored in results/ directory following project structure
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import scanpy as sc
import os
import sys
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from spafusion.encoder import GCNAutoencoder
from spafusion.high_order_matrix import process_adjacency_matrix
from spafusion.spafusion_processing import load_data
from spafusion.spafusion_utils import (
    setup_seed, adjacent_matrix_preprocessing, norm_adj,
    target_distribution, distribution_loss, assignment, clustering
)
from spafusion.spafusion_evaluate import eva

import spafusion_opt as opt


class TeeOutput:
    """Duplicate stdout to both console and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def load_dataset(name, data_path):
    """Load dataset from the project data structure"""
    full_path = os.path.join(data_path, name)
    
    adata_rna = sc.read_h5ad(os.path.join(full_path, 'adata_RNA.h5ad'))
    adata_adt = sc.read_h5ad(os.path.join(full_path, 'adata_ADT.h5ad'))
    
    adata_rna.var_names_make_unique()
    adata_adt.var_names_make_unique()
    
    label_path = os.path.join(full_path, 'label.npy')
    if os.path.exists(label_path):
        label = np.load(label_path)
    elif name == "Human_Lymph_Node_D1":
        csv_path = os.path.join(full_path, 'D1_annotation_labels.csv')
        if os.path.exists(csv_path):
            labels_df = pd.read_csv(csv_path)
            label = labels_df['labels'].values
        else:
            label = None
    else:
        label = None
    
    return adata_rna, adata_adt, label


def pre_train(x1, x2, spatial_adj1, feature_adj1, spatial_adj2, feature_adj2, 
              Mt1, Mt2, y, n_clusters, num_epoch, device, weight_list, lr, 
              run_dir, args):
    """Pre-training phase"""
    model = GCNAutoencoder(
        input_dim1=x1.shape[1], 
        input_dim2=x2.shape[1], 
        enc_dim1=args.enc_dim1, 
        enc_dim2=args.enc_dim2, 
        dec_dim1=args.dec_dim1,
        dec_dim2=args.dec_dim2, 
        latent_dim=args.latent_dim, 
        dropout=args.dropout, 
        num_layers=args.num_layers, 
        num_heads1=args.num_heads1, 
        num_heads2=args.num_heads2,
        n_clusters=n_clusters, 
        n_node=x1.shape[0]
    )

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pretrain_losses = []
    for epoch in range(num_epoch):
        Z, z1_tilde, z2_tilde, a11_hat, a12_hat, a21_hat, a22_hat, x13_hat, x23_hat, _ = \
            model(x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2, pretrain=True)

        loss_ae1 = F.mse_loss(a11_hat, spatial_adj1)
        loss_ae2 = F.mse_loss(a12_hat, feature_adj1)
        loss_ae3 = F.mse_loss(a21_hat, spatial_adj2)
        loss_ae4 = F.mse_loss(a22_hat, feature_adj2)

        loss_x1 = F.mse_loss(x13_hat, x1)
        loss_x2 = F.mse_loss(x23_hat, x2)

        loss_rec = (weight_list[0] * loss_ae1 + weight_list[1] * loss_ae2 + 
                   weight_list[2] * loss_ae3 + weight_list[3] * loss_ae4 + 
                   weight_list[4] * loss_x1 + weight_list[5] * loss_x2)

        loss = loss_rec
        pretrain_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"Pretrain Epoch: {epoch + 1}/{num_epoch}, loss: {loss.item():.8f}")

    pretrain_dir = os.path.join(run_dir, 'pretrain')
    os.makedirs(pretrain_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, 'pretrain_model.pth'))
    np.save(os.path.join(run_dir, 'pretrain_losses.npy'), np.array(pretrain_losses))
    
    return model, z1_tilde, z2_tilde


def train(model, x1, x2, spatial_adj1, feature_adj1, spatial_adj2, feature_adj2, 
          Mt1, Mt2, y, n_clusters, num_epoch, lambda1, lambda2, device, 
          weight_list, lr, run_dir, args):
    """Training phase"""
    
    with torch.no_grad():
        Z, z1_tilde, z2_tilde, _, _, _, _, _, _, _ = \
            model(x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2)

    centers1 = clustering(Z, y, n_clusters=n_clusters)

    model.cluster_centers1.data = torch.tensor(centers1).to(device)

    train_losses = []
    best_metrics = None
    best_pred = None
    best_Z = None
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        Z, z1_tilde, z2_tilde, a11_hat, a12_hat, a21_hat, a22_hat, x13_hat, x23_hat, Q = \
            model(x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2, pretrain=False)
        
        loss_ae1 = F.mse_loss(a11_hat, spatial_adj1)
        loss_ae2 = F.mse_loss(a12_hat, feature_adj1)
        loss_ae3 = F.mse_loss(a21_hat, spatial_adj2)
        loss_ae4 = F.mse_loss(a22_hat, feature_adj2)
        loss_x1 = F.mse_loss(x13_hat, x1)
        loss_x2 = F.mse_loss(x23_hat, x2)
        dense_loss1 = torch.mean((Z - z1_tilde) ** 2)
        dense_loss2 = torch.mean((Z - z2_tilde) ** 2)
        
        loss_rec = (weight_list[0] * loss_ae1 + weight_list[1] * loss_ae2 + 
                   weight_list[2] * loss_ae3 + weight_list[3] * loss_ae4 + 
                   weight_list[4] * loss_x1 + weight_list[5] * loss_x2)
        
        L_KL1 = distribution_loss(Q, target_distribution(Q[0].data))
        loss = loss_rec + lambda1 * L_KL1 + lambda2 * (dense_loss1 + dense_loss2)

        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"Train Epoch: {epoch + 1}/{num_epoch}, loss: {loss.item():.8f}")
   
        if y is not None:
            acc, f1, nmi, ari, ami, vms, fms, y_pred = assignment(Q[0].data, y)
            
            if best_metrics is None or ari > best_metrics['ARI']:
                best_metrics = {
                    'ACC': float(acc),
                    'F1': float(f1),
                    'NMI': float(nmi),
                    'ARI': float(ari),
                    'AMI': float(ami),
                    'VMS': float(vms),
                    'FMS': float(fms),
                    'epoch': epoch + 1
                }
                best_pred = y_pred.copy()
                best_Z = Z.data.cpu().numpy().copy()
                
                torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
                np.save(os.path.join(run_dir, 'best_predictions.npy'), best_pred)
        else:
            y_pred = torch.argmax(Q[0].data, dim=1).data.cpu().numpy()

    torch.save(model.state_dict(), os.path.join(run_dir, 'final_model.pth'))
    np.save(os.path.join(run_dir, 'final_predictions.npy'), y_pred)
    np.save(os.path.join(run_dir, 'latent_features.npy'), Z.data.cpu().numpy())
    np.save(os.path.join(run_dir, 'train_losses.npy'), np.array(train_losses))
    
    return best_metrics, best_pred, y_pred, Z.data.cpu().numpy()


def main():
    args = opt.args
    opt.print_config()
    
    setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(results_dir, f"SpaFusion_{args.name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Results will be saved to: {run_dir}")
    
    log_path = os.path.join(run_dir, 'training_log.txt')
    tee = TeeOutput(log_path)
    sys.stdout = tee

    opt.print_config()

    print("\n" + "=" * 60)
    print("Loading Dataset...")
    print("=" * 60)
    
    adata_omics1, adata_omics2, label = load_dataset(args.name, args.data_path)
    
    if label is not None:
        n_clusters = len(np.unique(label))
    elif args.n_clusters is not None:
        n_clusters = args.n_clusters
    else:
        n_clusters = 5
        print(f"Warning: No labels found and n_clusters not specified. Using default: {n_clusters}")
    
    print(f"Number of clusters: {n_clusters}")
    
    adata_omics1, adata_omics2 = load_data(
        adata_omics1=adata_omics1, 
        view1="RNA", 
        adata_omics2=adata_omics2, 
        view2="Protein", 
        n_neighbors=args.spatial_k, 
        k=args.adj_k
    )
    
    data1 = adata_omics1.obsm['feat'].copy()
    data2 = adata_omics2.obsm['feat'].copy()

    adj_path = os.path.join(run_dir, 'pre_adj')
    os.makedirs(adj_path, exist_ok=True)
    adj = adjacent_matrix_preprocessing(adata_omics1, adata_omics2, adj_path)

    feature_adj1 = adj['adj_feature_omics1']
    feature_adj2 = adj['adj_feature_omics2']
    spatial_adj1 = adj['adj_spatial_omics1']
    spatial_adj2 = adj['adj_spatial_omics2']

    mt1_path = os.path.join(adj_path, f"{args.name}_Mt1.npy")
    mt2_path = os.path.join(adj_path, f"{args.name}_Mt2.npy")
    Mt1 = process_adjacency_matrix(feature_adj1, mt1_path)
    Mt2 = process_adjacency_matrix(feature_adj2, mt2_path)

    feature_adj1 = norm_adj(feature_adj1)
    feature_adj2 = norm_adj(feature_adj2)
    spatial_adj1 = norm_adj(spatial_adj1)
    spatial_adj2 = norm_adj(spatial_adj2)
    Mt1 = norm_adj(Mt1)
    Mt2 = norm_adj(Mt2)
    
    data1 = torch.tensor(data1, dtype=torch.float32).to(device)
    data2 = torch.tensor(data2, dtype=torch.float32).to(device)
    feature_adj1 = torch.tensor(feature_adj1, dtype=torch.float32).to(device)
    feature_adj2 = torch.tensor(feature_adj2, dtype=torch.float32).to(device)
    spatial_adj1 = torch.tensor(spatial_adj1, dtype=torch.float32).to(device)
    spatial_adj2 = torch.tensor(spatial_adj2, dtype=torch.float32).to(device)
    Mt1 = torch.tensor(Mt1, dtype=torch.float32).to(device)
    Mt2 = torch.tensor(Mt2, dtype=torch.float32).to(device)

    spatial_adj1 = spatial_adj1 * feature_adj1
    spatial_adj2 = spatial_adj2 * feature_adj2

    print("\n" + "=" * 60)
    print("Dataset Information")
    print("=" * 60)
    print(f"n_clusters: {n_clusters}")
    print(f"data1 (RNA) shape: {data1.shape}")
    print(f"data2 (ADT) shape: {data2.shape}")
    print(f"feature_adj1 shape: {feature_adj1.shape}")
    print(f"Mt1 (high-order) shape: {Mt1.shape}")

    weight_list = opt.get_weight_list()

    print("\n" + "=" * 60)
    print("Pre-training Phase")
    print("=" * 60)
    
    model, z1_tilde, z2_tilde = pre_train(
        x1=data1, x2=data2, 
        spatial_adj1=spatial_adj1, feature_adj1=feature_adj1,
        spatial_adj2=spatial_adj2, feature_adj2=feature_adj2, 
        Mt1=Mt1, Mt2=Mt2, 
        y=label, n_clusters=n_clusters,
        num_epoch=args.pretrain_epoch, 
        device=device, 
        weight_list=weight_list, 
        lr=args.lr,
        run_dir=run_dir,
        args=args
    )

    print("\n" + "=" * 60)
    print("Training Phase")
    print("=" * 60)
    
    best_metrics, best_pred, final_pred, latent_Z = train(
        model=model,
        x1=data1, x2=data2, 
        spatial_adj1=spatial_adj1, feature_adj1=feature_adj1, 
        spatial_adj2=spatial_adj2, feature_adj2=feature_adj2, 
        Mt1=Mt1, Mt2=Mt2,
        y=label, n_clusters=n_clusters, 
        num_epoch=args.train_epoch, 
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        device=device, 
        weight_list=weight_list, 
        lr=args.lr,
        run_dir=run_dir,
        args=args
    )

    summary = {
        'model': 'SpaFusion',
        'dataset': args.name,
        'timestamp': timestamp,
        'n_clusters': int(n_clusters),
        'seed': int(args.seed),
        'pretrain_epoch': int(args.pretrain_epoch),
        'train_epoch': int(args.train_epoch),
        'learning_rate': float(args.lr),
        'lambda1': float(args.lambda1),
        'lambda2': float(args.lambda2),
        'spatial_k': int(args.spatial_k),
        'adj_k': int(args.adj_k),
        'weight_list': weight_list,
        'enc_dim1': int(args.enc_dim1),
        'enc_dim2': int(args.enc_dim2),
        'latent_dim': int(args.latent_dim),
        'dropout': float(args.dropout),
    }
    
    if label is not None and best_metrics is not None:
        summary['best_metrics'] = best_metrics
        
        acc, f1, nmi, ari, ami, vms, fms = eva(label, final_pred)
        summary['final_metrics'] = {
            'ACC': float(acc),
            'F1': float(f1),
            'NMI': float(nmi),
            'ARI': float(ari),
            'AMI': float(ami),
            'VMS': float(vms),
            'FMS': float(fms)
        }
        
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"Best results (epoch {best_metrics['epoch']}):")
        print(f"  ACC: {best_metrics['ACC']:.4f}")
        print(f"  F1:  {best_metrics['F1']:.4f}")
        print(f"  NMI: {best_metrics['NMI']:.4f}")
        print(f"  ARI: {best_metrics['ARI']:.4f}")
        print(f"  AMI: {best_metrics['AMI']:.4f}")
        print("-" * 60)
        print("Final results:")
        print(f"  ACC: {acc:.4f}")
        print(f"  F1:  {f1:.4f}")
        print(f"  NMI: {nmi:.4f}")
        print(f"  ARI: {ari:.4f}")
        print(f"  AMI: {ami:.4f}")
        print("=" * 60)

        if "Human_Lymph_Node_A1" in args.name:
            print("\n==================== Comparison with Paper (SpaFusion) ====================")
            print(f"Dataset: Human_Lymph_Node_A1")
            print("{:<15} {:<10} {:<10} {:<10} {:<10}".format("Method", "ACC", "NMI", "ARI", "AMI"))
            print("{:<15} {:<10} {:<10} {:<10} {:<10}".format("SpaFusion Paper", "0.6173", "0.4171", "0.3696", "0.4128"))
            print("{:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                "Ours (best)", 
                best_metrics['ACC'], 
                best_metrics['NMI'], 
                best_metrics['ARI'], 
                best_metrics['AMI']
            ))
            print("=" * 70)
        elif "Human_Lymph_Node_D1" in args.name:
            print("\n==================== Comparison with Paper (SpaFusion) ====================")
            print(f"Dataset: Human_Lymph_Node_D1")
            print("{:<15} {:<10} {:<10} {:<10} {:<10}".format("Method", "ACC", "NMI", "ARI", "AMI"))
            print("{:<15} {:<10} {:<10} {:<10} {:<10}".format("SpaFusion Paper", "0.6139", "0.4371", "0.3587", "0.4329"))
            print("{:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                "Ours (best)", 
                best_metrics['ACC'], 
                best_metrics['NMI'], 
                best_metrics['ARI'], 
                best_metrics['AMI']
            ))
            print("=" * 70)

    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nAll results saved to: {run_dir}")
    
    sys.stdout = tee.terminal
    tee.close()
    print(f"Training log saved to: {log_path}")
    print("=" * 60)
    print("SpaFusion training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
