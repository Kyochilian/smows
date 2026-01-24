# -*- coding:utf-8 -*-
"""
SpaFusion Training Script
Complete training pipeline for SpaFusion model
Author: polaris (original)
Adapted for SpaMICS comparison experiments
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import json
import sys
from datetime import datetime
import warnings

from spafusion_encoder import GCNAutoencoder
from spafusion_utils import (
    setup_seed, adjacent_matrix_preprocessing, norm_adj,
    target_distribution, distribution_loss, clustering, assignment,
    build_soft_spatial_adj, standardize_coords
)
from spafusion_high_order import process_adjacency_matrix
from spafusion_data_adapter import load_data_for_spafusion
from evaluation import eval
import spafusion_opt as opt

warnings.filterwarnings("ignore")


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


def pre_train(model, x1, x2, spatial_adj1, feature_adj1, spatial_adj2, feature_adj2, Mt1, Mt2, coords,
              y, n_clusters, num_epoch, device, weight_list, lr):
    """
    Pretrain the SpaFusion model.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pretrain_loss = []
    for epoch in range(num_epoch):
        Z, z1_tilde, z2_tilde, a11_hat, a12_hat, a21_hat, a22_hat, x13_hat, x23_hat, _, _ = \
            model(x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2, coords=coords, pretrain=True)

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
        pretrain_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Pretrain Epoch: {epoch + 1}/{num_epoch}, Loss: {loss.item():.6f}")

    return model, pretrain_loss


def train(model, x1, x2, spatial_adj1, feature_adj1, spatial_adj2, feature_adj2, Mt1, Mt2, coords,
          y, n_clusters, num_epoch, lambda1, lambda2, device, weight_list, lr, run_dir):
    """
    Train the SpaFusion model with clustering guidance.
    """
    # Initialize cluster centers
    with torch.no_grad():
        Z, z1_tilde, z2_tilde, _, _, _, _, _, _, _, _ = \
            model(x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2, coords=coords)

    centers1 = clustering(Z, y, n_clusters=n_clusters)
    model.cluster_centers1.data = torch.tensor(centers1).to(device)

    train_losses = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_metrics = None
    best_epoch = 0
    final_pred = None

    for epoch in range(num_epoch):
        Z, z1_tilde, z2_tilde, a11_hat, a12_hat, a21_hat, a22_hat, x13_hat, x23_hat, Q, gates = \
            model(x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2, coords=coords, pretrain=False)
        
        loss_ae1 = F.mse_loss(a11_hat, spatial_adj1)
        loss_ae2 = F.mse_loss(a12_hat, feature_adj1)
        loss_ae3 = F.mse_loss(a21_hat, spatial_adj2)
        loss_ae4 = F.mse_loss(a22_hat, feature_adj2)
        loss_x1 = F.mse_loss(x13_hat, x1)
        loss_x2 = F.mse_loss(x23_hat, x2)
        if gates is None:
            dense_loss = torch.mean((Z - z1_tilde) ** 2) + torch.mean((Z - z2_tilde) ** 2)
        else:
            per1 = torch.mean((Z - z1_tilde) ** 2, dim=1)
            per2 = torch.mean((Z - z2_tilde) ** 2, dim=1)
            dense_loss = torch.mean(gates[:, 0] * per1 + gates[:, 1] * per2)
        
        loss_rec = (weight_list[0] * loss_ae1 + weight_list[1] * loss_ae2 + 
                   weight_list[2] * loss_ae3 + weight_list[3] * loss_ae4 + 
                   weight_list[4] * loss_x1 + weight_list[5] * loss_x2)
        
        L_KL1 = distribution_loss(Q, target_distribution(Q[0].data))
        loss = loss_rec + lambda1 * L_KL1 + lambda2 * dense_loss

        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Train Epoch: {epoch + 1}/{num_epoch}, Loss: {loss.item():.6f}")

        # Evaluate and save best model
        if (epoch + 1) % 500 == 0 or epoch == num_epoch - 1:
            if y is not None:
                acc, f1, nmi, ari, ami, vms, fms, y_pred = assignment(Q[0].data, y)
                print(f"  -> ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}")
                
                if best_metrics is None or acc > best_metrics['ACC']:
                    best_metrics = {
                        'ACC': float(acc),
                        'F1': float(f1),
                        'NMI': float(nmi),
                        'ARI': float(ari),
                        'AMI': float(ami),
                        'epoch': epoch + 1
                    }
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
                    np.save(os.path.join(run_dir, 'best_predictions.npy'), y_pred)
            else:
                y_pred = torch.argmax(Q[0].data, dim=1).data.cpu().numpy()
            
            final_pred = y_pred

    return model, train_losses, best_metrics, best_epoch, final_pred, Z


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="SpaFusion Training")
    parser.add_argument('--name', type=str, default='Human_Lymph_Node_A1', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--spatial_k', type=int, default=9, help='Spatial neighbors')
    parser.add_argument('--adj_k', type=int, default=20, help='Feature neighbors')
    parser.add_argument('--soft_spatial', type=int, default=1, help='1 to use soft-weight spatial graph, 0 for binary')
    parser.add_argument('--tau_x', type=float, default=0.5, help='Feature similarity temperature for soft spatial weights')
    parser.add_argument('--tau_s', type=float, default=-1.0, help='Spatial distance temperature (<=0 for auto)')
    parser.add_argument('--lambda1', type=float, default=1, help='KL loss weight')
    parser.add_argument('--lambda2', type=float, default=0.1, help='Dense loss weight')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--pretrain_epoch', type=int, default=10000, help='Pretrain epochs')
    parser.add_argument('--train_epoch', type=int, default=350, help='Train epochs')
    parser.add_argument('--skip_pretrain', action='store_true', help='Skip pretraining phase')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of training runs after pretraining')

    parser.add_argument('--fusion_mode', type=str, default='cell_gate', help='Fusion mode: variance, cell_gate, or multiscale_unet')
    parser.add_argument('--ms_depth', type=int, default=3, help='Depth of multiscale UNet fusion')
    parser.add_argument('--ms_dims', type=str, default='', help='Comma-separated dims for multiscale fusion, empty=auto')
    parser.add_argument('--ms_reducer', type=str, default='mlp', help='Reducer type: mlp or gcn')
    parser.add_argument('--ms_use_adj', type=int, default=1, help='1 to use adjacency propagation in fusion, 0 otherwise')
    parser.add_argument('--ms_adj_type', type=str, default='mixed', help='Adjacency type for fusion propagation')
    args_input = parser.parse_args()
    
    # Update opt with input args
    opt.args.name = args_input.name
    opt.args.device = args_input.device
    opt.args.seed = args_input.seed
    opt.args.spatial_k = args_input.spatial_k
    opt.args.adj_k = args_input.adj_k
    opt.args.lambda1 = args_input.lambda1
    opt.args.lambda2 = args_input.lambda2
    opt.args.lr = args_input.lr
    opt.args.pretrain_epoch = args_input.pretrain_epoch
    opt.args.train_epoch = args_input.train_epoch
    opt.args.skip_pretrain = args_input.skip_pretrain

    opt.args.soft_spatial = bool(args_input.soft_spatial)
    opt.args.tau_x = args_input.tau_x
    opt.args.tau_s = args_input.tau_s

    opt.args.fusion_mode = args_input.fusion_mode
    opt.args.ms_depth = args_input.ms_depth
    opt.args.ms_reducer = args_input.ms_reducer
    opt.args.ms_use_adj = bool(args_input.ms_use_adj)
    opt.args.ms_adj_type = args_input.ms_adj_type
    if args_input.ms_dims:
        opt.args.ms_dims = [int(x) for x in args_input.ms_dims.split(',') if x.strip()]
    else:
        opt.args.ms_dims = None
    
    device = torch.device(opt.args.device if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("SpaFusion Configuration")
    print("=" * 60)
    print(f"Dataset        : {opt.args.name}")
    print(f"Device         : {device}")
    print(f"Seed           : {opt.args.seed}")
    print(f"Spatial K      : {opt.args.spatial_k}")
    print(f"Feature K      : {opt.args.adj_k}")
    print(f"Lambda1        : {opt.args.lambda1}")
    print(f"Lambda2        : {opt.args.lambda2}")
    print(f"Learning Rate  : {opt.args.lr}")
    print(f"Pretrain Epochs: {opt.args.pretrain_epoch}")
    print(f"Train Epochs   : {opt.args.train_epoch}")
    print("=" * 60)
    
    setup_seed(opt.args.seed)
    
    # Create results directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(results_dir, f"SpaFusion_{opt.args.name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results will be saved to: {run_dir}")
    
    # Load data
    print("\n" + "=" * 60)
    print("Loading Data...")
    print("=" * 60)
    data = load_data_for_spafusion(opt.args.name, opt.args.spatial_k, opt.args.adj_k)
    adata_omics1 = data['adata_omics1']
    adata_omics2 = data['adata_omics2']
    label = data['label']
    
    # Get features
    data1 = adata_omics1.obsm['feat'].copy()
    data2 = adata_omics2.obsm['feat'].copy()
    coords = standardize_coords(adata_omics1.obsm['spatial'].copy())
    
    # Determine number of clusters
    if label is not None:
        n_clusters = len(np.unique(label))
    else:
        n_clusters = 5  # Default
        
    print(f"n_clusters: {n_clusters}")
    print(f"data1 shape: {data1.shape}")
    print(f"data2 shape: {data2.shape}")
    
    # Construct adjacency matrices
    adj_path = os.path.join(run_dir, 'pre_adj')
    os.makedirs(adj_path, exist_ok=True)
    adj = adjacent_matrix_preprocessing(adata_omics1, adata_omics2, adj_path)
    
    feature_adj1 = adj['adj_feature_omics1']
    feature_adj2 = adj['adj_feature_omics2']
    spatial_adj1 = adj['adj_spatial_omics1']
    spatial_adj2 = adj['adj_spatial_omics2']
    
    # Compute high-order matrices
    Mt1 = process_adjacency_matrix(feature_adj1, os.path.join(adj_path, 'Mt1.npy'))
    Mt2 = process_adjacency_matrix(feature_adj2, os.path.join(adj_path, 'Mt2.npy'))
    
    # Normalize adjacency matrices
    feature_adj1 = norm_adj(feature_adj1)
    feature_adj2 = norm_adj(feature_adj2)
    if opt.args.soft_spatial:
        spatial_adj1 = build_soft_spatial_adj(data1, coords, spatial_adj1, tau_x=opt.args.tau_x, tau_s=opt.args.tau_s)
        spatial_adj2 = build_soft_spatial_adj(data2, coords, spatial_adj2, tau_x=opt.args.tau_x, tau_s=opt.args.tau_s)
    else:
        spatial_adj1 = norm_adj(spatial_adj1)
        spatial_adj2 = norm_adj(spatial_adj2)
    Mt1 = norm_adj(Mt1)
    Mt2 = norm_adj(Mt2)
    
    # Convert to tensors
    data1 = torch.tensor(data1, dtype=torch.float32).to(device)
    data2 = torch.tensor(data2, dtype=torch.float32).to(device)
    feature_adj1 = torch.tensor(feature_adj1, dtype=torch.float32).to(device)
    feature_adj2 = torch.tensor(feature_adj2, dtype=torch.float32).to(device)
    spatial_adj1 = torch.tensor(spatial_adj1, dtype=torch.float32).to(device)
    spatial_adj2 = torch.tensor(spatial_adj2, dtype=torch.float32).to(device)
    Mt1 = torch.tensor(Mt1, dtype=torch.float32).to(device)
    Mt2 = torch.tensor(Mt2, dtype=torch.float32).to(device)
    coords = torch.tensor(coords, dtype=torch.float32).to(device)

    # Initialize model
    model = GCNAutoencoder(
        input_dim1=data1.shape[1], 
        input_dim2=data2.shape[1], 
        enc_dim1=opt.args.enc_dim1, 
        enc_dim2=opt.args.enc_dim2, 
        dec_dim1=opt.args.dec_dim1,
        dec_dim2=opt.args.dec_dim2, 
        latent_dim=opt.args.latent_dim, 
        dropout=opt.args.dropout, 
        num_layers=opt.args.num_layers, 
        num_heads1=opt.args.num_heads1, 
        num_heads2=opt.args.num_heads2,
        n_clusters=n_clusters, 
        n_node=data1.shape[0],
        fusion_mode=opt.args.fusion_mode,
        ms_depth=opt.args.ms_depth,
        ms_dims=opt.args.ms_dims,
        ms_reducer=opt.args.ms_reducer,
        ms_use_adj=opt.args.ms_use_adj,
        ms_adj_type=opt.args.ms_adj_type,
    )

    if not opt.args.skip_pretrain:
        # Pretraining
        print("\n" + "=" * 60)
        print("Pretraining...")
        print("=" * 60)
        model, pretrain_losses = pre_train(
            model=model,
            x1=data1, x2=data2, 
            spatial_adj1=spatial_adj1, feature_adj1=feature_adj1,
            spatial_adj2=spatial_adj2, feature_adj2=feature_adj2, 
            Mt1=Mt1, Mt2=Mt2, coords=coords,
            y=label, n_clusters=n_clusters,
            num_epoch=opt.args.pretrain_epoch, 
            device=device, 
            weight_list=opt.args.weight_list, 
            lr=opt.args.lr
        )
        
        # Save pretrain model
        torch.save(model.state_dict(), os.path.join(run_dir, 'pretrain_model.pth'))
    else:
        print("\n" + "=" * 60)
        print("Skipping Pretraining...")
        print("=" * 60)
        model.to(device)
        pretrain_losses = []
    
    # Start log capture
    log_path = os.path.join(run_dir, 'training_log.txt')
    tee = TeeOutput(log_path)
    sys.stdout = tee
    
    # Save pretrained model state for reuse
    pretrained_state = model.state_dict().copy()
    
    # Multi-run training
    n_runs = args_input.n_runs
    all_metrics = {'ACC': [], 'F1': [], 'NMI': [], 'ARI': [], 'AMI': [], 'VMS': [], 'FMS': []}
    best_overall_metrics = None
    best_overall_pred = None
    best_overall_Z = None
    
    for run_idx in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Training Run {run_idx + 1}/{n_runs}")
        print("="*60)
        
        # Reset model to pretrained state
        model.load_state_dict(pretrained_state)
        setup_seed(opt.args.seed + run_idx)  # Different seed for each run
        
        model, train_losses, best_metrics, best_epoch, final_pred, Z = train(
            model=model,
            x1=data1, x2=data2,
            spatial_adj1=spatial_adj1, feature_adj1=feature_adj1,
            spatial_adj2=spatial_adj2, feature_adj2=feature_adj2,
            Mt1=Mt1, Mt2=Mt2, coords=coords,
            y=label, n_clusters=n_clusters,
            num_epoch=opt.args.train_epoch,
            lambda1=opt.args.lambda1,
            lambda2=opt.args.lambda2,
            device=device,
            weight_list=opt.args.weight_list,
            lr=opt.args.lr,
            run_dir=run_dir
        )
        
        # Collect metrics
        if label is not None and best_metrics is not None:
            all_metrics['ACC'].append(best_metrics['ACC'])
            all_metrics['F1'].append(best_metrics['F1'])
            all_metrics['NMI'].append(best_metrics['NMI'])
            all_metrics['ARI'].append(best_metrics['ARI'])
            all_metrics['AMI'].append(best_metrics['AMI'])
            
            # Compute VMS and FMS for this run
            acc, f1, nmi, ari, ami, vms, fms = eval(label, final_pred)
            all_metrics['VMS'].append(vms)
            all_metrics['FMS'].append(fms)
            
            print(f"Run {run_idx + 1} Best: ACC={best_metrics['ACC']:.4f}, NMI={best_metrics['NMI']:.4f}, ARI={best_metrics['ARI']:.4f}")
            
            # Track best overall
            if best_overall_metrics is None or best_metrics['ACC'] > best_overall_metrics['ACC']:
                best_overall_metrics = best_metrics
                best_overall_pred = final_pred
                best_overall_Z = Z
    
    # Save best model and results
    torch.save(model.state_dict(), os.path.join(run_dir, 'final_model.pth'))
    if best_overall_pred is not None:
        np.save(os.path.join(run_dir, 'final_predictions.npy'), best_overall_pred)
    if best_overall_Z is not None:
        np.save(os.path.join(run_dir, 'latent_features.npy'), best_overall_Z.data.cpu().numpy())
    np.save(os.path.join(run_dir, 'pretrain_losses.npy'), np.array(pretrain_losses))
    np.save(os.path.join(run_dir, 'train_losses.npy'), np.array(train_losses))
    
    # Print Performance Summary
    print("\n" + "="*60)
    print(f"Performance Summary ({n_runs} runs):")
    print("="*60)
    for metric in ['ACC', 'F1', 'NMI', 'ARI', 'AMI', 'VMS', 'FMS']:
        if all_metrics[metric]:
            mean_val = np.mean(all_metrics[metric])
            std_val = np.std(all_metrics[metric])
            print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    print("="*60)
    
    # Save summary
    summary = {
        'method': 'SpaFusion',
        'dataset': opt.args.name,
        'timestamp': timestamp,
        'n_clusters': int(n_clusters),
        'seed': int(opt.args.seed),
        'n_runs': n_runs,
        'spatial_k': int(opt.args.spatial_k),
        'adj_k': int(opt.args.adj_k),
        'lambda1': float(opt.args.lambda1),
        'lambda2': float(opt.args.lambda2),
        'learning_rate': float(opt.args.lr),
        'pretrain_epoch': int(opt.args.pretrain_epoch),
        'train_epoch': int(opt.args.train_epoch),
    }
    
    if label is not None and all_metrics['ACC']:
        summary['performance_summary'] = {
            metric: {'mean': float(np.mean(all_metrics[metric])), 'std': float(np.std(all_metrics[metric]))}
            for metric in ['ACC', 'F1', 'NMI', 'ARI', 'AMI', 'VMS', 'FMS'] if all_metrics[metric]
        }
        if best_overall_metrics is not None:
            summary['best_metrics'] = best_overall_metrics
    
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Comparison with Paper
    print("\n==================== Comparison with Paper ====================")
    if "Human_Lymph_Node_A1" in opt.args.name or opt.args.name == "A1":
        paper = {"ARI": 0.319, "NMI": 0.399, "ACC": 0.629, "AMI": 0.396, "F1": 0.337}
        dataset_short = "A1"
    elif "Human_Lymph_Node_D1" in opt.args.name or opt.args.name == "D1":
        paper = {"ARI": 0.351, "NMI": 0.384, "ACC": 0.599, "AMI": 0.379, "F1": 0.323}
        dataset_short = "D1"
    else:
        paper = None
        dataset_short = opt.args.name

    if paper and all_metrics['ACC']:
        print(f"Dataset: {dataset_short}")
        print("{:<10} {:<15} {:<10}".format("Metric", "Ours (mean±std)", "Paper"))
        print("-" * 40)
        for m in ["ACC", "F1", "NMI", "ARI", "AMI"]:
            mean_val = np.mean(all_metrics[m])
            std_val = np.std(all_metrics[m])
            val_paper = paper.get(m, "-")
            print("{:<10} {:.4f}±{:.4f}    {:<10}".format(m, mean_val, std_val, val_paper))
    else:
        print(f"No paper data available for dataset: {dataset_short}")
    print("="*60)

    # Visualization (once at end)
    print("\n" + "="*60)
    print("Generating Visualizations...")
    print("="*60)
    try:
        from visualize_spatial import plot_combined_visualizations, plot_modality_weights
        import scanpy as sc
        
        adata_vis = adata_omics1.copy()
        if best_overall_Z is not None:
            adata_vis.obsm['X_emb'] = best_overall_Z.data.cpu().numpy()
        if best_overall_pred is not None:
            adata_vis.obs['SpaFusion'] = best_overall_pred.astype(str)
            adata_vis.obs['SpaFusion'] = adata_vis.obs['SpaFusion'].astype('category')
        
        plot_combined_visualizations(adata_vis, key='SpaFusion', save_dir=run_dir, prefix='SpaFusion')
        adata_vis.write(os.path.join(run_dir, 'spafusion_results.h5ad'))
        print(f"Saved AnnData with results to: {os.path.join(run_dir, 'spafusion_results.h5ad')}")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nAll results saved to: {run_dir}")
    print("="*60)
    
    # Close log capture
    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()
    print(f"Training log saved to: {log_path}")



if __name__ == '__main__':
    main()
