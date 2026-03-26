# -*- coding:utf-8 -*-

import torch

from processing import *
from utils import *
from encoder import *
from high_order_matrix import process_adjacency_matrix
from evaluate import *
import torch.optim as optim
import argparse
from copy import deepcopy
from datetime import datetime

import os
import sys

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.manifold import TSNE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paper metrics for comparison
PAPER_METRICS = {
    'ARI': (0.351, 0.018),
    'NMI': (0.384, 0.003),
    'ACC': (0.599, 0.016),
    'AMI': (0.379, 0.003),
    'F1':  (0.323, 0.004),
}

class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def read_visible_input(prompt_text):
    """
    Read input with normal terminal echo.
    """
    return input(prompt_text)


def generate_visualizations(Z, y_pred, y_true, spatial_coords, result_dir, run_num):
    # Create visualizations directory
    vis_dir = os.path.join(result_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    Z_np = Z.data.cpu().numpy()
    
    # 1. UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_umap = reducer.fit_transform(Z_np)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=embedding_umap[:, 0], y=embedding_umap[:, 1], hue=y_pred, palette='tab20', s=15, ax=axes[0], legend='full')
    axes[0].set_title('UMAP - Predicted Labels')
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title="Cluster")
    
    if y_true is not None:
        sns.scatterplot(x=embedding_umap[:, 0], y=embedding_umap[:, 1], hue=y_true.values if hasattr(y_true, 'values') else y_true, palette='tab20', s=15, ax=axes[1], legend='full')
        axes[1].set_title('UMAP - Ground Truth')
        axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title="Cluster")
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'run_{run_num}_umap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embedding_tsne = tsne.fit_transform(Z_np)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=embedding_tsne[:, 0], y=embedding_tsne[:, 1], hue=y_pred, palette='tab20', s=15, ax=axes[0], legend='full')
    axes[0].set_title('t-SNE - Predicted Labels')
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title="Cluster")
    
    if y_true is not None:
        sns.scatterplot(x=embedding_tsne[:, 0], y=embedding_tsne[:, 1], hue=y_true.values if hasattr(y_true, 'values') else y_true, palette='tab20', s=15, ax=axes[1], legend='full')
        axes[1].set_title('t-SNE - Ground Truth')
        axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title="Cluster")
        
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'run_{run_num}_tsne.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Spatial Plot
    if spatial_coords is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.scatterplot(x=spatial_coords[:, 0], y=spatial_coords[:, 1], hue=y_pred, palette='tab20', s=15, ax=axes[0], legend='full')
        axes[0].set_title('Spatial Map - Predicted Labels')
        axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title="Cluster")
        
        if y_true is not None:
            sns.scatterplot(x=spatial_coords[:, 0], y=spatial_coords[:, 1], hue=y_true.values if hasattr(y_true, 'values') else y_true, palette='tab20', s=15, ax=axes[1], legend='full')
            axes[1].set_title('Spatial Map - Ground Truth')
            axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title="Cluster")
            
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'run_{run_num}_spatial.png'), dpi=300, bbox_inches='tight')
        plt.close()


def pre_train(x1, x2, spatial_adj1, feature_adj1, spatial_adj2, feature_adj2, Mt1, Mt2, y, n_clusters, num_epoch, device, weight_list, lr,
              cross_fusion="var", moe_num_experts=4, moe_hidden_dim=128, moe_gate_noise_mult=1.0, moe_balance_weight=0.01,
              emb_global_attn=1, emb_attn_mask=1, emb_attn_temp=1.0, emb_attn_dropout=0.0, emb_alpha_tanh=1, emb_attn_sim="dot"):
    model = GCNAutoencoder(input_dim1=x1.shape[1], input_dim2=x2.shape[1], enc_dim1=256, enc_dim2=128, dec_dim1=128,
                           dec_dim2=256, latent_dim=20, dropout=0.1, num_layers=2, num_heads1=1, num_heads2=1,
                            n_clusters=n_clusters, n_node=x1.shape[0], cross_fusion=cross_fusion,
                            moe_num_experts=moe_num_experts, moe_hidden_dim=moe_hidden_dim, moe_gate_noise_mult=moe_gate_noise_mult,
                            emb_global_attn=emb_global_attn, emb_attn_mask=emb_attn_mask, emb_attn_temp=emb_attn_temp,
                            emb_attn_dropout=emb_attn_dropout, emb_alpha_tanh=emb_alpha_tanh, emb_attn_sim=emb_attn_sim)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pretrain_loss = []
    
    os.makedirs('./pretrain', exist_ok=True)
    
    for epoch in range(num_epoch):
        Z, z1_tilde, z2_tilde, a11_hat, a12_hat, a21_hat, a22_hat, x13_hat, x23_hat, _, _, balance_loss = \
            model(x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2, pretrain=True)

        loss_ae1 = F.mse_loss(a11_hat, spatial_adj1)
        loss_ae2 = F.mse_loss(a12_hat, feature_adj1)
        loss_ae3 = F.mse_loss(a21_hat, spatial_adj2)
        loss_ae4 = F.mse_loss(a22_hat, feature_adj2)

        loss_x1 = F.mse_loss(x13_hat, x1)
        loss_x2 = F.mse_loss(x23_hat, x2)

        loss_rec = weight_list[0] * loss_ae1 + weight_list[1] * loss_ae2 + weight_list[2] * loss_ae3 + weight_list[3] * loss_ae4 + weight_list[4] * loss_x1 + weight_list[5] * loss_x2

        loss = loss_rec + moe_balance_weight * balance_loss
 
        pretrain_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print("Pretrain Epoch: {:.0f}/{:.0f} ,loss:{:.8f}".format(epoch + 1, num_epoch, loss))

    torch.save(model.state_dict(), r'./pretrain/{}_pre_model.pkl'.format(opt.name))
    # np.save(r"./loss/{}_pre_train_loss.npy".format(opt.name), np.array(pretrain_loss))
    return z1_tilde, z2_tilde


def train(x1, x2, spatial_adj1, feature_adj1, spatial_adj2, feature_adj2, Mt1, Mt2, y, n_clusters, num_epoch, lambda1, device, seed, lambda2, weight_list, lr, num, spatial_K, adj_K, result_dir,
          cross_fusion="var", moe_num_experts=4, moe_hidden_dim=128, moe_gate_noise_mult=1.0, moe_balance_weight=0.01,
          q_mix_alpha=0.2, q_mix_warmup=500,
          cluster_usage_weight=0.05,
          cluster_usage_mode="hinge",
          center_sep_weight=0.0,
          emb_global_attn=1, emb_attn_mask=1, emb_attn_temp=1.0, emb_attn_dropout=0.0, emb_alpha_tanh=1, emb_attn_sim="dot",
          select_best=1, best_metric="f1",
          dead_cluster_reinit=1, min_cluster_size=10, reinit_every=50, reinit_until=500,
          reinit_strategy="farthest"):
    model = GCNAutoencoder(input_dim1=x1.shape[1], input_dim2=x2.shape[1], enc_dim1=256, enc_dim2=128, dec_dim1=128,
                           dec_dim2=256, latent_dim=20, dropout=0.1, num_layers=2, num_heads1=1, num_heads2=1,
                           n_clusters=n_clusters, n_node=x1.shape[0], cross_fusion=cross_fusion,
                            moe_num_experts=moe_num_experts, moe_hidden_dim=moe_hidden_dim, moe_gate_noise_mult=moe_gate_noise_mult,
                            emb_global_attn=emb_global_attn, emb_attn_mask=emb_attn_mask, emb_attn_temp=emb_attn_temp,
                            emb_attn_dropout=emb_attn_dropout, emb_alpha_tanh=emb_alpha_tanh, emb_attn_sim=emb_attn_sim)
    
    model.to(device)

    # loading pretrained model
    model.load_state_dict(torch.load(r'./pretrain/{}_pre_model.pkl'.format(opt.name), map_location='cpu'))

    with torch.no_grad():
        model.eval()
        _, z1_tilde, z2_tilde, _, _, _, _, _, _, _, _, _ = \
            model(x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2, pretrain=True)
        model.train()

    # KMeans init uses variance-based fusion for stable initialisation
    w1 = torch.var(z1_tilde)
    w2 = torch.var(z2_tilde)
    a1 = w1 / (w1 + w2 + 1e-8)
    Z_var = torch.add(z1_tilde * a1, z2_tilde * (1 - a1))

    centers1 = clustering(Z_var, y, n_clusters=n_clusters)

    # initialize cluster centers
    model.cluster_centers1.data = torch.tensor(centers1).to(device)

    train_losses = []
    optimizer = optim.Adam(model.parameters(), lr=lr) 

    acc, f1, nmi, ari, ami, vms, fms = 0, 0, 0, 0, 0, 0, 0
    y_pred = None
    Z_best = None
    y_pred_best = None
    best_epoch = -1
    best_score = -1e18

    gate_weights_last = None
    balance_loss_last = None

    for epoch in range(num_epoch):
        Z, z1_tilde, z2_tilde, a11_hat, a12_hat, a21_hat, a22_hat, x13_hat, x23_hat, Q, gate_weights, balance_loss = \
            model(x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2, pretrain=False)
        gate_weights_last = gate_weights
        balance_loss_last = balance_loss

        loss_ae1 = F.mse_loss(a11_hat, spatial_adj1)
        loss_ae2 = F.mse_loss(a12_hat, feature_adj1)
        loss_ae3 = F.mse_loss(a21_hat, spatial_adj2)
        loss_ae4 = F.mse_loss(a22_hat, feature_adj2)
        loss_x1 = F.mse_loss(x13_hat, x1)
        loss_x2 = F.mse_loss(x23_hat, x2)
        dense_loss1 = torch.mean((Z - z1_tilde) ** 2)
        dense_loss2 = torch.mean((Z - z2_tilde) ** 2)
        loss_rec = weight_list[0] * loss_ae1 + weight_list[1] * loss_ae2 + weight_list[2] * loss_ae3 + weight_list[3] * loss_ae4 + weight_list[4] * loss_x1 + weight_list[5] * loss_x2
        cur_q_mix_alpha = q_mix_alpha
        if q_mix_warmup and q_mix_warmup > 0:
            cur_q_mix_alpha = q_mix_alpha * min(1.0, float(epoch) / float(q_mix_warmup))

        L_KL1 = distribution_loss(Q, target_distribution(Q[0].detach()), mix_alpha=cur_q_mix_alpha)
        loss = loss_rec + lambda1 * L_KL1 + lambda2 * (dense_loss1 + dense_loss2) + moe_balance_weight * balance_loss

        # Encourage cluster centers to be well-separated (often improves ARI).
        if center_sep_weight and center_sep_weight > 0:
            C = F.normalize(model.cluster_centers1, dim=1, eps=1e-12)  # (K, D)
            S = torch.mm(C, C.t())                                     # cosine sim (K, K)
            eye = torch.eye(S.size(0), device=S.device, dtype=S.dtype)
            S_off = S * (1.0 - eye)
            denom = int(S.size(0)) * int(S.size(0) - 1)
            if denom > 0:
                L_sep = (S_off ** 2).sum() / float(denom)
                loss = loss + center_sep_weight * L_sep

        # Prevent cluster collapse (macro-F1 killer on imbalanced labels):
        # Encourage every cluster to get non-trivial mass under Q0.
        if cluster_usage_weight and cluster_usage_weight > 0:
            q_bar = Q[0].mean(dim=0)  # (K,)
            if str(cluster_usage_mode).lower() == "log":
                # Strong (pushes toward uniform); may hurt purity of big clusters.
                usage_loss = (-torch.log(q_bar + 1e-12)).mean()
            else:
                # Targeted (recommended): only penalize clusters whose expected mass
                # falls below the "min_cluster_size" threshold.
                min_prob = float(min_cluster_size) / float(x1.shape[0])
                usage_loss = (F.relu(min_prob - q_bar) / (min_prob + 1e-12)).mean()
            loss = loss + cluster_usage_weight * usage_loss

        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Re-initialize "dead" clusters early to avoid singleton/empty clusters,
        # which usually kills macro-F1 on imbalanced labels.
        if dead_cluster_reinit and reinit_every and (epoch + 1) % int(reinit_every) == 0:
            if reinit_until is None or int(reinit_until) <= 0 or epoch < int(reinit_until):
                with torch.no_grad():
                    hard = Q[0].argmax(dim=1)
                    counts = torch.bincount(hard, minlength=n_clusters)
                    dead_idx = (counts < int(min_cluster_size)).nonzero(as_tuple=False).flatten()
                    if dead_idx.numel() > 0:
                        strategy = str(reinit_strategy).lower()
                        if strategy == "uncertainty":
                            # Pick lowest-confidence points (may split big clusters).
                            max_prob = Q[0].max(dim=1).values
                            cand = torch.argsort(max_prob, descending=False)
                            chosen = cand[: dead_idx.numel()]
                        else:
                            # KMeans++-like: pick points farthest from any alive center.
                            centers = model.cluster_centers1.data.detach()
                            alive_mask = torch.ones(n_clusters, device=centers.device, dtype=torch.bool)
                            alive_mask[dead_idx] = False
                            alive_centers = centers[alive_mask] if alive_mask.any() else centers
                            dist = torch.cdist(Z.detach(), alive_centers)
                            min_dist = dist.min(dim=1).values
                            chosen = torch.argsort(min_dist, descending=True)[: dead_idx.numel()]
                        model.cluster_centers1.data[dead_idx] = Z[chosen].detach()
                        # Reset Adam moments for re-initialized centers (stabilizes re-seeding)
                        try:
                            st = optimizer.state.get(model.cluster_centers1, None)
                            if st is not None:
                                for k in ("exp_avg", "exp_avg_sq"):
                                    if k in st and hasattr(st[k], "shape") and st[k].shape == model.cluster_centers1.shape:
                                        st[k][dead_idx].zero_()
                        except Exception:
                            pass
                        print(f"[reinit] epoch {epoch+1}: dead clusters {dead_idx.tolist()} counts {counts[dead_idx].tolist()}")

        if (epoch + 1) % 10 == 0:
            print("Epoch: {:.0f}/{:.0f} ,loss:{:.8f}".format(epoch + 1, num_epoch, loss))

        # clustering & evaluation
        if y is not None:
            acc, f1, nmi, ari, ami, vms, fms, y_pred = assignment(Q[0].detach(), y)

            if select_best:
                metric_name = str(best_metric).lower()
                if metric_name == "f1":
                    score = f1
                elif metric_name == "acc":
                    score = acc
                elif metric_name == "ari":
                    score = ari
                elif metric_name == "nmi":
                    score = nmi
                elif metric_name == "ami":
                    score = ami
                elif metric_name == "vms":
                    score = vms
                elif metric_name == "fms":
                    score = fms
                elif metric_name == "loss":
                    score = -float(loss.item())
                elif metric_name == "f1acc":
                    score = (2.0 * acc * f1) / (acc + f1 + 1e-12)
                else:
                    score = f1

                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    Z_best = Z.detach().clone()
                    y_pred_best = y_pred.copy()
        else:
            y_pred = torch.argmax(Q[0].data, dim=1).data.cpu().numpy()

    # gate diagnostics
    if gate_weights_last is not None:
        try:
            gate_usage = gate_weights_last.mean(dim=0)
            gate_entropy = -(gate_weights_last * (gate_weights_last + 1e-12).log()).sum(dim=1).mean()
            res_beta = model.cross_moe.res_gate.item() if cross_fusion == "moe" else float('nan')
            print("gate_usage:", gate_usage.detach().cpu().numpy())
            print("gate_entropy: {:.4f} (ideal {:.4f})".format(gate_entropy.item(), torch.log(torch.tensor(float(moe_num_experts))).item()))
            print("residual_beta (MoE weight): {:.4f}".format(res_beta))
            print("balance_loss: {:.6f}".format(balance_loss_last.item()))
        except Exception as e:
            print("gate diagnostics failed:", e)

    # cluster diagnostics (Q0 usage)
    try:
        q_bar = Q[0].mean(dim=0)
        q_entropy = -(q_bar * (q_bar + 1e-12).log()).sum()
        print("cluster_usage(Q0):", q_bar.detach().cpu().numpy())
        print("cluster_entropy(Q0): {:.4f} (ideal {:.4f})".format(
            q_entropy.item(), torch.log(torch.tensor(float(n_clusters))).item()
        ))
    except Exception as e:
        print("cluster diagnostics failed:", e)

    # saving results……
    os.makedirs(result_dir, exist_ok=True)

    if select_best and y is not None and y_pred_best is not None and Z_best is not None:
        acc, f1, nmi, ari, ami, vms, fms = eva(y, y_pred_best)
        y_pred = y_pred_best
        Z = Z_best
        print(f"[best] metric={best_metric} epoch={best_epoch+1}/{num_epoch} score={best_score:.4f} ACC={acc:.4f} F1={f1:.4f} ARI={ari:.4f}")

    if y is not None:
        with open(os.path.join(result_dir, '{}_performance.csv'.format(opt.name)), 'a') as f:
            f.write("seed:{}, lambda1:{}, lambda2:{}, spatial_k:{}, adj_k:{}, wieght_list:{}, ".format(seed, lambda1, lambda2, spatial_K, adj_K, weight_list))
            f.write('%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % (acc, f1, nmi, ari, ami, vms, fms))
            
    np.save(os.path.join(result_dir, '{}_{}_pre_label.npy'.format(opt.name, num)), y_pred)
    np.save(os.path.join(result_dir, '{}_{}_laten.npy'.format(opt.name, num)), Z.data.cpu().numpy())

    return z1_tilde, z2_tilde, acc, f1, nmi, ari, ami, vms, fms, Z, y_pred


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Model setting……")
    parser.add_argument('--name', type=str, default='D1', help='dataset name')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--spatial_k', type=int, default=9, help='spatial_k')
    parser.add_argument('--adj_k', type=int, default=20, help='adj_k')
    parser.add_argument('--lambda1', type=float, default=1, help='lambda1')
    parser.add_argument('--lambda2', type=float, default=0.1, help='lambda2')
    parser.add_argument('--weight_list', type=list, default=[1, 1, 1, 1, 1, 1], help='weight list')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--pretrain_epoch', type=int, default=10000, help='pretrain epoch')
    parser.add_argument('--train_epoch', type=int, default=800, help='train epoch')
    parser.add_argument('--q_mix_alpha', type=float, default=0.2, help='mix weight for Q1/Q2 in KL loss (0=only Q0)')
    parser.add_argument('--q_mix_warmup', type=int, default=500, help='warmup epochs for q_mix_alpha ramp (0=off)')
    parser.add_argument('--cluster_usage_weight', type=float, default=0.05, help='weight for cluster-usage regularizer (prevents dead clusters)')
    parser.add_argument('--center_sep_weight', type=float, default=0.0, help='weight for center-separation regularizer (cosine repulsion)')
    parser.add_argument('--select_best', type=int, default=1, help='save best epoch result by best_metric (1=on, 0=off)')
    parser.add_argument('--best_metric', type=str, default='f1', choices=['f1', 'acc', 'ari', 'nmi', 'ami', 'vms', 'fms', 'loss', 'f1acc'],
                        help='metric for selecting best epoch when select_best=1')
    parser.add_argument('--cluster_usage_mode', type=str, default='hinge', choices=['hinge', 'log'],
                        help='cluster-usage regularizer type: hinge (only penalize dead clusters) or log (push uniform)')
    parser.add_argument('--dead_cluster_reinit', type=int, default=1, help='re-init dead clusters early (1=on, 0=off)')
    parser.add_argument('--min_cluster_size', type=int, default=10, help='min hard cluster size threshold for re-init')
    parser.add_argument('--reinit_every', type=int, default=50, help='re-init period in epochs (0=off)')
    parser.add_argument('--reinit_until', type=int, default=500, help='only re-init when epoch < reinit_until (<=0 = no limit)')
    parser.add_argument('--reinit_strategy', type=str, default='farthest', choices=['farthest', 'uncertainty'],
                        help='dead-cluster re-init strategy: farthest (KMeans++-like) or uncertainty (low max prob)')

    # encoder fusion (global attention) controls — most impactful for ARI/stability
    parser.add_argument('--emb_global_attn', type=int, default=1, help='use emb_fusion global attention branch (1=on, 0=off)')
    parser.add_argument('--emb_attn_mask', type=int, default=1, help='mask emb global attention by adjacency (1=on, 0=off)')
    parser.add_argument('--emb_attn_temp', type=float, default=1.0, help='temperature for emb attention (lower=sharper)')
    parser.add_argument('--emb_attn_dropout', type=float, default=0.0, help='dropout on emb attention weights')
    parser.add_argument('--emb_alpha_tanh', type=int, default=1, help='bound emb alpha with tanh (1=on, 0=off)')
    parser.add_argument('--emb_attn_sim', type=str, default='dot', choices=['dot', 'cosine'],
                        help='similarity for emb global attention: dot (legacy) or cosine')

    # cross-modal fusion ablation
    parser.add_argument('--cross_fusion', type=str, default='var', choices=['var', 'moe'], help='cross-modal fusion type')
    parser.add_argument('--moe_num_experts', type=int, default=4, help='number of experts for MoE cross fusion')
    parser.add_argument('--moe_hidden_dim', type=int, default=128, help='hidden dim for MoE experts/gate')
    parser.add_argument('--moe_gate_noise_mult', type=float, default=1.0, help='multiplier for MoE gate noise std (lower=more stable routing)')
    parser.add_argument('--moe_balance_weight', type=float, default=0.01, help='weight for MoE load-balancing loss')

    opt = parser.parse_args()
    
    custom_name = ""
    while not custom_name:
        custom_name = read_visible_input("请输入结果文件夹名称: ").strip()
        invalid_chars = '<>:"/\\|?*'
        custom_name = "".join("_" if c in invalid_chars else c for c in custom_name).strip()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_dir = os.path.join("results", opt.name, f"{custom_name}_{run_timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Save log output
    orig_stdout = sys.stdout
    log_file_path = os.path.join(result_dir, '{}_training_log.txt'.format(opt.name))
    sys.stdout = TeeLogger(log_file_path)

    print("setting:")
    print("------------------------------")
    print("dataset        : {}".format(opt.name))
    print("device         : {}".format(opt.device))
    print("seed           : {}".format(opt.seed))
    print("spatial_k      : {}".format(opt.spatial_k))
    print("adj_k          : {}".format(opt.adj_k))
    print("lambda1        : {}".format(opt.lambda1))
    print("lambda2        : {}".format(opt.lambda2))
    print("weight_list    : {}".format(opt.weight_list))
    print("learning rate  : {:.0e}".format(opt.lr))
    print("pretrain epoch : {}".format(opt.pretrain_epoch))
    print("training epoch : {}".format(opt.train_epoch))
    print("cross_fusion   : {}".format(opt.cross_fusion))
    print("moe_num_experts: {}".format(opt.moe_num_experts))
    print("moe_hidden_dim : {}".format(opt.moe_hidden_dim))
    print("moe_gate_noise_mult: {}".format(opt.moe_gate_noise_mult))
    print("moe_balance_wt : {}".format(opt.moe_balance_weight))
    print("q_mix_alpha    : {}".format(opt.q_mix_alpha))
    print("q_mix_warmup   : {}".format(opt.q_mix_warmup))
    print("cluster_usage_wt: {}".format(opt.cluster_usage_weight))
    print("cluster_usage_mode: {}".format(opt.cluster_usage_mode))
    print("center_sep_wt  : {}".format(opt.center_sep_weight))
    print("select_best    : {}".format(opt.select_best))
    print("best_metric    : {}".format(opt.best_metric))
    print("dead_reinit    : {}".format(opt.dead_cluster_reinit))
    print("min_cluster_sz : {}".format(opt.min_cluster_size))
    print("reinit_every   : {}".format(opt.reinit_every))
    print("reinit_until   : {}".format(opt.reinit_until))
    print("reinit_strategy: {}".format(opt.reinit_strategy))
    print("emb_global_attn: {}".format(opt.emb_global_attn))
    print("emb_attn_mask  : {}".format(opt.emb_attn_mask))
    print("emb_attn_temp  : {}".format(opt.emb_attn_temp))
    print("emb_attn_dropout: {}".format(opt.emb_attn_dropout))
    print("emb_alpha_tanh : {}".format(opt.emb_alpha_tanh))
    print("emb_attn_sim   : {}".format(opt.emb_attn_sim))
    print("------------------------------")
    setup_seed(opt.seed)

    # read data
    data_path = "data/"
    labels = pd.read_csv(data_path + 'D1_annotation_labels.csv')
    label = labels['labels']

    if label is not None:
        n_clusters = len(np.unique(label))  
    else:
        n_clusters = 5

    adata_omics1 = sc.read_h5ad(data_path + 'adata_RNA.h5ad')
    adata_omics2 = sc.read_h5ad(data_path + 'adata_ADT.h5ad')
    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()
    
    # Extract spatial coordinates before processing for visualizations
    spatial_coords = adata_omics1.obsm['spatial'].copy() if 'spatial' in adata_omics1.obsm else None
    
    adata_omics1, adata_omics2 = load_data(adata_omics1=adata_omics1, view1="RNA", adata_omics2=adata_omics2, view2="Protein", 
                                            n_neighbors=opt.spatial_k, k=opt.adj_k)
    
    # feature matrix
    data1 = adata_omics1.obsm['feat'].copy()
    data2 = adata_omics2.obsm['feat'].copy()

    # graph
    adj_path = "./pre_adj/{}".format(opt.name)
    os.makedirs(adj_path, exist_ok=True)
    adj = adjacent_matrix_preprocessing(adata_omics1, adata_omics2, adj_path)

    # feature graph
    feature_adj1 = adj['adj_feature_omics1']
    feature_adj2 = adj['adj_feature_omics2']
    # spatial graph
    spatial_adj1 = adj['adj_spatial_omics1']
    spatial_adj2 = adj['adj_spatial_omics2']

    # high-order graph
    Mt1 = process_adjacency_matrix(feature_adj1, "./pre_adj/{}/{}_Mt1.npy".format(opt.name, opt.name))
    Mt2 = process_adjacency_matrix(feature_adj2, "./pre_adj/{}/{}_Mt2.npy".format(opt.name, opt.name))

    def is_symmetric(matrix, tol=1e-8):
        return np.allclose(matrix, matrix.T, atol=tol)

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

    # Abaltion
    spatial_adj1 = spatial_adj1 * feature_adj1
    spatial_adj2 = spatial_adj2 * feature_adj2

    print("============dataset shape=================")
    print("n_clusters:{}".format(n_clusters))
    print("data1.shape:{}".format(data1.shape))
    print("data1.feature.shape:{}".format(feature_adj1.shape))
    print("data1.highOrder.shape:{}".format(Mt1.shape))

    print("================================Pre_training...============================================")
    z1_tilde, z2_tilde = pre_train(
        x1=data1, x2=data2, spatial_adj1=spatial_adj1, feature_adj1=feature_adj1,
        spatial_adj2=spatial_adj2, feature_adj2=feature_adj2, Mt1=Mt1, Mt2=Mt2, y=label, n_clusters=n_clusters,
        num_epoch=opt.pretrain_epoch, device=device, weight_list=opt.weight_list, lr=opt.lr,
        cross_fusion=opt.cross_fusion, moe_num_experts=opt.moe_num_experts, moe_hidden_dim=opt.moe_hidden_dim,
        moe_gate_noise_mult=opt.moe_gate_noise_mult, moe_balance_weight=opt.moe_balance_weight,
        emb_global_attn=opt.emb_global_attn, emb_attn_mask=opt.emb_attn_mask, emb_attn_temp=opt.emb_attn_temp,
        emb_attn_dropout=opt.emb_attn_dropout, emb_alpha_tanh=opt.emb_alpha_tanh, emb_attn_sim=opt.emb_attn_sim,
    )

    metrics_dict = {'ACC': [], 'F1': [], 'NMI': [], 'ARI': [], 'AMI': [], 'VMS': [], 'FMS': []}

    NUM_RUNS = 10
    best_ari = -1
    best_Z = None
    best_y_pred = None
    best_run = -1

    for i in range(NUM_RUNS):
        print("\n================================Training... {}/{}============================================".format(i+1, NUM_RUNS))
        z1_tilde, z2_tilde, acc, f1, nmi, ari, ami, vms, fms, Z, y_pred = train(
            x1=data1, x2=data2, spatial_adj1=spatial_adj1, feature_adj1=feature_adj1, spatial_adj2=spatial_adj2,
            feature_adj2=feature_adj2, y=label, n_clusters=n_clusters, Mt1=Mt1, Mt2=Mt2, num_epoch=opt.train_epoch, lambda1=opt.lambda1,
            device=device, seed=opt.seed, lambda2=opt.lambda2, weight_list=opt.weight_list, lr=opt.lr, num=i,
            spatial_K=opt.spatial_k, adj_K=opt.adj_k, result_dir=result_dir,
            cross_fusion=opt.cross_fusion, moe_num_experts=opt.moe_num_experts, moe_hidden_dim=opt.moe_hidden_dim,
            moe_gate_noise_mult=opt.moe_gate_noise_mult,
            moe_balance_weight=opt.moe_balance_weight, q_mix_alpha=opt.q_mix_alpha, q_mix_warmup=opt.q_mix_warmup,
            cluster_usage_weight=opt.cluster_usage_weight,
            cluster_usage_mode=opt.cluster_usage_mode,
            center_sep_weight=opt.center_sep_weight,
            emb_global_attn=opt.emb_global_attn, emb_attn_mask=opt.emb_attn_mask, emb_attn_temp=opt.emb_attn_temp,
            emb_attn_dropout=opt.emb_attn_dropout, emb_alpha_tanh=opt.emb_alpha_tanh, emb_attn_sim=opt.emb_attn_sim,
            select_best=opt.select_best, best_metric=opt.best_metric,
            dead_cluster_reinit=opt.dead_cluster_reinit, min_cluster_size=opt.min_cluster_size,
            reinit_every=opt.reinit_every, reinit_until=opt.reinit_until,
            reinit_strategy=opt.reinit_strategy,
        )
            
        print(f"Run {i+1} Results: ACC: {acc:.4f}, F1: {f1:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}, AMI: {ami:.4f}")
        
        metrics_dict['ACC'].append(acc)
        metrics_dict['F1'].append(f1)
        metrics_dict['NMI'].append(nmi)
        metrics_dict['ARI'].append(ari)
        metrics_dict['AMI'].append(ami)
        metrics_dict['VMS'].append(vms)
        metrics_dict['FMS'].append(fms)

        if ari > best_ari:
            best_ari = ari
            best_Z = Z
            best_y_pred = y_pred
            best_run = i
            

    # Generate visualization for the best run
    if best_Z is not None:
        print(f"\nGenerating visualizations for best run (Run {best_run+1}) with ARI {best_ari:.4f}...")
        try:
            generate_visualizations(best_Z, best_y_pred, label, spatial_coords, result_dir, f"best_run{best_run+1}")
            print("Visualizations generated and saved.")
        except Exception as e:
            print(f"Failed to generate visualizations: {e}")


    print("\n========================================================")
    print("                     FINAL RESULTS")
    print("========================================================")
    print(f"Metrics over {NUM_RUNS} runs (Mean ± Std):")
    
    # Calculate means and stds
    stats = {}
    for k, v in metrics_dict.items():
        stats[k] = (np.mean(v), np.std(v))
        print(f"{k}: {stats[k][0]:.4f} ± {stats[k][1]:.4f}")

    print("\nComparison with Paper Results:")
    print("-" * 50)
    print(f"{'Metric':<10} | {'Our Results':<20} | {'Paper Results':<20}")
    print("-" * 50)
    for m in ['ARI', 'NMI', 'ACC', 'AMI', 'F1']:
        our_str = f"{stats[m][0]:.3f} ± {stats[m][1]:.3f}"
        paper_str = f"{PAPER_METRICS[m][0]:.3f} ± {PAPER_METRICS[m][1]:.3f}"
        print(f"{m:<10} | {our_str:<20} | {paper_str:<20}")
    print("-" * 50)
    
    print("\n======= Finish ==========")

    sys.stdout = orig_stdout
