# -*- coding:utf-8 -*-
"""
Author: kyochilian
Date: 2026.02
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

from processing import *
from utils import *
from encoder import *
from high_order_matrix import process_adjacency_matrix
from evaluate import *
import torch.optim as optim
import time
import argparse
from copy import deepcopy

import os

import warnings
warnings.filterwarnings("ignore")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not installed. Install with: pip install wandb")


class TrainingLogger:
    """
    WandB 日志记录器 - 单次运行版本
    
    所有日志记录在同一个 WandB run 中，使用前缀区分不同阶段：
    - pretrain/ : 预训练阶段
    - train/run0/, train/run1/, ... : 各次训练
    - summary/ : 最终汇总
    """
    
    def __init__(self, config, use_wandb=True, project_name="SpaFusion", run_name=None):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.config = config
        self.global_step = 0
        self.current_prefix = ""
        
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=run_name or f"{config['name']}_seed{config['seed']}",
                config=config,
                reinit=True
            )
            print(f"✓ WandB initialized: {run_name or 'default'}")
    
    def set_phase(self, phase_name):
        """设置当前阶段前缀，如 'pretrain' 或 'train/run0'"""
        self.current_prefix = phase_name + "/" if phase_name else ""
        self.global_step = 0  # 每个阶段重置 step
    
    def log(self, data_dict, step=None):
        """通用日志记录"""
        if self.use_wandb:
            prefixed_data = {self.current_prefix + k: v for k, v in data_dict.items()}
            if step is not None:
                wandb.log(prefixed_data, step=step)
            else:
                wandb.log(prefixed_data)
    
    def log_pretrain(self, epoch, loss_dict):
        """记录预训练损失（每5个epoch记录一次）"""
        if self.use_wandb and epoch % 5 == 0:
            prefixed_data = {"pretrain/" + k: v for k, v in loss_dict.items()}
            prefixed_data["pretrain/epoch"] = epoch
            wandb.log(prefixed_data)
    
    def log_train(self, run_idx, epoch, loss_dict, metrics_dict=None):
        if self.use_wandb:
            prefix = f"train/run{run_idx}/"
            log_data = {prefix + k: v for k, v in loss_dict.items()}
            log_data[prefix + "epoch"] = epoch
            if metrics_dict:
                log_data.update({prefix + k: v for k, v in metrics_dict.items()})
            wandb.log(log_data)
    
    def log_run_summary(self, run_idx, metrics_dict):
        if self.use_wandb:
            prefix = f"runs/run{run_idx}/"
            for k, v in metrics_dict.items():
                wandb.log({prefix + k: v})
    
    def log_final_summary(self, metrics_df):
        if self.use_wandb:
            # 记录均值和标准差
            for col in metrics_df.columns:
                wandb.run.summary[f"final/{col}_mean"] = metrics_df[col].mean()
                wandb.run.summary[f"final/{col}_std"] = metrics_df[col].std()
            
            # 创建表格
            table = wandb.Table(dataframe=metrics_df)
            wandb.log({"final/metrics_table": table})
            
            # 记录最佳结果
            wandb.run.summary["best_ARI"] = metrics_df['ARI'].max()
            wandb.run.summary["best_NMI"] = metrics_df['NMI'].max()
            wandb.run.summary["best_ACC"] = metrics_df['ACC'].max()
    
    def finish(self):
        if self.use_wandb:
            wandb.finish()
            print("✓ WandB run finished")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pre_train(x1, x2, spatial_adj1, feature_adj1, spatial_adj2, feature_adj2, Mt1, Mt2, y, n_clusters, num_epoch, device, weight_list, lr, logger=None):
    model = GCNAutoencoder(input_dim1=x1.shape[1], input_dim2=x2.shape[1], enc_dim1=256, enc_dim2=128, dec_dim1=128,
                           dec_dim2=256, latent_dim=20, dropout=0.1, num_layers=2, num_heads1=1, num_heads2=1,
                           n_clusters=n_clusters, n_node=x1.shape[0])

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pretrain_loss = []
    for epoch in range(num_epoch):
        Z, z1_tilde, z2_tilde, a11_hat, a12_hat, a21_hat, a22_hat, x13_hat, x23_hat, _ = \
            model(x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2, pretrain=True)

        loss_ae1 = F.mse_loss(a11_hat, spatial_adj1)
        loss_ae2 = F.mse_loss(a12_hat, feature_adj1)
        loss_ae3 = F.mse_loss(a21_hat, spatial_adj2)
        loss_ae4 = F.mse_loss(a22_hat, feature_adj2)

        loss_x1 = F.mse_loss(x13_hat, x1)
        loss_x2 = F.mse_loss(x23_hat, x2)

        loss_rec = weight_list[0] * loss_ae1 + weight_list[1] * loss_ae2 + weight_list[2] * loss_ae3 + weight_list[3] * loss_ae4 + weight_list[4] * loss_x1 + weight_list[5] * loss_x2

        loss = loss_rec
 
        pretrain_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_dict = {
            "total_loss": loss.item(),
            "loss_spatial_adj1": loss_ae1.item(),
            "loss_feature_adj1": loss_ae2.item(),
            "loss_spatial_adj2": loss_ae3.item(),
            "loss_feature_adj2": loss_ae4.item(),
            "loss_recon_x1": loss_x1.item(),
            "loss_recon_x2": loss_x2.item(),
        }
        logger.log_pretrain(epoch, loss_dict)

        if epoch % 500 == 0:
            print("Epoch: {:.0f}/{:.0f} ,loss:{:.8f}".format(epoch + 1, num_epoch, loss))

    torch.save(model.state_dict(), r'./pretrain/{}_pre_model.pkl'.format(opt.name))
    return z1_tilde, z2_tilde


def train(x1, x2, spatial_adj1, feature_adj1, spatial_adj2, feature_adj2, Mt1, Mt2, y, n_clusters, num_epoch, lambda1, device, seed, lambda2, weight_list, lr, run_idx, spatial_K, adj_K, result_dir, logger=None):
    model = GCNAutoencoder(input_dim1=x1.shape[1], input_dim2=x2.shape[1], enc_dim1=256, enc_dim2=128, dec_dim1=128,
                           dec_dim2=256, latent_dim=20, dropout=0.1, num_layers=2, num_heads1=1, num_heads2=1,
                           n_clusters=n_clusters, n_node=x1.shape[0])
    
    model.to(device)

    model.load_state_dict(torch.load(r'./pretrain/{}_pre_model.pkl'.format(opt.name), map_location='cpu'))

    with torch.no_grad():
        Z, z1_tilde, z2_tilde, _, _, _, _, _, _, _ = \
            model(x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2)

    centers1 = clustering(Z, y, n_clusters=n_clusters)  

    model.cluster_centers1.data = torch.tensor(centers1).to(device)

    train_losses = []
    ari_ = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_ari = 0
    best_metrics = None

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
        loss_rec = weight_list[0] * loss_ae1 + weight_list[1] * loss_ae2 + weight_list[2] * loss_ae3 + weight_list[3] * loss_ae4 + weight_list[4] * loss_x1 + weight_list[5] * loss_x2
        L_KL1 = distribution_loss(Q, target_distribution(Q[0].data))
        loss = loss_rec + lambda1 * L_KL1 + lambda2 * (dense_loss1 + dense_loss2)

        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if y is not None:
            acc, f1, nmi, ari, ami, vms, fms, y_pred = assignment((Q[0]).data, y)
            
            loss_dict = {
                "total_loss": loss.item(),
                "loss_rec": loss_rec.item(),
                "loss_kl": L_KL1.item(),
                "loss_consistency": (dense_loss1 + dense_loss2).item(),
            }
            metrics_dict = {
                "ACC": acc,
                "F1": f1,
                "NMI": nmi,
                "ARI": ari,
                "AMI": ami,
            }
            logger.log_train(run_idx, epoch, loss_dict, metrics_dict)
                
            if ari > best_ari:
                best_ari = ari
                best_metrics = (acc, f1, nmi, ari, ami, vms, fms)
        else:
            y_pred = torch.argmax(Q[0].data, dim=1).data.cpu().numpy()

        if epoch % 50 == 0:
            print("Epoch: {:.0f}/{:.0f} ,loss:{:.8f}".format(epoch + 1, num_epoch, loss))

    if logger and best_metrics:
        logger.log_run_summary(run_idx, {
            "best_ACC": best_metrics[0],
            "best_F1": best_metrics[1],
            "best_NMI": best_metrics[2],
            "best_ARI": best_metrics[3],
            "best_AMI": best_metrics[4],
        })


    if y is not None and best_metrics is not None:
        with open(os.path.join(result_dir, '{}_performance.csv'.format(opt.name)), 'a') as f:
            f.write("seed:{}, lambda1:{}, lambda2:{}, spatial_k:{}, adj_k:{}, wieght_list:{}, ".format(seed, lambda1, lambda2, spatial_K, adj_K, weight_list))
            f.write('%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n' % best_metrics)
    else:
        pass

    np.save(os.path.join(result_dir, '{}_{}_pre_label.npy'.format(opt.name, run_idx)), y_pred)
    np.save(os.path.join(result_dir, '{}_{}_laten.npy'.format(opt.name, run_idx)), Z.data.cpu().numpy())

    return z1_tilde, z2_tilde, best_metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Model settings")
    parser.add_argument('--name', type=str, default='D1', help='dataset name')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--spatial_k', type=int, default=9, help='spatial_k')
    parser.add_argument('--adj_k', type=int, default=20, help='adj_k')
    parser.add_argument('--lambda1', type=float, default=1, help='lambda1')
    parser.add_argument('--lambda2', type=float, default=0.1, help='lambda2')
    parser.add_argument('--weight_list', type=list, default=[1, 1, 1, 1, 1, 1], help='weight list')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--pretrain_epoch', type=int, default=10000, help='pretrain epoch')
    parser.add_argument('--train_epoch', type=int, default=350, help='train epoch')
    parser.add_argument('--use_wandb', action='store_true', default=True, help='Enable WandB logging')
    parser.add_argument('--wandb_project', type=str, default='SpaFusion', help='WandB project name')
    parser.add_argument('--skip_pretrain', action='store_true', default=False, help='Skip pretraining and use existing pretrained model')

    opt = parser.parse_args()

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
    print("use_wandb      : {}".format(opt.use_wandb))
    print("skip_pretrain  : {}".format(opt.skip_pretrain))
    print("------------------------------")
    setup_seed(opt.seed)

    # ========= 配置信息 =========
    config = vars(opt)
    
    # ========= 生成时间戳结果文件夹 =========
    training_timestamp = time.strftime('%Y%m%d_%H%M%S')
    result_dir = f'./results/{opt.name}/{training_timestamp}'
    os.makedirs(result_dir, exist_ok=True)
    print(f"Results will be saved to: {result_dir}")
    
    # ========= 创建单一的 WandB logger =========
    # 所有日志记录在同一个 run 中，使用前缀区分阶段
    logger = TrainingLogger(
        config=config,
        use_wandb=opt.use_wandb,
        project_name=opt.wandb_project,
        run_name=f"{opt.name}_seed{opt.seed}_{training_timestamp}"
    )

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

    # ========= 预训练逻辑 =========
    pretrain_model_path = r'./pretrain/{}_pre_model.pkl'.format(opt.name)
    
    if opt.skip_pretrain:
        # 检查预训练模型是否存在
        if os.path.exists(pretrain_model_path):
            print("================================Skipping Pre_training...=========================================")
            print(f"Using existing pretrained model: {pretrain_model_path}")
        else:
            raise FileNotFoundError(f"Pretrained model not found: {pretrain_model_path}. Please run pretraining first or set --skip_pretrain to False.")
    else:
        print("================================Pre_training...============================================")
        z1_tilde, z2_tilde = pre_train(x1=data1, x2=data2, spatial_adj1=spatial_adj1, feature_adj1=feature_adj1,
                                    spatial_adj2=spatial_adj2, feature_adj2=feature_adj2, Mt1=Mt1, Mt2=Mt2, y=label, n_clusters=n_clusters,
                                    num_epoch=opt.pretrain_epoch, device=device, weight_list=opt.weight_list, lr=opt.lr,
                                    logger=logger)

    # ========= 10次训练，使用同一个 logger =========
    all_metrics = []
    for i in range(10):
        print("================================Training... {}============================================".format(i))
        
        z1_tilde, z2_tilde, metrics = train(x1=data1, x2=data2, spatial_adj1=spatial_adj1, feature_adj1=feature_adj1, spatial_adj2=spatial_adj2,
                                feature_adj2=feature_adj2, y=label, n_clusters=n_clusters, Mt1=Mt1, Mt2=Mt2, num_epoch=opt.train_epoch, lambda1=opt.lambda1,
                                device=device, seed=opt.seed, lambda2=opt.lambda2, weight_list=opt.weight_list, lr=opt.lr, run_idx=i, 
                                spatial_K=opt.spatial_k, adj_K=opt.adj_k, result_dir=result_dir,
                                logger=logger)
        
        if metrics:
            all_metrics.append(metrics)

    # ========= 记录最终汇总指标 =========
    if all_metrics:
        df = pd.DataFrame(all_metrics, columns=['ACC', 'F1', 'NMI', 'ARI', 'AMI', 'VMS', 'FMS'])
        
        # 记录到 WandB
        logger.log_final_summary(df)
        
        print("\n==================== 10-run Metrics Summary ====================")
        print(f"Dataset: {opt.name}")
        print(f"Runs: {len(df)}")
        print("\nMean ± Std:")
        for col in df.columns:
            print(f"  {col}: {df[col].mean():.4f} ± {df[col].std():.4f}")
        
        print("\n==================== Comparison with Paper ====================")
        if opt.name == 'A1':
            paper = {'ARI': 0.319, 'NMI': 0.399, 'ACC': 0.629, 'AMI': 0.396, 'F1': 0.337}
        elif opt.name == 'D1':
            paper = {'ARI': 0.351, 'NMI': 0.384, 'ACC': 0.599, 'AMI': 0.379, 'F1': 0.323}
        else:
            paper = None
        
        if paper:
            for metric, paper_val in paper.items():
                if metric in df.columns:
                    our_val = df[metric].mean()
                    diff = our_val - paper_val
                    print(f"{metric:>3s}: Ours={our_val:.4f} | Paper={paper_val:.4f} | Diff={diff:+.4f}")
        else:
            print("No paper baseline for this dataset.")
    
    # ========= 保存训练日志 =========
    log_path = os.path.join(result_dir, 'training_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Training Log - {training_timestamp}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n" + "=" * 50 + "\n\n")
        if all_metrics:
            f.write("Performance Summary (10 runs):\n")
            for col in df.columns:
                f.write(f"  {col}: {df[col].mean():.4f} ± {df[col].std():.4f}\n")
    print(f"Training log saved to: {log_path}")
    
    # ========= 保存空间坐标供可视化使用 =========
    coords = adata_omics1.obsm['spatial']
    np.save(os.path.join(result_dir, 'spatial_coords.npy'), coords)
    print(f"Spatial coords saved to: {result_dir}/spatial_coords.npy")

    print("\n============================ Finish ==================================")

    # 关闭 WandB
    logger.finish()
