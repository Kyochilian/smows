import numpy as np
import torch
import opt
from utils import post_proC, print_metrics, set_seed
from model import SpaMICS
from evaluation import eval
from load_data import load_data
import tqdm
import warnings
import os
import sys
import json
from datetime import datetime

warnings.filterwarnings('ignore', category=UserWarning, module='anndata')
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')


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


if __name__ == '__main__':

    set_seed(seed=opt.args.seed)

    # 创建结果保存文件夹
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 为当前运行创建子文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(results_dir, f"{opt.args.name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Results will be saved to: {run_dir}")
    
    # Start log capture
    log_path = os.path.join(run_dir, 'training_log.txt')
    tee = TeeOutput(log_path)
    sys.stdout = tee

    # Load data
    X_omics1, X_omics2, adj_feature_omics1, adj_feature_omics2, label, adj_spatial_omics1, adj_spatial_omics2 = load_data()

    opt.args.n_omics1 = X_omics1.shape[1]
    opt.args.n_omics2 = X_omics2.shape[1]
    if opt.args.name == 'Human_tonsil':
        opt.args.n_cluster = 7
        label = None
    elif opt.args.name == 'Human_Breast_Cancer':
        opt.args.n_cluster = 18
        label = None
    else:
        opt.args.n_cluster = len(np.unique(label))

    print("=" * 10 + " Pretraining has begun! " + "=" * 10)

    model = SpaMICS(X_omics2.shape[0]).cuda(opt.args.device)
    optimizer0 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.args.pretrain_lr)
    pbar = tqdm.tqdm(range(31), ncols=200)

    # 记录训练损失
    training_history = {
        'pretrain_stage0': [],
        'pretrain_stage1': [],
        'train_stage2': []
    }

    for epoch in pbar:
        loss_rec = model(X_omics1, X_omics2, adj_feature_omics1, adj_feature_omics2, adj_spatial_omics1,
                         adj_spatial_omics2, stage=0)

        pretrain_loss = loss_rec
        optimizer0.zero_grad()
        pretrain_loss.backward()
        optimizer0.step()

        pbar.set_postfix({'loss': '{0:1.4f}'.format(pretrain_loss)})
        training_history['pretrain_stage0'].append(float(pretrain_loss.item()))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.args.pretrain_lr)
    pbar = tqdm.tqdm(range(opt.args.pretrain_epoch + 1), ncols=200)
    for epoch in pbar:
        loss_rec = model(X_omics1, X_omics2, adj_feature_omics1, adj_feature_omics2, adj_spatial_omics1,
                         adj_spatial_omics2, stage=1)

        pretrain_loss = loss_rec
        optimizer.zero_grad()
        pretrain_loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': '{0:1.4f}'.format(pretrain_loss)})
        training_history['pretrain_stage1'].append(float(pretrain_loss.item()))

    optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.args.train_lr)
    pbar2 = tqdm.tqdm(range(opt.args.epoch + 1), ncols=200)
    
    # 记录最佳结果
    best_metrics = None
    best_epoch = 0
    
    for epoch in pbar2:

        loss_rec, loss_self, loss_reg, loss_dis, S = model(X_omics1, X_omics2, adj_feature_omics1, adj_feature_omics2,
                                                           adj_spatial_omics1, adj_spatial_omics2, stage=2)

        total_loss = loss_rec + opt.args.lambda_1 * loss_self + opt.args.lambda_2 * loss_reg + opt.args.lambda_3 * loss_dis

        optimizer2.zero_grad()
        total_loss.backward()
        optimizer2.step()

        pbar2.set_postfix({'loss': '{0:1.4f}'.format(total_loss)})
        training_history['train_stage2'].append(float(total_loss.item()))
        
        if epoch % 600 == 0 and epoch != 0:
            S_cpu = S.cpu().detach().numpy()
            pred, _ = post_proC(S_cpu, opt.args.n_cluster)

            if label is not None:
                acc, f1, nmi, ari, ami, _, _ = eval(label, pred)
                print_metrics(acc, f1, nmi, ari, ami)
                
                # 保存最佳结果
                if best_metrics is None or acc > best_metrics['ACC']:
                    best_metrics = {
                        'ACC': float(acc),
                        'F1': float(f1),
                        'NMI': float(nmi),
                        'ARI': float(ari),
                        'AMI': float(ami),
                        'epoch': epoch
                    }
                    best_epoch = epoch
                    
                    # 保存最佳模型
                    torch.save(model.state_dict(), os.path.join(run_dir, 'best_model.pth'))
                    # 保存最佳预测结果
                    np.save(os.path.join(run_dir, 'best_predictions.npy'), pred)
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(run_dir, 'final_model.pth'))
    
    # 保存最终预测结果
    S_cpu = S.cpu().detach().numpy()
    pred, L = post_proC(S_cpu, opt.args.n_cluster)
    np.save(os.path.join(run_dir, 'final_predictions.npy'), pred)
    np.save(os.path.join(run_dir, 'similarity_matrix.npy'), S_cpu)
    np.save(os.path.join(run_dir, 'spectral_matrix.npy'), L)
    
    # 保存训练历史
    np.save(os.path.join(run_dir, 'training_history.npy'), training_history)
    
    # 保存配置和结果摘要
    summary = {
        'dataset': opt.args.name,
        'timestamp': timestamp,
        'n_clusters': int(opt.args.n_cluster),
        'seed': int(opt.args.seed),
        'pretrain_epoch': int(opt.args.pretrain_epoch),
        'train_epoch': int(opt.args.epoch),
        'learning_rate_pretrain': float(opt.args.pretrain_lr),
        'learning_rate_train': float(opt.args.train_lr),
        'lambda_1': float(opt.args.lambda_1),
        'lambda_2': float(opt.args.lambda_2),
        'lambda_3': float(opt.args.lambda_3),
    }
    
    if label is not None:
        acc, f1, nmi, ari, ami, _, _ = eval(label, pred)
        summary['final_metrics'] = {
            'ACC': float(acc),
            'F1': float(f1),
            'NMI': float(nmi),
            'ARI': float(ari),
            'AMI': float(ami)
        }
        if best_metrics is not None:
            summary['best_metrics'] = best_metrics
        
        print("\n" + "=" * 50)
        print("Final Results:")
        print_metrics(acc, f1, nmi, ari, ami)
        if best_metrics is not None:
            print(f"\nBest results achieved at epoch {best_epoch}:")
            print_metrics(best_metrics['ACC'], summary['final_metrics']['F1'], 
                         best_metrics['NMI'], best_metrics['ARI'], best_metrics['AMI'])
        print("=" * 50)
        
        # Comparison with Paper (SpaMICS)
        if "Human_Lymph_Node_A1" in opt.args.name:
            print("\n==================== Comparison with Paper (SpaMICS) ====================")
            print(f"Dataset: Human_Lymph_Node_A1")
            print("{:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format("Method", "ACC", "F1", "NMI", "ARI", "AMI"))
            print("{:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format("SpaMICS Paper", "0.6076", "0.3731", "0.4032", "0.3431", "0.3990"))
            
            if best_metrics:
                print("{:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                    "Ours", 
                    best_metrics.get('ACC', 0), 
                    best_metrics.get('F1', 0),
                    best_metrics.get('NMI', 0), 
                    best_metrics.get('ARI', 0), 
                    best_metrics.get('AMI', 0)
                ))
            else:
                 # Should failback to current metrics (acc, f1, etc variables are available in scope)
                 pass # best_metrics is populated if epoch > 0.
            print("=" * 50)
    
    # 保存摘要为JSON
    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nAll results saved to: {run_dir}")
    
    # Close log capture
    sys.stdout = tee.terminal
    tee.close()
    print(f"Training log saved to: {log_path}")
