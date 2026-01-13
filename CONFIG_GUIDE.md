# 如何通过修改 config.py 改变 SpaFusion 运行行为

## ✅ 确认：真正使用了模块化结构

运行以下命令验证：
```bash
python -c "from encoder import GCNAutoencoder; print('使用的模型来自:', GCNAutoencoder.__module__)"
# 输出: 使用的模型来自: models.fusion
```

这证明 main.py 确实在使用 `models/` 目录下的代码！

---

## 方式1: 修改 config.py 默认值（推荐）

直接编辑 `config.py` 文件来改变默认行为：

### 示例1: 修改训练参数

```python
# config.py 中找到：
TRAINING = {
    'default': {
        'pretrain_epoch': 10000,  # 改为 5000 缩短预训练
        'train_epoch': 350,       # 改为 200
        'lambda1': 1.0,           # 改为 0.5
        'lambda2': 0.1,           # 改为 0.05
        ...
    }
}
```

修改后直接运行：
```bash
python main.py  # 自动使用新的配置
```

### 示例2: 修改模型架构

```python
# config.py 中：
MODEL_CONFIGS = {
    'default': {
        'latent_dim': 20,     # 改为 32，增加潜在空间维度
        'enc_dim1': 256,      # 改为 512，增加编码器容量
        'dropout': 0.1,       # 改为 0.2
        'num_layers': 2,      # 改为 3，增加 Transformer 层数
    }
}
```

修改后直接运行：
```bash
python main.py  # 使用新的模型架构
```

### 示例3: 添加新的消融实验配置

```python
# config.py 中添加：
TRAINING = {
    'default': {...},
    'ablation_no_kl': {...},
    
    # 新增：测试不同的 lambda2 值
    'my_ablation': {
        'pretrain_epoch': 10000,
        'train_epoch': 350,
        'lambda1': 1.0,
        'lambda2': 0.5,  # 测试更大的一致性权重
        ...
    }
}
```

运行新配置：
```bash
python main.py --config_name my_ablation
```

---

## 方式2: 命令行覆盖（临时测试）

不修改 config.py，直接在命令行覆盖：

```bash
# 测试不同的 lambda1 值
python main.py --lambda1 0.5

# 测试不同的学习率
python main.py --lr 0.001

# 多个参数同时覆盖
python main.py --lambda1 0.5 --lambda2 0.2 --train_epoch 200

# 使用不同的配置文件 + 覆盖部分参数
python main.py --config_name ablation_no_kl --lr 0.001
```

---

## 方式3: 混合使用（最灵活）

1. 在 config.py 设置常用配置
2. 命令行临时覆盖测试参数

**示例场景：**
```python
# config.py 设置项目默认值
TRAINING = {
    'default': {
        'pretrain_epoch': 10000,
        'train_epoch': 350,
        # ... 其他常用值
    }
}
```

```bash
# 大部分用默认值，只改 lambda1 测试
python main.py --lambda1 0.3
python main.py --lambda1 0.5
python main.py --lambda1 0.8
```

---

## 优先级说明

配置优先级（从高到低）：
1. **命令行参数**（`--lambda1 0.5`）
2. **config.py 的值**
3. 代码硬编码的后备值

示例：
```bash
# config.py 中 lambda1 = 1.0
python main.py               # 使用 1.0
python main.py --lambda1 2.0 # 使用 2.0（命令行覆盖）
```

---

## 验证配置是否生效

运行时会显示实际使用的配置：

```bash
$ python main.py --lambda1 0.8

============================================================
Loading configuration from config.py...
✓ Loaded config profile: 'default' for dataset 'D1'
============================================================
CONFIGURATION:
============================================================
Command-line overrides applied: lambda1
------------------------------------------------------------
dataset        : D1
lambda1        : 0.8        # ← 确认覆盖成功
lambda2        : 0.1        # ← 使用 config.py 的值
...
```

---

## 常见修改场景

### 1. 快速测试（减少训练时间）
```python
# config.py
TRAINING = {
    'quick_test': {
        'pretrain_epoch': 1000,  # 原来 10000
        'train_epoch': 50,       # 原来 350
        ...
    }
}
```
```bash
python main.py --config_name quick_test
```

### 2. 增大模型容量
```python
# config.py
MODEL_CONFIGS = {
    'large': {
        'enc_dim1': 512,
        'enc_dim2': 256,
        'latent_dim': 32,
        ...
    }
}
```

### 3. 消融实验：禁用 Transformer
由于 Transformer 在代码中硬编码，需要修改 `models/fusion.py`：
```python
# models/fusion.py 的 forward 方法中
def forward(self, ...):
    # 注释掉 Transformer 部分
    # z13 = self.trans_encoder1(...)
    z13 = torch.zeros_like(z11)  # 用零替代
```

---

## 总结

| 修改方式 | 适用场景 | 修改位置 |
|---------|---------|---------|
| 修改 config.py | 默认配置、常用设置 | `config.py` |
| 命令行参数 | 临时测试、参数扫描 | `python main.py --xxx` |
| 修改 models/ | 模型架构、消融实验 | `models/fusion.py` 等 |

**推荐工作流：**
1. 在 `config.py` 设置项目默认值
2. 用命令行快速测试不同参数
3. 需要修改模型结构时，编辑 `models/` 下的文件
