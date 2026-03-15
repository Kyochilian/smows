# SpaFusion: A multi-level fusion model for clustering spatial multi-omics data
The source code and input data of SpaFusion.
![framework](Spafusion.jpg)



## Requirement
SpaFusion is implemented in the pytorch framework, please run SpaFusion on CUDA. 
```
conda env create -f environment.yml
```

## Usage
## Clone this repo.
```
git clone https://github.com/polarisChen/SpaFusion.git
```

### Example command
Take the dataset "Human_lymph_node_D1" as an example
```
python main.py
```


 Plan to implement                                                                                                    │
│                                                                                                                      │
│ Context                                                                                                              │
│                                                                                                                      │
│ 你提供的 moe.py 是一个 **Dense MoE（密集混合专家）**融合模块：对每个样本/节点都会计算所有专家输出，再用 softmax      │
│ 门控权重加权求和（无 top-k 路由、无 capacity、无 load-balance loss）。                                               │
│                                                                                                                      │
│ 目标：把该 MoE 用到 SpaFusion 的 information fusion 中，客观判断是否能带来       │
│ ARI/NMI/ACC 等指标提升。                                                                                             │
│                                                                                                                      │
│ ---                                                                                                                  │

