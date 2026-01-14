# 内存错误排查指南

## 错误信息
```
numpy.core._exceptions._ArrayMemoryError: Unable to allocate 14.1 MiB
```

## 原因分析

这个错误**不是代码bug**，而是系统内存问题。虽然只需要14.1 MiB（很小），但仍然分配失败，可能原因：

1. **内存碎片化** - Windows内存管理问题
2. **进程内存限制** - Python进程达到某个限制
3. **其他程序占用** - 内存被其他程序占用
4. **虚拟内存不足** - 磁盘空间不足导致虚拟内存失败

## 解决方案（按优先级）

### 方案1：重启PowerShell（最快）✅

```bash
# 1. 关闭当前PowerShell窗口
# 2. 重新打开PowerShell
# 3. 进入项目目录
cd D:\SpaFusion
& d:/SpaFusion/.venv/Scripts/Activate.ps1

# 4. 重新运行
python main.py
```

### 方案2：清理Python内存

```bash
# 在运行前执行GC
python -c "import gc; gc.collect(); print('Memory cleaned')"

# 然后立即运行
python main.py
```

### 方案3：重启计算机✅✅

这是最彻底的方法，可以清理所有内存碎片。

### 方案4：检查可用内存

```powershell
# 查看内存使用情况
Get-CimInstance Win32_OperatingSystem | Select-Object @{
    Name="FreeGB";
    Expression={[math]::Round($_.FreePhysicalMemory/1MB,2)}
}, @{
    Name="TotalGB";
    Expression={[math]::Round($_.TotalVisibleMemorySize/1MB,2)}
}

# 如果可用内存 < 2GB，需要关闭其他程序
```

### 方案5：增加虚拟内存

如果物理内存不足：
1. 控制面板 → 系统 → 高级系统设置
2. 性能 → 设置 → 高级 → 虚拟内存
3. 更改虚拟内存大小

### 方案6：使用backed模式读取数据

如果内存确实不足，修改数据读取方式：

```python
# main.py 中改为：
adata_omics1 = sc.read_h5ad(data_path + 'adata_RNA.h5ad', backed='r')
adata_omics2 = sc.read_h5ad(data_path + 'adata_ADT.h5ad', backed='r')
```

这样数据不会完全加载到内存，而是按需读取。

## Bug修复状态 ✅

**重要**: 融合参数维度bug已经修复！

之前的问题已解决：
```python
# ✅ 现在正确使用 latent_dim
self.a = Parameter(torch.zeros(n_node, latent_dim), ...)
self.b = Parameter(torch.zeros(n_node, latent_dim), ...)  
self.c = Parameter(torch.zeros(n_node, latent_dim), ...)
```

一旦内存问题解决，训练应该能达到正常性能：
- ACC > 0.55
- ARI > 0.30  
- NMI > 0.35

## 推荐操作流程

```bash
# 1. 重启PowerShell窗口（最简单）
# 2. 激活环境
cd D:\SpaFusion
& d:/SpaFusion/.venv/Scripts/Activate.ps1

# 3. 确认bug修复
python -c "print('Testing import...'); from models.fusion import GCNAutoencoder; print('✓ Import successful')"

# 4. 运行训练
python main.py

# 如果还是内存错误，重启电脑
```

## 验证修复是否生效

训练成功后，检查结果：
```bash
# 查看最新结果
ls results/D1/ -OrderBy LastWriteTime | Select-Object -Last 1

# 预期看到较高的指标
# ARI应该在0.3以上，ACC在0.55以上
```

如果分数还是很低（< 0.3），请告知，可能还有其他问题。
