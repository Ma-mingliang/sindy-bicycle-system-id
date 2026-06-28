# E: 训练循环与评估管线审计

## 1. 训练循环 (neural_ode_v9.py:74-188)

### 1.1 训练流程

```
for epoch in range(n_epochs):
    for batch in dataloader:  # shuffle=True (BUG)
        states, actions, deltas = batch
        # 1. 前向传播: pred_dsdot = ode_func(states, actions)
        # 2. 单步损失: loss_single = MSE(pred_dsdot, targets)
        # 3. 多步 rollout: (broken by shuffle)
        # 4. 一致性损失: (wrong dimensions)
        # 5. Jacobian 正则化: (expensive)
        # 6. 反向传播 + 优化器步
```

### 1.2 课程调度

```python
# neural_ode_v9.py:108-115
curriculum_levels = [1, 5, 10, 20]  # 默认
rollout_len = curriculum_levels[min(epoch // 50, len(curriculum_levels)-1)]
```

**问题**: 线性进度，无自适应。每 50 epoch 自动升级，无论验证误差如何。

### 1.3 优化器

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)
```

**学习率**: 固定 1e-3，无 warmup。

## 2. 数据管线 (data_loader_v9.py + data_loader.py)

### 2.1 数据流

```
stage2_dataset_150k.npz
  → load_7d_data(): 8D→7D, episode detection, train/test split
  → get_test_segments(): 连续段提取 (segment_length=1100)
  → DataLoader(shuffle=True)  # BUG for multi-step
```

### 2.2 数据集统计

- 总样本: ~150k
- Episodes: 59
- 训练/测试: 75/25 按 episode 划分
- 平均 episode 长度: ~2542 样本

### 2.3 段长度约束

V9 请求 `segment_length=1100`。测试段必须来自单个 episode 内至少 1101 个连续样本。短于 1101 样本的 episode 被排除。

## 3. 评估管线 (evaluation_v9.py)

### 3.1 multi_step_evaluate 函数

```python
def multi_step_evaluate(model, segment, horizons=[1,5,10,50,100,200,500]):
    for H in horizons:
        for t in range(len(segment) - H):
            # 自回归 rollout H 步
            s = segment[t]
            for step in range(H):
                s = model.predict(s, action)
            # 计算 NMAE
            nmae = compute_nmae(predicted, ground_truth)
            # 检查存活
            survived = check_survival(predicted)
```

### 3.2 NMAE 计算 (evaluation_v9.py:15-25)

```python
def compute_nmae(pred, true, state_std):
    return np.mean(np.abs(pred - true) / state_std)
```

**注意**: NMAE 仅在存活轨迹上计算。早期失败的轨迹被排除，导致长 horizon NMAE 偏乐观。

### 3.3 存活率检测 (evaluation_v9.py:35-50)

```python
def check_survival(state):
    limits = PHYSICAL_LIMITS
    if abs(state[0]) > limits['e_y']: return False      # e_y > 5m
    if abs(state[1]) > limits['e_psi']: return False    # e_psi > π
    if abs(state[2]) > limits['v']: return False        # v > 5 m/s
    if abs(state[5]) > limits['delta']: return False    # delta > π/2
    return True
```

**注意**: 仅检查 4 个关键维度，theta/theta_dot/delta_dot 未检查。

### 3.4 静默异常捕获 (evaluation_v9.py:91)

```python
try:
    # 评估代码
except:
    pass  # 静默吞掉所有异常
```

**风险**: 隐藏数值错误、NaN 传播等问题。

## 4. 评估与训练的 Gap

| 维度 | 训练 | 评估 |
|------|------|------|
| Rollout 长度 | 1-20 步 | 1-500 步 |
| 初始状态 | 随机 batch 采样 | 连续段起点 |
| 数据顺序 | shuffle (broken) | 时间顺序 |
| 归一化 | 混合空间 | 物理空间 |
| 存活检查 | 无 | 每步检查 |

**核心 Gap**: 模型从未在训练中见过 20 步以上的 rollout，但被要求在评估中预测 500 步。

## 5. V12 改进

### 5.1 SequentialSegmentDataset

修复数据顺序问题，每个样本是连续时间段。

### 5.2 扩展课程

```python
# V12: '1,5,10,20,50,100' vs V9: '1,5,10,20'
```

### 5.3 新问题

扩展课程暴露了长 horizon 不稳定性，导致常数预测崩溃。

## 6. 总结

| 问题 | 严重度 | 影响 |
|------|--------|------|
| 训练-评估 rollout 长度 gap (20 vs 500) | CRITICAL | 模型未训练长 horizon |
| shuffle 破坏多步损失 | CRITICAL | 多步梯度为噪声 |
| 静默异常捕获 | HIGH | 隐藏评估错误 |
| NMAE 仅在存活轨迹计算 | MEDIUM | 偏乐观的指标 |
| 存活检查不完整 | LOW | 仅检查 4/7 维度 |
