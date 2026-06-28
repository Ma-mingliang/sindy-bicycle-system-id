# D: 算法、网络架构与损失函数审计

## 1. ODEFunc 骨干网络 (neural_ode_v9.py:41-59)

### 1.1 架构

```python
class ODEFunc(nn.Module):
    def __init__(self, state_dim=7, action_dim=1, hidden_dim=128, depth=2, activation='tanh'):
        # 输入: state_dim + action_dim = 8
        # 隐藏层: depth 个 hidden_dim 层
        # 输出: state_dim = 7
        # 末层: Xavier gain=0.1, 零偏置
```

**所有模型变体共享此骨干网络**（V9/V10/V11/V12）。

### 1.2 末层初始化

```python
nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
nn.init.zeros_(self.net[-1].bias)
```

**效果**: 初始输出接近零，模型初始行为接近恒等映射（s_next ≈ s）。这对 Neural ODE 训练稳定性至关重要。

### 1.3 激活函数

默认 `tanh`，输出范围 [-1, 1]。tanh 的饱和特性提供了隐式的输出边界，但梯度消失可能阻碍学习。

## 2. 损失函数组件 (neural_ode_v9.py:100-163)

### 2.1 单步损失 (MSE)

```python
loss_single = F.mse_loss(pred_dsdot, train_dsdot)
```

**权重**: 1.0（主导项）

### 2.2 多步 Rollout 损失

```python
# neural_ode_v9.py:128-138
for step in range(1, rollout_len):
    dsdt = self.ode_func(s_cur, u_cur)
    s_cur = s_cur + dsdt * self._dt
    loss_multi += F.mse_loss(s_cur, targets[step])
loss_multi /= rollout_len
```

**权重**: 0.3

**关键问题**:
1. **shuffle=True** (line 96): batch 元素不保证时间连续，多步 rollout 从错误的初始状态开始
2. **归一化不匹配**: s_cur 在物理空间，targets 在归一化空间（见 B 报告）

### 2.3 一致性损失

```python
# neural_ode_v9.py:143-148
pred_next = states + pred_dsdot * self._dt
loss_consistency = F.mse_loss(pred_next, next_states)
```

**权重**: 0.1

**问题**: 缺少 std_ratio 修正。pred_dsdot 是归一化速率，乘以 dt 后量纲不匹配 next_states（物理空间）。

### 2.4 Jacobian 正则化

```python
# neural_ode_v9.py:150-163
J = torch.autograd.functional.jacobian(self.ode_func, s_mean)
loss_jacobian = torch.norm(J, p='fro')
```

**权重**: 0.01

**目的**: 限制 ODE 的 Lipschitz 常数，防止轨迹爆炸。

**计算成本**: 每次调用需要完整 Jacobian 计算（O(n²) 前向传播），是训练中最昂贵的组件。

## 3. 损失权重分析

| 组件 | 权重 | 有效贡献 | 问题 |
|------|------|----------|------|
| 单步 MSE | 1.0 | ~77% | 正常 |
| 多步 Rollout | 0.3 | ~0% (bug) | shuffle + 归一化 bug 使梯度为噪声 |
| 一致性 | 0.1 | ~8% | 量纲不匹配 |
| Jacobian | 0.01 | ~15% | 计算昂贵但贡献有限 |

**实际有效损失**: 近似为 `1.0 * loss_single + 0.01 * loss_jacobian`

## 4. V12 修复对比

### 4.1 SequentialSegmentDataset (neural_ode_v12.py)

```python
class SequentialSegmentDataset(Dataset):
    def __init__(self, states, actions, segment_len):
        # 返回连续时间片段，每个样本是 (states[i:i+H], actions[i:i+H])
```

**修复**: 每个 batch 元素是自包含的时间段，shuffle=True 不再破坏时间连续性。

### 4.2 RK4 积分 (neural_ode_v12.py)

替代 Euler 积分，单步误差从 O(dt²) 降到 O(dt⁵)。

### 4.3 收缩性正则化 (neural_ode_contractive.py)

```python
# 强制 Jacobian 特征值为负（收缩性）
eigenvalues = torch.linalg.eigvalsh(J)
loss_contractive = F.relu(eigenvalues).mean()
```

**权重**: 0.05，warmup 20 epochs

## 5. 自适应课程调度 (neural_ode_adaptive_curriculum.py)

### 5.1 升级条件

```python
if val_loss < threshold * best_loss:
    curriculum_level += 1
```

### 5.2 问题

- 无回归检测：进入高级别后误差增加不会回退
- 阈值敏感：0.95 太激进，可能导致过早升级
- 无存活率门控：即使存活率为 0 也可能升级

## 6. 总结

| 问题 | 严重度 | 根因 |
|------|--------|------|
| 多步损失因 shuffle 而失效 | CRITICAL | DataLoader shuffle=True |
| 一致性损失量纲不匹配 | HIGH | 缺少 std_ratio 修正 |
| Jacobian 正则化成本高收益低 | MEDIUM | O(n²) 前向传播 |
| 课程调度无回归保护 | MEDIUM | 缺少回退机制 |
| 损失权重未经系统调优 | LOW | 需要消融实验 |
