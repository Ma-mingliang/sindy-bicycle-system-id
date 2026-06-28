# Rollout 分布偏移分析

**更新时间**: 2026-06-28 18:10
**RUN_ID**: 20260628_175824_neural_ode_72h

---

## 1. 问题描述

### 核心问题
v9 Neural ODE 在训练时使用真实状态（teacher forcing），但在推理时递归使用预测状态（free rollout）。这导致训练和推理的输入分布不一致。

### 量化影响
- 训练时输入分布: 真实状态分布 P_real
- 推理时输入分布: 预测状态分布 P_pred
- 随着 rollout 步数增加，P_pred 偏离 P_real 越来越远

---

## 2. 分布偏移机制

### 2.1 单步偏移

```
训练: s_true → f_θ(s_true) → pred
推理: s_pred → f_θ(s_pred) → pred'

如果 s_pred ≠ s_true，则 pred' ≠ pred
```

### 2.2 多步累积

```
H=1: s_0 → f_θ(s_0) → s_1_pred
H=2: s_1_pred → f_θ(s_1_pred) → s_2_pred
...
H=n: s_{n-1}_pred → f_θ(s_{n-1}_pred) → s_n_pred

每一步都累积前一步的误差
```

### 2.3 误差放大

如果 f_θ 的 Jacobian 有特征值 > 1，误差会被放大：
```
δs_{n+1} ≈ J_θ * δs_n
其中 J_θ = ∂f_θ/∂s

如果 ||J_θ|| > 1，误差指数增长
```

---

## 3. V9 的缓解措施

### 3.1 多步课程学习

```python
rollout_curriculum = '1,5,10,20'

# 训练过程中逐步增加 rollout 步数
epoch 0-49:   rollout_steps = 1
epoch 50-99:  rollout_steps = 5
epoch 100-149: rollout_steps = 10
epoch 150-199: rollout_steps = 20
```

**效果**:
- 部分缓解分布偏移
- 但最大只到 20 步
- H=100+ 时仍然存在显著偏移

### 3.2 Jacobian 正则化

```python
loss_jacobian = ||J_θ||²
```

**效果**:
- 平滑动力学函数
- 减小 Jacobian 范数
- 但可能过约束模型

---

## 4. 已知问题

### 4.1 课程学习的局限

- 最大 rollout 只到 20 步
- H=50+ 时未被训练覆盖
- 长 rollout 会导致梯度爆炸

### 4.2 一致性损失的局限

```python
# 一致性损失：theta 应与 theta_dot 一致
theta_pred = s_next_norm[:, 3]
theta_gt = sb[:, 3] + theta_dot * dt
loss_consistency = mse(theta_pred, theta_gt)
```

- 只约束了 theta 和 theta_dot 的关系
- 未约束其他状态
- 可能不足以防止长期漂移

### 4.3 无自由 rollout 训练损失

- 训练损失全部基于单步或多步
- 没有直接优化长期 rollout 目标
- 可能导致长期性能不佳

---

## 5. 解决方案

### 5.1 增加 Rollout 长度

- 将课程学习扩展到 50 或 100 步
- 需要处理梯度爆炸问题
- 可能需要梯度裁剪或停止梯度

### 5.2 Scheduled Sampling

- 训练时随机选择使用真实状态或预测状态
- 逐步增加使用预测状态的概率
- 缓解训练-推理分布不一致

### 5.3 Pushforward Training (DAgger 类)

- 使用 rollout 数据重新训练
- 类似 DAgger 的在线学习
- 减少分布偏移

### 5.4 Trajectory Loss

- 直接优化整个轨迹的误差
- 而不是单步误差
- 需要长序列训练

---

## 6. 区分实验

### 实验 1: 不同 Rollout 长度

- 比较 rollout = 1, 5, 10, 20, 50, 100
- 观察 H=100 和 H=500 的性能变化
- 如果更长 rollout 改善，说明分布偏移是问题

### 实验 2: Scheduled Sampling

- 训练时随机使用真实/预测状态
- 比较 scheduled sampling vs 纯 teacher forcing
- 如果改善，说明分布偏移是问题

### 实验 3: Trajectory Loss

- 使用整个轨迹的损失训练
- 比较 trajectory loss vs 单步 loss
- 如果改善，说明需要长期目标

---

## 7. 预期结果

| 方法 | H=10 | H=100 | H=500 | 预期 |
|------|------|-------|-------|------|
| V9 (rollout=20) | 0.0628 | 0.5064 | 0.5529 | 基线 |
| Rollout=50 | ? | ? | ? | 可能改善 |
| Rollout=100 | ? | ? | ? | 可能改善或爆炸 |
| Scheduled Sampling | ? | ? | ? | 可能改善 |
| Trajectory Loss | ? | ? | ? | 可能改善 |

---

## 8. 结论

**待验证**: 分布偏移是否是长时域误差的主要瓶颈

**区分实验**: 比较不同 rollout 长度和 scheduled sampling

**预期**: 如果分布偏移是主要瓶颈，增加 rollout 长度或使用 scheduled sampling 应该显著改善 H=100+ 的性能
