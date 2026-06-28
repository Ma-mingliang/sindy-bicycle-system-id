# V9 架构详细分析

**更新时间**: 2026-06-28 18:00
**RUN_ID**: 20260628_175824_neural_ode_72h

---

## 1. 模型架构

### ODEFunc (核心网络)
```python
class ODEFunc(nn.Module):
    def __init__(self, hidden=64, depth=3, activation='tanh'):
        # 输入层: Linear(8, 64) + Tanh
        # 隐藏层: Linear(64, 64) + Tanh (重复 depth-1 次)
        # 输出层: Linear(64, 7)
        # 最后一层初始化: bias=0, weight=Xavier(gain=0.1)
```

### 参数量计算
- 输入层: 8 * 64 + 64 = 576
- 隐藏层: (64 * 64 + 64) * 2 = 8,320
- 输出层: 64 * 7 + 7 = 455
- **总计: 9,351**

---

## 2. 训练流程

### 数据准备
```python
train_s = states / state_std          # 归一化状态
train_a = actions / action_std        # 归一化动作
train_dsdot = deltas / (delta_std * dt)  # 归一化导数目标
```

### 损失函数
```
L = L_single + λ_multi * L_multi + λ_consistency * L_consistency + λ_jacobian * L_jacobian
```

- **L_single**: 单步导数预测 MSE
- **L_multi**: 多步 rollout 状态预测 MSE
- **L_consistency**: theta 与 theta_dot 的物理一致性
- **L_jacobian**: 动力学函数的 Jacobian 平滑正则化

### 课程学习
```
epoch 0-49:   rollout_steps = 1
epoch 50-99:  rollout_steps = 5
epoch 100-149: rollout_steps = 10
epoch 150-199: rollout_steps = 20
```

### 优化器
- Adam, lr=1e-3, weight_decay=0
- CosineAnnealingLR
- 梯度裁剪: max_norm=1.0

---

## 3. 推理流程

### 单步预测
```python
def predict(s, tau):
    s_norm = s / state_std
    a_norm = tau / action_std
    dsdt_norm = model(s_norm, a_norm)
    dsdt = dsdt_norm * delta_std * dt
    return s + dsdt
```

### 积分方式
- **当前**: Euler 方法 `x_{t+1} = x_t + dt * f_θ(x_t, u_t)`
- **未使用**: RK4 或其他高阶方法

---

## 4. 评估协议

### 多步评估
```python
def multi_step_evaluate(model, segments, state_std, horizons):
    for h in horizons:
        for seg in segments:
            s_cur = seg['states'][0]
            for step in range(h):
                s_next = model.predict(s_cur, actions[step])
                # 检查存活条件
                # 计算 NMAE
```

### 存活条件
- 数值存活: 无 NaN/Inf, |state| < 100
- 物理存活: 各状态在物理限制内
  - e_y: ±5m, e_psi: ±π, v: ±5m/s
  - theta: ±π, theta_dot: ±10rad/s
  - delta: ±π/2, delta_dot: ±10rad/s

### NMAE 计算
```
NMAE = mean(|predicted - actual| / state_std)
```

---

## 5. 配置搜索空间

| 配置 | hidden | depth | activation | lr | wd |
|------|--------|-------|------------|-----|-----|
| 1 | 128 | 2 | tanh | 1e-3 | 0 |
| 2 | 128 | 2 | tanh | 3e-4 | 1e-5 |
| 3 | 128 | 3 | tanh | 1e-3 | 0 |
| 4 | 64 | 3 | tanh | 1e-3 | 0 |
| 5 | 128 | 2 | silu | 1e-3 | 0 |
| 6 | 128 | 2 | tanh | 1e-4 | 1e-4 |

**最佳配置**: config3_seed43 (hidden=64, depth=3, tanh, lr=1e-3, seed=43)

---

## 6. 关键发现

1. **depth=3 显著优于 depth=2** (score 0.99 vs 1.20)
2. **Tanh 激活优于 SiLU/ReLU**
3. **lr=1e-3 优于 lr=1e-4**
4. **seed=43 表现最佳**
5. **残差方法无显著收益** — 15种变体均未超越纯 Neural ODE
