# B: 状态归一化与动力学审计

## 1. 状态空间定义

### 7D 状态向量 (config_v9.py:3)

| 索引 | 名称 | 含义 | 物理单位 |
|------|------|------|----------|
| 0 | e_y | 横向误差 | m |
| 1 | e_psi | 航向误差 | rad |
| 2 | v | 速度 | m/s |
| 3 | theta | 路径参数 | rad |
| 4 | theta_dot | 路径参数导数 | rad/s |
| 5 | delta | 前轮转角 | rad |
| 6 | delta_dot | 前轮转角导数 | rad/s |

### 8D→7D 映射 (data_loader.py:26-29)

`IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]` 丢弃索引 5（kappa，曲率）。数据集注释称 kappa "恒为零"。

**风险**: 无运行时验证。如果 kappa 非零，丢弃会导致信息丢失。

## 2. 物理限制 (config_v9.py:14-22)

```python
PHYSICAL_LIMITS = {
    'e_y': 5.0,        # 横向误差 ±5m
    'e_psi': np.pi,    # 航向误差 ±π rad
    'v': 5.0,          # 速度 ±5 m/s
    'theta': np.pi,    # 路径参数 ±π rad
    'theta_dot': 10.0, # 路径参数导数 ±10 rad/s
    'delta': np.pi/2,  # 前轮转角 ±π/2 rad
    'delta_dot': 10.0, # 前轮转角导数 ±10 rad/s
}
```

**用途**: 仅用于评估存活率检测，**未**用于训练约束或状态裁剪。

## 3. V9 归一化方案 (neural_ode_v9.py)

### 3.1 标准差计算

```python
# data_loader.py:68-77 (从训练数据计算)
state_std = np.std(train_states, axis=0)   # shape (7,)
delta_std = np.std(train_deltas, axis=0)   # shape (7,)
```

### 3.2 训练目标归一化 (neural_ode_v9.py:92)

```python
train_dsdot = torch.FloatTensor(deltas / (self._delta_std * self._dt))
```

**含义**: 目标是**速率**（delta / (std * dt)），单位为"标准差/秒"。

### 3.3 推理去归一化 (neural_ode_v9.py:197)

```python
dsdt = dsdt_norm * self._delta_std * self._dt
s_next = s + dsdt
```

**模型输出**: 归一化速率 → 乘以 delta_std * dt → 得到物理空间增量 → 加到当前状态。

### 3.4 V9 归一化空间一致性

V9 在**物理空间**中进行 rollout：
- 输入: 物理空间状态 s
- 输出: 归一化速率 dsdt_norm
- 去归一化: dsdt = dsdt_norm * delta_std * dt
- 更新: s_next = s + dsdt (物理空间)

**结论**: V9 的归一化在单步推理中是自洽的。

## 4. V12 归一化方案对比 (neural_ode_v12.py)

### 4.1 训练目标

```python
targets = deltas / state_std  # delta 归一化（非速率）
```

### 4.2 推理

```python
s_norm = s / state_std
dsdt_norm = model(s_norm, u)
s_next_norm = s_norm + dsdt_norm
s_next = s_next_norm * state_std
```

### 4.3 V12 归一化空间

V12 在**归一化空间**中进行 rollout：
- 输入: s_norm = s / state_std
- 模型输出: dsdt_norm（归一化增量）
- 更新: s_next_norm = s_norm + dsdt_norm
- 去归一化: s_next = s_next_norm * state_std

## 5. 关键发现：多步 Rollout 归一化不匹配

### 5.1 V9 的隐含假设

V9 的多步 rollout（neural_ode_v9.py:128-138）：

```python
for step in range(1, rollout_len):
    dsdt = self.ode_func(s_cur, u_cur)  # 输出: 归一化速率
    s_cur = s_cur + dsdt * self._dt      # BUG: 混合空间
```

**问题**: `s_cur` 在物理空间，`dsdt * self._dt` 的量纲是 `delta_std * dt * dt`（非物理增量）。正确的应该是 `dsdt * delta_std * dt`。

### 5.2 影响分析

这个 bug 意味着多步 rollout 中的每步更新使用了错误的缩放因子：
- 正确: `s + dsdt_norm * delta_std * dt`
- 实际: `s + dsdt_norm * dt`

差了一个 `delta_std` 因子（各维度约 0.001-0.1），导致多步更新的增量**远小于**应有值。

**悖论性好处**: 这个 bug 使 V9 的多步预测偏保守（增量被缩小），恰好保持了存活率。

### 5.3 V12 的修复与新问题

V12 修复了归一化空间一致性，但引入了新问题：
- 扩展 rollout 课程（1→100 步）暴露了模型在长 horizon 上的不稳定性
- 常数预测崩溃：模型学会输出零增量以最小化长 horizon 损失

## 6. Std 计算问题

### 6.1 零 Std 保护 (data_loader.py:74-77)

```python
state_std[state_std < 1e-10] = 1.0
delta_std[delta_std < 1e-10] = 1.0
```

**风险**: 静默掩盖常量特征。如果某维度恒为零（如 kappa），std 替换为 1.0 后该维度的归一化信号被压缩到接近零。

### 6.2 重复保护 (neural_ode_v9.py:82-83)

```python
self._delta_std = np.maximum(delta_std, 1e-8)
```

与 data_loader 的保护重复，阈值不同（1e-8 vs 1e-10）。

## 7. 积分方法

### 7.1 V9: 前向 Euler (neural_ode_v9.py:135)

```python
s_cur = s_cur + dsdt * self._dt  # dt = 1/30
```

**截断误差**: O(dt²) = O(1/900) ≈ 0.0011 每步

### 7.2 V12: RK4 (neural_ode_v12.py)

```python
k1 = f(s)
k2 = f(s + dt/2 * k1)
k3 = f(s + dt/2 * k2)
k4 = f(s + dt * k3)
s_next = s + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
```

**截断误差**: O(dt⁵) = O(1/30⁵) ≈ 4e-8 每步

RK4 在单步精度上显著优于 Euler，但增加了 4 倍计算量。

## 8. 总结

| 问题 | 严重度 | 影响 |
|------|--------|------|
| V9 多步 rollout 归一化不匹配 | HIGH | 增量被缩小 delta_std 倍，保守但错误 |
| V12 归一化修复后暴露长 horizon 不稳定性 | HIGH | 常数预测崩溃，0% 存活率 |
| 无物理限制约束训练 | MEDIUM | 模型可能学到物理上不可能的状态 |
| 零 std 静默替换 | LOW | 掩盖常量特征但影响有限 |
| 重复 std 保护 | LOW | 代码冗余，无功能影响 |
