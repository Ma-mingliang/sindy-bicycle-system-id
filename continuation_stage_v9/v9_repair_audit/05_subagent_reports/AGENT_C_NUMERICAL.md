# AGENT C: 数值与数据流审计报告

## 审计入口

- `NeuralODEV9.train()` (neural_ode_v9.py L74): 归一化训练数据，构建训练目标，运行多步 rollout 损失训练
- `NeuralODEV9.predict()` (neural_ode_v9.py L190): 单步推理，反归一化输出
- `NeuralODEV12.train()` (neural_ode_v12.py L823): V12 不同归一化约定
- `NeuralODEV12.predict()` (neural_ode_v12.py L745): V12 单步推理
- `multi_step_evaluate()` (evaluation_v9.py L39): 评估循环调用 model.predict()

---

## 问题 #1: V9 多步 Rollout 归一化不匹配（已确认 BUG）

**严重度: CRITICAL** — 此 BUG 破坏了多步 rollout 损失，这是学习长 horizon 动力学的主要机制。

**文件**: `canonical_node/neural_ode_v9.py`, L131-137

**代码**:
```python
s_cur = sb[:n_roll].clone()
for step in range(rollout_steps):
    a_cur = ab[step:step + n_roll]
    dsdt = self._model(s_cur, a_cur)
    s_cur = s_cur + dsdt * self._dt    # <-- BUG
    target = sb[step + 1:step + 1 + n_roll]
    loss_multi = loss_multi + nn.functional.mse_loss(s_cur, target)
```

**数学推导**:

设:
- `s_phys` = 物理状态向量
- `delta = s_next_phys - s_phys` = 物理状态变化
- `state_std[i]` = 状态 `i` 的标准差
- `delta_std[i]` = `delta[i]` 的标准差
- `dt = 1/30`

V9 训练目标构建 (L92):
```
y = delta / (delta_std * dt)
```

模型学习: `model(s_norm, a_norm) = delta / (delta_std * dt)`

DataLoader 输入 `sb` = `states / state_std`（归一化状态）。

多步 rollout 需要下一个归一化状态:
```
s_next_norm = s_next_phys / state_std = (s_phys + delta) / state_std
            = s_norm + delta / state_std
```

模型输出 `dsdt = delta / (delta_std * dt)`，所以:
```
delta = dsdt * delta_std * dt
```

代入:
```
s_next_norm = s_norm + dsdt * delta_std * dt / state_std
```

但 V9 计算:
```
s_cur_new = s_cur + dsdt * dt = s_norm + delta / delta_std
```

要使两者相等，需要:
```
delta / delta_std = delta / state_std
=> delta_std = state_std  (对所有分量)
```

**通常 `delta_std != state_std`。** delta（样本间状态变化）与状态本身有不同的分布。例如横向误差 `e_y` 的 `state_std ~ 0.5m` 但 `delta_std ~ 0.01m`（小的逐步变化）。这种不匹配可能相差一个数量级或更多。

**具体误差大小**: 如果 `delta_std[0] / state_std[0] = 0.02`（对 `e_y` 典型），则每步 rollout 增量放大 50 倍。20 步后误差灾难性累积。

**影响**: 多步损失训练模型匹配损坏的轨迹，而非真实归一化轨迹。这解释了为什么 V9 的多步损失几乎无益于单步训练，以及为什么长 horizon rollout 快速退化。

**修复**: 将 L135 替换为:
```python
s_cur = s_cur + dsdt * self._dt * (self._delta_std / self._state_std)
```

---

## 问题 #2: V9 一致性损失归一化 BUG（已确认 BUG）

**严重度: MEDIUM** — 降低一致性正则化项的质量。

**文件**: `canonical_node/neural_ode_v9.py`, L143-148

**代码**:
```python
s_next_norm = sb + pred * self._dt
theta_pred = s_next_norm[:, 3]
theta_dot = sb[:, 4]
theta_gt = sb[:, 3] + theta_dot * self._dt    # <-- BUG
loss_consistency = nn.functional.mse_loss(theta_pred, theta_gt)
```

**数学推导**:

物理运动学约束:
```
theta_next = theta + theta_dot * dt
```

归一化空间:
```
theta_next / state_std[3] = theta / state_std[3] + (theta_dot / state_std[4]) * dt * (state_std[4] / state_std[3])
```

V9 计算:
```
theta_gt = sb[:, 3] + sb[:, 4] * dt
         = theta / state_std[3] + theta_dot / state_std[4] * dt
```

正确公式:
```
theta_gt = theta / state_std[3] + theta_dot / state_std[4] * dt * (state_std[4] / state_std[3])
         = theta / state_std[3] + theta_dot * dt / state_std[3]
```

V9 的公式缺少因子 `state_std[4] / state_std[3]`。

**V12 修复** (L1030):
```python
self._std_ratio = float(state_std[4] / state_std[3])
...
theta_gt = s_flat[:, 3] + s_flat[:, 4] * self._dt * self._std_ratio
```

---

## 问题 #3: V9 多步 Rollout 时间顺序（已确认 BUG）

**严重度: CRITICAL** — 与问题 #1 叠加。

**文件**: `canonical_node/neural_ode_v9.py`, L95-97 和 L131-137

**问题**: DataLoader 使用 `shuffle=True` (L96)。多步损失假设 `sb[i+1]` 是 `sb[i]` 的时间后继:
```python
s_cur = sb[:n_roll].clone()  # sb[0..n_roll-1]
...
target = sb[step + 1:step + 1 + n_roll]  # sb[step+1..step+n_roll]
```

洗牌后，`sb[i]` 和 `sb[i+1]` 是来自不同驾驶段的随机时间步。时间连续性假设被破坏。

**V12 修复**: 使用 `SequentialSegmentDataset` 返回连续轨迹段。段间洗牌，段内不洗牌。

---

## 问题 #4: V9 predict() 单步正确

**文件**: `canonical_node/neural_ode_v9.py`, L190-198

**验证**:
```
dsdt_norm = delta / (delta_std * dt)    [模型输出]
dsdt = dsdt_norm * delta_std * dt = delta
s_next = s + delta = s_next_phys        [正确]
```

单步预测数学上正确。反归一化链正确反转训练归一化。

---

## 问题 #5: V12 归一化与 V9 不等价（设计选择）

**严重度: INFO** — 不同约定，非 bug。

**V9 约定**:
```
s_norm = s / state_std
target = delta / (delta_std * dt)     [速率，在 delta_std 单位]
```

**V12 约定**:
```
s_norm = s / state_std
target = delta / state_std            [归一化状态变化]
```

**关系**: V12 目标 = V9 目标 × `delta_std * dt / state_std`（逐分量）。

V12 的约定更简洁:
1. 模型输出直接给出归一化状态增量
2. 积分简单: `s_next = s + model(s, a)`
3. 无混淆速率与逐步变化的风险

---

## 问题 #6: V12 积分函数注释误导（非 BUG）

**严重度: LOW** — 代码正确，注释错误。

**文件**: `canonical_node/neural_ode_v12.py`, L311-317

注释说 "targets = deltas / (delta_std * dt)"，这是 V9 的约定。V12 实际训练目标是 `deltas / state_std`。

---

## 问题 #7: V12 RK4 积分正确

**严重度: INFO** — 数学验证正确。

---

## 问题 #8: V12 euler_substep 积分正确

---

## 问题 #9: 残差模型使用 V9 predict() 一致

**严重度: INFO** — 与 V9 单步约定一致。

---

## 问题 #10: 物理限制检查正确

---

## 问题 #11: NMAE 计算正确

---

## 问题 #12: 梯度流无 Detach

**严重度: INFO** — 设计意图，有缓解措施（梯度裁剪 max_norm=1.0）。

---

## 汇总表

| # | 问题 | 严重度 | V9 | V12 | 描述 |
|---|------|--------|----|----|------|
| 1 | 多步 rollout 归一化 | CRITICAL | BUG | 已修复 | rollout 更新缺少 `delta_std/state_std` 因子 |
| 2 | 一致性损失归一化 | MEDIUM | BUG | 已修复 | 缺少 `state_std[4]/state_std[3]` 比率 |
| 3 | 时间顺序 | CRITICAL | BUG | 已修复 | shuffle=True 破坏多步损失假设 |
| 4 | predict() 单步 | OK | OK | OK | V9 反归一化链正确 |
| 5 | 归一化约定 | INFO | N/A | N/A | V12 使用不同但一致的约定 |
| 6 | 误导性注释 | LOW | N/A | BUG | 注释描述 V9 约定 |
| 7 | RK4 积分 | OK | N/A | OK | 数学验证正确 |
| 8 | euler_substep | OK | N/A | OK | 正确的子步累积 |
| 9 | 残差模型一致性 | OK | OK | N/A | 正确使用 V9 predict() |
| 10 | 物理限制单位 | OK | OK | OK | 均为物理单位 |
| 11 | NMAE 计算 | OK | OK | OK | 无量纲比率，正确 |
| 12 | 梯度流 | OK | OK | OK | 设计意图 |

---

## 建议

- **修复 V9 多步 rollout** (问题 #1): 在 L135 添加 `self._delta_std / self._state_std` 因子
- **修复 V9 一致性损失** (问题 #2): 添加 `state_std[4] / state_std[3]` 比率
- **修复 V9 时间顺序** (问题 #3): 采用 V12 的 `SequentialSegmentDataset`
- **修复 V12 注释** (问题 #6): 更新积分函数文档
- **新模型使用 V12 归一化约定**: `delta / state_std` 更简洁不易出错

---

*报告由 Agent C（数值与数据流审计）生成*
*日期: 2026-06-28*
