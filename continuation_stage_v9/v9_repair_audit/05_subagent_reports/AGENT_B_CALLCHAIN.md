# AGENT B: 代码调用链审计报告

## 发现 1: V9 DataLoader shuffle=True 破坏多步 Rollout (CRITICAL)

**问题**: shuffled DataLoader 破坏多步损失的时间顺序

**涉及文件**: `canonical_node/neural_ode_v9.py`
**涉及函数**: `NeuralODEV9.train()` (L94-97, L128-138)

**实际行为**:
V9 创建 `TensorDataset` 并洗牌所有时间步:
```python
ds = torch.utils.data.TensorDataset(train_s, train_a, train_dsdot)
loader = torch.utils.data.DataLoader(ds, batch_size=self.config.batch_size, shuffle=True)
```

多步 rollout (L128-138) 假设 `sb[i+1]` 是 `sb[i]` 的时间后继:
```python
s_cur = sb[:n_roll].clone()
for step in range(rollout_steps):
    a_cur = ab[step:step + n_roll]
    dsdt = self._model(s_cur, a_cur)
    s_cur = s_cur + dsdt * self._dt
    target = sb[step + 1:step + 1 + n_roll]
    loss_multi = loss_multi + nn.functional.mse_loss(s_cur, target)
```

shuffle=True 后，`sb[0]` 和 `sb[1]` 是来自不同 episode 的随机时间步。rollout 使用 action `ab[step]` 但比较状态 `sb[step+1]`（完全不同的随机时间步）。多步损失变为纯噪声。

**V12 修复**: `SequentialSegmentDataset` 返回连续轨迹段。`shuffle=True` 在段间洗牌（而非段内），保持时间顺序。

**置信度**: 99%

---

## 发现 2: V9 多步 Rollout 使用错误归一化尺度 (CRITICAL)

**问题**: Rollout 混合 delta_std 和 state_std 归一化

**涉及文件**: `canonical_node/neural_ode_v9.py`
**涉及函数**: `NeuralODEV9.train()` L92, L135; `NeuralODEV9.predict()` L190-198

**实际行为**:
V9 训练目标 (L92):
```python
train_dsdot = torch.FloatTensor(deltas / (self._delta_std * self._dt))
```
模型学习: `output = deltas / (delta_std * dt)`

V9 rollout 步 (L135):
```python
s_cur = s_cur + dsdt * self._dt  # s_cur 是 s/state_std，dsdt 是 deltas/(delta_std*dt)
```
结果: `s_cur + dsdt * dt` = `s_norm + deltas / delta_std`

但 rollout 目标 (L136):
```python
target = sb[step + 1:step + 1 + n_roll]  # = s_next / state_std
```

预测: `s/state_std + deltas/delta_std`
目标: `s/state_std + deltas/state_std`

除非 `delta_std[i] == state_std[i]` 对所有状态 i，否则两者不同。对自行车模型，delta_std 和 state_std 通常是不同的量（状态变化的标准差 vs 状态的标准差），所以每步引入系统偏差。

**V12 修复**: 训练目标 `deltas / state_std` (L875)，积分 `s + model(s, a)` (L317)，保持在 state 归一化空间。

**数学证明**:
- V9: prediction = `s_norm + deltas/delta_std`, target = `s_norm + deltas/state_std`
- 每步差异: `deltas * (1/delta_std - 1/state_std)` -- delta_std != state_std 时非零
- 误差随 rollout 步累积

**置信度**: 99%

---

## 发现 3: V9 单步 predict() 数学正确

**问题**: V9 predict() 反归一化正确抵消

**涉及文件**: `canonical_node/neural_ode_v9.py`
**涉及函数**: `NeuralODEV9.predict()` (L190-198)

**实际行为**:
```python
def predict(self, s, tau):
    s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0)
    a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0)
    with torch.no_grad():
        dsdt_norm = self._model(s_norm, a_norm).numpy()[0]
    dsdt = dsdt_norm * self._delta_std * self._dt
    return s + dsdt
```

往返验证:
1. 模型输出: `dsdt_norm` ≈ `deltas / (delta_std * dt)`
2. 反归一化: `dsdt = dsdt_norm * delta_std * dt` = `deltas / (delta_std * dt) * delta_std * dt` = `deltas`
3. 返回: `s + deltas` = `s_next` -- 正确

单步预测正确，因为 `delta_std * dt` 因子在反归一化中抵消。

**置信度**: 98%

---

## 发现 4: V9 一致性损失缺少 std_ratio 修正 (MEDIUM)

**问题**: 一致性损失混合归一化基底而无比率修正

**涉及文件**: `canonical_node/neural_ode_v9.py`
**涉及函数**: `NeuralODEV9.train()` (L141-148)

**实际行为**:
```python
s_next_norm = sb + pred * self._dt  # pred = deltas/(delta_std*dt), 所以 = s_norm + deltas/delta_std
theta_pred = s_next_norm[:, 3]      # theta/state_std[3] + delta_theta/delta_std[3]
theta_dot = sb[:, 4]                # theta_dot/state_std[4]
theta_gt = sb[:, 3] + theta_dot * self._dt  # theta/state_std[3] + (theta_dot/state_std[4]) * dt
```

物理约束: `theta_next = theta + theta_dot * dt`
归一化空间: `theta_next/state_std[3] = theta/state_std[3] + theta_dot * dt / state_std[3]`

V9 的 `theta_gt` = `theta/state_std[3] + (theta_dot/state_std[4]) * dt`

仅当 `state_std[3] == state_std[4]` 时正确。通常 theta（弧度）和 theta_dot（弧度/秒）标准差不同。

V12 修复 (L1030): `theta_gt = s_flat[:, 3] + s_flat[:, 4] * self._dt * self._std_ratio`

**置信度**: 97%

---

## 发现 5: V12 _integrate_euler 有误导性注释 (LOW, 仅文档)

**问题**: V12 积分注释引用 V9 归一化

**涉及文件**: `canonical_node/neural_ode_v12.py`
**涉及函数**: `_integrate_euler()` (L310-317)

**实际行为**:
```python
def _integrate_euler(model, s, a, dt):
    """Standard Euler integration: s_next = s + f(s, a).

    NOTE: The model is trained on targets = deltas / (delta_std * dt),
    so the model output is ALREADY scaled by dt. Do NOT multiply by dt again.
    """
    return s + model(s, a)
```

注释说 "targets = deltas / (delta_std * dt)"，这是 V9 的归一化。V12 实际训练 `deltas / state_std` (L875)。代码正确但注释误导。

**置信度**: 100%

---

## 发现 6: V12 存储 _delta_std 但预测中未使用 (INFO)

**问题**: V12 _delta_std 存储但未使用

**涉及文件**: `canonical_node/neural_ode_v12.py`
**涉及函数**: `NeuralODEV12.train()` (L852), `NeuralODEV12.predict()` (L745-772)

**实际行为**:
V12 训练时存储 `self._delta_std = delta_std.copy()` (L852)，但 `predict()` 和 `predict_batch()` 都不引用它。这是设计: V12 模型输出在 state 归一化空间 (`deltas / state_std`)，反归一化只需 `state_std`。

这不是 bug -- 是正确的设计。但 V12 检查点中的 `_delta_std` 是未使用的数据。

**置信度**: 100%

---

## 发现 7: 导入链正确，无遮蔽 (VERIFIED)

**问题**: 模块导入验证清洁

**涉及文件**: `run_v9_fixed_rerun.py`, `canonical_node/data_loader_v9.py`, `canonical_node/__init__.py`

**实际行为**:
1. `run_v9_fixed_rerun.py` 添加 v8 和 v9 路径到 sys.path (L15-16)
2. `canonical_node/data_loader_v9.py` 添加 v8 路径并从 `canonical_7d.data_loader` 导入
3. V9 使用 `canonical_node` 包名; V8 使用 `canonical_7d` 包名 -- 无冲突
4. `canonical_node/__init__.py` 正确重新导出所有公共类

**置信度**: 100%

---

## 发现 8: 配置加载正确 (VERIFIED)

**问题**: NeuralODEConfig 和 V12Config 正确加载和应用

**涉及文件**: `run_v9_fixed_rerun.py`, `canonical_node/neural_ode_v9.py`, `canonical_node/neural_ode_v12.py`

**实际行为**:
- `run_v9_fixed_rerun.py` 显式创建配置 (L126-183)
- `NeuralODEV9.__init__` 接受可选配置，默认 `NeuralODEConfig()` (L66)
- `NeuralODEV12.__init__` 接受可选配置，默认 `V12Config()` (L614)
- 实验配置正确设置 `model_class='v12'` 或 `model_class='v9'`

**置信度**: 100%

---

## 发现 9: 评估调用链正确 (VERIFIED)

**问题**: multi_step_evaluate 正确调用 model.predict()

**涉及文件**: `canonical_node/evaluation_v9.py`, `run_v9_fixed_rerun.py`

**实际行为**:
```python
# evaluation_v9.py L74:
s_next = model.predict(s_cur, actions_seg[step])
```

V9 和 V12 的 `predict()` 都接受物理状态和物理动作，返回物理下一状态。评估接口一致。评估使用自回归 rollout（将 `s_cur = s_next` 传递），误差累积。

**置信度**: 100%

---

## 发现 10: V9 predict_batch() 正确但评估中未使用 (INFO)

**涉及文件**: `canonical_node/neural_ode_v9.py`, `canonical_node/evaluation_v9.py`

`predict_batch()` (L200-208) 使用与 `predict()` 相同的正确反归一化。但 `multi_step_evaluate()` 调用 `model.predict(s_cur, actions_seg[step])`（单样本），不调用 `predict_batch()`。

**置信度**: 100%

---

## 发现 11: V12 自适应课程在段过短时正确回退 (INFO)

**涉及文件**: `canonical_node/neural_ode_v12.py`
**涉及函数**: `SequentialSegmentDataset.set_max_horizon()` (L228-253)

当 `set_max_horizon(max_h)` 被调用且没有足够长的段时，数据集回退到所有段并发出警告。

**置信度**: 100%

---

## 发现 12: NeuralODEContractive 与 V9 有相同的多步 Rollout BUG (BY DESIGN)

**涉及文件**: `canonical_node/neural_ode_contractive.py`
**涉及函数**: `NeuralODEContractive.train()` (L274-289, L294-299)

`neural_ode_contractive.py` (V11) 有相同的:
1. `shuffle=True` (L243-244)
2. 错误归一化 rollout: `s_cur = s_cur + dsdt * self._dt` (L281-289)
3. 缺少 std_ratio (L294-299)

这是预期的 -- V11 在 V9 代码基础上构建，在 bug 被识别之前。

**置信度**: 99%

---

## 架构洞察

- **归一化空间分离**: V9 和 V12 都归一化模型输入，但 V9 的输出归一化与输入不同（delta_std vs state_std），在多步损失中造成不一致。V12 统一归一化: 输入和目标都用 state_std。

- **段采样**: V12 的 `SequentialSegmentDataset` 是关键结构修复。存储完整轨迹为 `Segment` dataclass，返回连续切片。DataLoader 在段间洗牌，不在段内洗牌。

- **课程调度**: V9 使用基于时间的简单课程 (L118-120)。V12 使用自适应调度器，基于验证误差推进。

---

## 稳健发现汇总

| ID | 严重度 | 摘要 |
|----|--------|------|
| BUG-001 | CRITICAL | shuffle=True 破坏 V9 多步 rollout 的时间顺序 |
| BUG-002 | CRITICAL | V9 rollout 混合 delta_std 和 state_std 归一化，导致系统误差累积 |
| BUG-003 | MEDIUM | V9 一致性损失缺少 theta 和 theta_dot 之间的 std_ratio 修正 |
| INFO-001 | INFO | V9 单步 predict() 尽管归一化不寻常但数学正确 |
| DOC-001 | LOW | V12 _integrate_euler 注释引用 V9 归一化（误导） |
| INFO-002 | INFO | V12 存储 _delta_std 但预测中未使用 |
| VER-001 | INFO | 导入链验证清洁 |
| VER-002 | INFO | 配置加载验证正确 |
| VER-003 | INFO | 评估调用链验证正确 |
| INFO-003 | INFO | predict_batch() 在评估路径中未使用 |
| INFO-004 | INFO | SequentialSegmentDataset 回退行为安全 |
| INFO-005 | INFO | NeuralODEContractive 共享 V9 bug（设计如此） |

---

## 新开发建议

- 遵循 V12 归一化约定: 训练 `deltas / state_std`，积分 `s + model(s, a)`，反归一化 `* state_std`
- 使用 `SequentialSegmentDataset` 模式进行任何未来多步损失计算
- 需要多步损失时不要使用 `shuffle=True` 对单个时间步
- 修复 V12 `_integrate_euler` 的过时注释

---

*报告由 Agent B（代码调用链审计）生成*
*日期: 2026-06-28*
