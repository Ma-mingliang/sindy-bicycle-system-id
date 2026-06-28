# AGENT A: V9 版本与上下文审计报告

## 1. V9 vs V8 代码差异（行为层面）

### 1.1 模型架构变更
- **V8**: 使用 GP（高斯过程）和 SINDy（多项式回归）模型，位于 `canonical_7d/models.py`
- **V9**: 用 Neural ODE（`neural_ode_v9.py`）替代 GP/SINDy：多层 MLP，hidden=128/64，depth=2/3，tanh 激活，学习 dx/dt = f(x, u)

**行为影响**: V8 模型基于内存（GP 存储训练数据），V9 是参数化函数逼近器。过拟合和泛化特性根本不同。

### 1.2 训练目标归一化（关键差异）
- **V8** (`data_loader.py` L56): `Y = deltas / delta_std` — GP/SINDy 学习归一化 delta
- **V9** (`neural_ode_v9.py` L92): `train_dsdot = deltas / (self._delta_std * self._dt)` — 学习 ds/dt（速率）

**行为影响**: V9 额外缩放了 `1/dt` 因子（dt=1/30，目标放大 ~30 倍）。改变梯度景观和损失尺度。

### 1.3 多步 Rollout 损失
- **V8**: 无多步训练损失，每个样本独立
- **V9** (`neural_ode_v9.py` L128-138): 添加多步 rollout 损失

### 1.4 数据加载与洗牌
- **V8**: 无 DataLoader 洗牌问题
- **V9** (`neural_ode_v9.py` L94-97): `DataLoader(shuffle=True)` 随机打乱时间步

### 1.5 训练课程
- **V8**: 固定训练配置，无课程概念
- **V9**: 使用 rollout 课程 '1,5,10,20'

### 1.6 评估
- **V8**: 三种评估模式，segment_length=200
- **V9**: 简化为 NMAE + 物理存活检查，segment_length=1100

### 1.7 物理限制检查
- **V8**: 单一 `state_limit=100.0`
- **V9**: 每个状态维度独立限制（e_y=5, e_psi=π, v=5 等）

### 1.8 附加损失项
- **V9** 添加一致性损失、Jacobian 正则化、物理约束损失
- **V8** 无辅助损失

---

## 2. V9 实际执行路径

### 2.1 数据管道
1. `data_loader_v9.py` 通过 `sys.path` 注入调用 V8 的 `load_7d_data()`
2. V8 加载器读取 `stage2_dataset_150k.npz`，提取 7D 状态（丢弃 index 5 的 kappa）
3. 按 episode 分割训练/测试（75%/25%），计算 state_std, action_std, delta_std

### 2.2 训练路径 (NeuralODEV9.train)
1. 归一化状态: `train_s = states / state_std` (L90)
2. 归一化动作: `train_a = actions / action_std` (L91)
3. **归一化目标**: `train_dsdot = deltas / (delta_std * dt)` (L92) ← 关键归一化
4. 创建 `TensorDataset` + `DataLoader(shuffle=True)` (L94-97)
5. 每个 epoch:
   - 计算 `rollout_steps` (L117-120)
   - 单步损失: MSE(model(sb, ab), yb) (L124-125)
   - 多步损失: rollout rollout_steps 步 (L128-138)
   - 一致性损失: theta 一致性 (L141-148)
   - Jacobian 正则化 (L151-163)
   - 总损失: single + 0.3*multi + 0.1*consistency + 0.01*jacobian (L165-168)

### 2.3 预测路径 (NeuralODEV9.predict)
1. 归一化输入: `s_norm = s / state_std`, `a_norm = tau / action_std` (L191-192)
2. 模型前向: `dsdt_norm = model(s_norm, a_norm)` (L195)
3. 反归一化: `dsdt = dsdt_norm * delta_std * dt` (L197)
4. Euler 步: `return s + dsdt` (L198)

**关键分析**: 单步预测的归一化链是一致的。但多步训练 rollout 使用洗牌数据，假设 `sb[i+1]` 是 `sb[i]` 的时间后继。

### 2.4 评估路径
1. 对每个 horizon H 和每个 segment:
   - 从 `s0 = segment['states'][0]` 开始
   - 每步: `s_next = model.predict(s_cur, actions[step])`
   - 检查存活（物理限制）
   - 计算逐步 NMAE

---

## 3. 发现的问题

### 问题 1: shuffle=True 破坏多步损失
**状态**: 已确认
**严重度**: CRITICAL

**代码证据**:
- `neural_ode_v9.py` L96-97: `DataLoader(ds, batch_size=..., shuffle=True)`
- `neural_ode_v9.py` L133-137: 多步 rollout 目标使用 `sb[step + 1:step + 1 + n_roll]`
- `neural_ode_v12.py` L130-138: V12 文档明确指出此问题

**分析**: shuffle=True 后，每个 batch 包含来自随机轨迹的随机时间步。多步损失假设 batch 中相邻样本是时间相邻的，实际是随机配对，损失变为纯噪声。

**V12 修复**: 使用 `SequentialSegmentDataset` 返回连续轨迹段，在段间洗牌而非时间步间洗牌。

### 问题 2: 训练 Rollout 上限 (20) vs 评估 (500+)
**状态**: 已确认
**严重度**: CRITICAL

**代码证据**:
- `config_v9.py` L24: `rollout_curriculum = '1,5,10,20'` — 最大训练 rollout 20 步
- `config_v9.py` L13: `ROLLOUT_HORIZONS = [1, 5, 10, 20, 50, 100, 200, 500, 1000]` — 评估到 1000
- `neural_ode_v9.py` L117-120: 课程逻辑永不设置 rollout_steps > 20

**分析**: 即使多步损失正确，模型也只训练到 H=20。H=500 评估需要外推 25 倍。

### 问题 3: V9 与 V12 归一化差异
**状态**: 已确认（信息性）
**严重度**: HIGH → 降级为 INFORMATIONAL

**代码证据**:
- V9 L92: `deltas / (delta_std * dt)` — 按 delta_std*dt 归一化
- V12 L875: `deltas / state_std` — 按 state_std 归一化

**分析**: 两种方法各自内部一致。V9 方法更符合 ODE 原理（模型学习 dx/dt），V12 方法更简洁。**这不是 bug，是设计选择。**

### 问题 4: 长 Rollout 导致常数预测崩溃
**状态**: 高概率
**严重度**: HIGH

**分析**: V9 基线的 NMAE 曲线（H=1: 0.006 → H=500: 0.998，100% 存活）完全符合常数预测器的特征：
1. H=1 误差小（初始状态接近下一状态）
2. 100% 存活（初始状态在物理限制内）
3. H=500 NMAE ≈ 1.0（归一化均值约为 0）

**置信度**: 85%。需要检查实际预测值才能完全确认。

### 问题 5: 一致性损失归一化错误
**状态**: 高概率
**严重度**: MEDIUM

**代码证据** (`neural_ode_v9.py` L143-148):
```python
theta_gt = sb[:, 3] + theta_dot * self._dt  # 缺少 std_ratio
```
物理约束: `theta_next = theta + theta_dot * dt`
归一化空间: 需要 `state_std[4] / state_std[3]` 修正因子

**V12 修复** (L1030): `theta_gt = s_flat[:, 3] + s_flat[:, 4] * self._dt * self._std_ratio`

### 问题 6: 多步 Rollout 的 dt 缩放
**状态**: 排除（内部一致）
**严重度**: N/A

**重新评估**: V9 rollout `s_cur += dsdt * dt = s_norm + delta/delta_std`，这是归一化空间的正确结果。问题不在 dt 缩放，而在洗牌的时间顺序。

### 问题 7: V9 MLP 架构局限性
**状态**: 高概率
**严重度**: MEDIUM

**分析**: 简单 MLP + tanh + 2-3 层可能缺乏表示复杂自行车动力学的能力。最后一层小初始化（gain=0.1）偏向近零输出，与洗牌问题结合更容易收敛到恒等函数。

---

## 4. 问题汇总表

| # | 问题 | 状态 | 严重度 | V12 修复状态 |
|---|------|------|--------|-------------|
| 1 | shuffle=True 破坏多步损失 | 已确认 | CRITICAL | 已修复 |
| 2 | 训练 rollout 上限 20 vs 评估 500+ | 已确认 | CRITICAL | 部分修复 |
| 3 | 归一化 delta_std*dt vs state_std | 信息性 | INFO | 设计差异 |
| 4 | 长 rollout 常数预测崩溃 | 高概率 | HIGH | 部分解决 |
| 5 | 一致性损失尺度错误 | 高概率 | MEDIUM | 已修复 |
| 6 | 多步 dt 缩放 | 排除 | N/A | N/A |
| 7 | MLP 架构容量不足 | 高概率 | MEDIUM | 未解决 |

---

## 5. 自我挑战

### 挑战 1: shuffle=True 诊断是否确定？
- v9_fixed_seq（仅修复 shuffle）H=500 NMAE=0.476，但存活率 0%
- 如果 shuffle 是唯一问题，修复后应同时恢复精度和稳定性
- **结论**: shuffle 是必要但不充分条件

### 挑战 2: 归一化差异是否真的是问题？
- 两种方法各自内部一致
- **结论**: 重新分类为 INFORMATIONAL

### 挑战 3: V9 基线结果能否用其他方式解释？
- 替代解释：初始状态偏置、物理限制宽松、测试数据特性
- **置信度**: 85% 是常数预测

### 挑战 4: V12 修复是否充分？
- v9_fixed_seq: H=500 NMAE=0.476, 0% 存活
- v9_fixed_seq_contract: H=500 NMAE=0.958, 100% 存活
- 均不满足验收标准
- **更深层问题**: 7D Neural ODE 架构是否存在根本局限？

---

*报告由 Agent A（V9 版本与上下文审计）生成*
*日期: 2026-06-28*
*读取文件: 14 个源文件，跨 V8 和 V9*
