# 下一阶段基线审计报告

> **生成时间**: 2026-06-24
> **项目**: sindy_bicycle (D:\系统辨识作业\sindy_bicycle)
> **审计目的**: 建立可复现评估体系前的基线事实确认
> **证据级别**: CONFIRMED (代码验证), INFERRED (推理), UNKNOWN (未确认)

---

## 1. 项目路线边界

### 1.1 4D 路线 (Meijaard 基准)

| 项目 | 值 | 证据 |
|------|-----|------|
| 状态维度 | 4 | CONFIRMED |
| 状态定义 | [phi, delta, phi_dot, delta_dot] | CONFIRMED |
| 输入 | tau (转向力矩, N·m) | CONFIRMED |
| 控制周期 | dt = 1/30 s (30 Hz) | CONFIRMED |
| 前进速度 | v = 3.5 m/s | CONFIRMED |
| 系统稳定性 | 开环不稳定 (特征值实部 +1.54) | CONFIRMED |
| 动力学模型 | Meijaard 2007 非线性 + RK4 (5子步) | CONFIRMED |
| 控制器 | LQR (Q=diag([1000,100,10,1]), R=0.2) | CONFIRMED |

**代码路径**: `methods_common.py:14-80`

### 1.2 8D 路线 (简化 Whipple)

| 项目 | 值 | 证据 |
|------|-----|------|
| 状态维度 | 8 | CONFIRMED |
| 状态定义 | [ey, epsi, v, theta, theta_dot, k, delta, delta_dot] | CONFIRMED |
| 输入 | 转向力矩 | CONFIRMED |
| 数据文件 | bicycle_data.npz (468样本), bicycle_data_improved.npz (2083样本) | CONFIRMED |
| 模型类型 | SINDy 多项式 (55×8 系数矩阵) | CONFIRMED |

**代码路径**: `bicycle_dynamics.py:87-160`, `data_collector.py`, `world_model.py`

### 1.3 两条路线关系

**完全独立**，无共享组件：
- 不同的动力学模型 (Meijaard vs 简化 Whipple)
- 不同的数据文件
- 不同的归一化方式
- 不同的评估代码

**证据**: CONFIRMED by 代码检查

---

## 2. 当前模型输入、输出和预测公式

### 2.1 GP 基线模型

```
输入: 5D 归一化 [phi/std, delta/std, phi_dot/std, delta_dot/std, tau/action_std]
输出: 4D 归一化 delta (状态增量)
反归一化: s_next = s + delta_norm * delta_std
```

**GP 配置**:
- 核函数: `ConstantKernel(1.0) * RBF(length_scale=1.0)`
- 最大样本数: 5000 (随机下采样, 无固定 seed)
- GP 数量: 4 个独立 GP
- ARD: 否
- 优化器: L-BFGS (n_restarts_optimizer=2, alpha=1e-6)

**已知问题**: L-BFGS 收敛警告，最优值接近上界 100000.0

**证据**: CONFIRMED by `methods_classic.py:20-59`

### 2.2 NN 残差模型 (ResidualNet)

```
架构: 2层 128-hidden SiLU MLP
输入: 5D (4 state + 1 action, 归一化)
输出: 4D (状态增量残差, 归一化)
```

**证据**: CONFIRMED by `methods_nn.py:21-33`

### 2.3 GP+NN 集成预测公式

```
s_next = GP(s, tau) + mean(NN_1..n)(s_norm, a_norm) * delta_std * residual_scale
```

其中:
- `residual_scale = 0.3` (固定缩放)
- `mean(NN_1..n)` 是 n 个 NN 模型输出的均值

**证据**: CONFIRMED by `test_ensemble.py:147-153`

### 2.4 归一化方式

**纯 std 缩放，无均值减法**:

```
s_norm = s / state_std
a_norm = tau / action_std
delta_norm = delta / delta_std
```

**归一化常数** (从 30000 样本计算):
- `state_std = [0.288, 0.173, 1.154, 0.578]`
- `action_std = 28.867`
- `delta_std = [0.038, 0.066, 0.149, 3.633]`

**证据**: CONFIRMED by `methods_common.py:108-111`

---

## 3. 评估模式代码定义

### Mode A: Teacher Forcing

```python
# 每一步都使用真实状态
model_next = model(real_state, real_action)
```

**用途**: 检查单步拟合能力，不代表自由 rollout。

### Mode B: 固定动作序列开环

```python
# 先从真实系统生成固定动作序列
# 模型使用相同动作序列，状态从自身预测滚动
for step in range(n_steps):
    tau = fixed_actions[step]
    s_real = real_step(s_real, tau)
    s_model = model(s_model, tau)
```

**用途**: 隔离纯动力学模型误差。

### Mode C: 混合模式 (当前使用)

```python
# 控制输入由真实状态计算，模型状态从自身预测滚动
for step in range(n_steps):
    tau = tau_func(step, s_real)  # tau 来自真实状态
    s_real = real_step(s_real, tau)
    s_model = model(s_model, tau)  # 模型使用自己的预测
```

**代码路径**: `methods_evaluate.py:21-46`

**关键问题**: `tau_func` 中使用 `states['real']` 计算 tau，但模型使用自己的状态。

### Mode D: 完整闭环 (未实现)

```python
# 控制器完全使用模型预测状态
for step in range(n_steps):
    tau_real = controller(s_real)
    tau_model = controller(s_model)
    s_real = real_step(s_real, tau_real)
    s_model = model(s_model, tau_model)
```

**当前状态**: 未在现有代码中实现。

---

## 4. 现有结果对应的真实配置

### 4.1 报告结果

| 配置 | 报告值 (rad) | 来源 | 实际配置 |
|------|-------------|------|----------|
| GP + Ensemble(5) | 0.064 | REPORT_全面改进总结.md | 7模型+5轮DAgger |
| GP + Ensemble(7) + DAgger(5) | 0.066 | test_ensemble_optimization.py | 7模型+5轮DAgger |
| GP + Ensemble(5) + DAgger(3) | 0.15 | 标准配置 | 5模型+3轮DAgger |
| GP-B4 (稀疏GP) | 0.42 | gp_improvement/GP_IMPROVEMENT_REPORT.md | Nystroem 500组件 |

**关键发现**: 报告的 0.064 结果使用了优化配置 (7模型+5轮)，但报告中写的是 "集成(5模型)"。

### 4.2 快速审计结果 (run_fast_audit.py)

| 配置 | 步数 | MAE (rad) | 说明 |
|------|------|-----------|------|
| GP+Ensemble(5) 无DAgger | 200 | 0.67-1.01 | 基线配置 |
| GP+Ensemble(5) 2轮DAgger | 200 | 0.78-1.06 | DAgger效果有限 |

**环境**: Python 3.9.23, sklearn 1.6.1, numpy 1.23.5, GP样本数=2000, epoch=50

**证据**: CONFIRMED by `deep_handoff_audit/fast_audit_results.json`

### 4.3 统计审计结果冲突

**evaluation_results.json**:
- 所有 5 个 seed 给出完全相同的 phi_error (0.14764429611837954)
- **问题**: seed 不影响结果，可能是 bug

**statistical_audit_results.json**:
- 5 个 seed 的 MAE 范围: 0.000298-0.000449 rad
- **问题**: 数量级与 0.066/0.15 完全不同

**冲突**: 两个文件的 MAE 数量级不同，且与 0.066/0.15 结果完全不同。

**证据**: CONFIRMED by `deep_handoff_audit/evaluation_results.json` 和 `statistical_audit_results.json`

---

## 5. 已知报告冲突

| # | 报告声明 | 审计结果 | 冲突类型 | 严重程度 |
|---|----------|----------|----------|----------|
| 1 | "GP+Ensemble(5)=0.064" | 需要7模型+5轮配置 | 配置不一致 | 中 |
| 2 | "GP基线7.023 rad" | GP-B4=0.42更好 | 方法遗漏 | 低 |
| 3 | "DAgger减少分布偏移" | delta_dot偏移可达8σ | 部分正确 | 低 |
| 4 | "4D系统稳定" | 特征值+1.54 | 事实错误 | 高 |
| 5 | "0.064不可复现" | 0.066可复现(7+5配置) | 信息更新 | 低 |
| 6 | "集成方法最有效" | GP-B4更有效(纯GP) | 方法遗漏 | 低 |
| 7 | GP超参数优化 | lbfgs收敛警告 | 代码问题 | 中 |

**证据**: CONFIRMED by `deep_handoff_audit/REPORT_CONFLICT_MATRIX.md`

---

## 6. 数据文件、模型文件及代码版本

### 6.1 核心代码文件

| 文件 | 行数 | 说明 | 最后修改 |
|------|------|------|----------|
| methods_common.py | 113 | Meijaard参数、动力学、LQR、数据生成 | - |
| methods_classic.py | 208 | GP基线、PINN、参数辨识 | - |
| methods_nn.py | 190 | ResidualNet、SINDy+NN、NeuralODE | - |
| methods_evaluate.py | 110 | 评估函数(make_tau_func, run_trajectory) | - |
| test_ensemble.py | 289 | GP+NN集成方法+DAgger | - |
| test_ensemble_optimization.py | ~310 | 优化配置(7模型+5轮) | - |
| meijaard_dynamics.py | 259 | Meijaard 2007基准动力学 | - |
| bicycle_dynamics.py | 228 | 8D简化Whipple模型 | - |
| data_collector.py | ~370 | 8D数据收集 | - |
| world_model.py | ~345 | SINDy世界模型 | - |

### 6.2 数据文件

| 文件 | 维度 | 样本数 | 说明 |
|------|------|--------|------|
| meijaard_openloop_data.npz | 4D | 124,523 | 4D开环数据 |
| meijaard_openloop_data_v35.npz | 4D | 29,478 | v=3.5数据 |
| meijaard_openloop_data_v5.npz | 4D | 113,337 | v=5.0数据 |
| bicycle_data.npz | 8D | 468 | 8D原始数据 |
| bicycle_data_improved.npz | 8D | 2,083 | 8D改进数据 |
| sindy_model.npz | 8D | 55×8 | SINDy系数 |
| sindy_full_model.npz | 4D | 21×4 | 4D SINDy系数 |
| meijaard_sindy.npz | 4D | 21×4 | 4D SINDy (含标签) |
| meijaard_sindy_v35.npz | 4D | 21×4 | v=3.5 SINDy |
| meijaard_sindy_v5.npz | 4D | 21×4 | v=5.0 SINDy |

### 6.3 SINDy 4D 模型

| 文件 | 系数形状 | 说明 |
|------|----------|------|
| sindy_full_model.npz | (21, 4) | 4D SINDy (无标签) |
| meijaard_sindy.npz | (21, 4) | 4D SINDy (含标签, 默认v) |
| meijaard_sindy_v35.npz | (21, 4) | 4D SINDy (v=3.5) |
| meijaard_sindy_v5.npz | (21, 4) | 4D SINDy (v=5.0) |

**证据**: CONFIRMED by 文件检查

---

## 7. 当前不能证明的结论

| 结论 | 原因 | 置信度 |
|------|------|--------|
| 0.064 精确来源 | 无法确定原始代码版本 | UNKNOWN |
| 长期稳定性 (>500步) | 仅测试500步 | UNKNOWN |
| 实体自行车适用性 | 仅在仿真器上测试 | UNKNOWN |
| Mode D 闭环性能 | 未实现 Mode D | UNKNOWN |
| 全状态准确性 | 仅评估 phi | UNKNOWN |
| 多 seed 复现性 | evaluation_results.json seed 无效 | UNKNOWN |
| 0.066 在标准配置下复现 | 标准配置约 0.15 | UNKNOWN |

---

## 8. 下一阶段要验证的核心假设

### 8.1 必须验证的假设

| # | 假设 | 验证方法 | 优先级 |
|---|------|----------|--------|
| 1 | GP+NN+DAgger 单步预测准确 | Mode A 评估 | 高 |
| 2 | 固定动作序列多步 rollout 准确 | Mode B 评估 | 高 |
| 3 | 混合模式能跟随真实轨迹 | Mode C 评估 | 高 |
| 4 | 完整闭环系统仍然稳定 | Mode D 评估 | 最高 |
| 5 | 0.066 在固定配置下可复现 | 多 seed 统计 | 高 |
| 6 | DAgger 改善所有状态 | 全状态评估 | 中 |
| 7 | 不确定性与误差正相关 | 不确定性校准 | 中 |
| 8 | 模型适合多长预测时域 | 多时域评估 | 高 |

### 8.2 需要查明的冲突

| # | 冲突 | 查明方法 |
|---|------|----------|
| 1 | evaluation_results.json seed 完全相同 | 检查生成脚本 |
| 2 | statistical_audit_results.json MAE 数量级不同 | 检查指标定义 |
| 3 | 两个统计文件与 0.066/0.15 不可比 | 检查预测时域和模式 |

---

## 9. 环境信息

| 项目 | 值 |
|------|-----|
| Python | 3.9.23 (conda-forge) |
| NumPy | 1.23.5 |
| SciPy | 1.10.1 |
| scikit-learn | 1.6.1 |
| PyTorch | 2.7.1+cu128 |
| CUDA | True |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| 平台 | Windows 11 |

**证据**: CONFIRMED by 运行时检查

---

## 10. 关键代码路径索引

### 4D 路由

```
methods_common.py:14-29   → Meijaard参数
methods_common.py:31-42   → LQR控制器
methods_common.py:48-78   → 非线性动力学 (RK4, 5子步)
methods_common.py:84-112  → 训练数据生成
methods_classic.py:20-59  → GP基线
methods_nn.py:21-33       → ResidualNet架构
test_ensemble.py:33-141   → 集成训练+DAgger
test_ensemble.py:147-165  → 集成预测
test_ensemble.py:171-220  → 评估 (Mode C)
methods_evaluate.py:8-18  → make_tau_func
methods_evaluate.py:21-46 → run_trajectory (Mode C)
```

### 8D 路由

```
bicycle_dynamics.py:87-160 → 8D动力学
data_collector.py:1-370    → 数据收集
world_model.py:1-345       → SINDy世界模型
sindy_model.npz            → SINDy系数
```

---

## 11. 下一步行动

1. **建立统一评估配置**: 创建 `configs/reproducible_world_model_evaluation.yaml`
2. **实现四种评估模式**: 创建 `evaluate_world_models.py`
3. **查明统计文件冲突**: 生成 `reports/STATISTICAL_RESULT_PROVENANCE_AUDIT.md`
4. **数据版本审计**: 生成 `reports/DATA_PROVENANCE_AND_HASH_AUDIT.md`
5. **运行完整评估**: 30 个 seed, 1-1000 步时域

---

*文档生成时间: 2026-06-24*
*审计工具: Claude Code*
*项目: sindy_bicycle*
