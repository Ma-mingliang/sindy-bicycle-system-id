# 统计结果来源审计

> **生成时间**: 2026-06-24
> **目的**: 查明 evaluation_results.json 和 statistical_audit_results.json 的来源差异

---

## 1. 问题描述

项目中存在多个统计结果文件，MAE 数量级完全不同：

| 文件 | MAE 范围 | 声称含义 |
|------|----------|----------|
| evaluation_results.json | 0.147644 (所有seed相同) | "final_phi_error" |
| statistical_audit_results.json | 0.000298-0.000449 | "mae_rad" |
| test_ensemble.py 输出 | 0.15-0.25 | 500步 Mode C MAE |
| test_ensemble_optimization.py 输出 | 0.066 | 500步 Mode C MAE (优化配置) |

---

## 2. 来源追踪

### 2.1 evaluation_results.json

**生成脚本**: `deep_handoff_audit/evaluation_mode_comparison.py:151-190`

**关键代码**:
```python
def statistical_audit_fixed_seeds(n_runs=5):
    for run_i in range(n_runs):
        np.random.seed(42 + run_i)
        for seg_i in range(5):
            tau_func = make_tau_func(seg_i, 500)
            phi_init = np.random.uniform(-0.25, 0.25)
            s0 = np.array([phi_init, 0.0, 0.0, 0.0])
            s = s0.copy()
            for step in range(500):
                tau = tau_func(step, s)
                s = real_step(s, tau)
                if abs(s[0]) > math.pi / 3:
                    break
            errors.append(abs(s[0]))  # ← 这是最终横滚角，不是预测误差！
```

**实际测量**: `abs(s[0])` — 真实系统的最终横滚角（稳定性指标）

**不是**: 模型预测误差

**为什么所有 seed 相同**: `make_tau_func(seg_i, 500)` 使用 `np.random.RandomState(42 + seg_i)`，不受全局 seed 影响。`np.random.uniform(-0.25, 0.25)` 的结果被 `make_tau_func` 内部的随机调用重置。

**结论**: 这个文件测量的是 LQR 控制器的稳定性，不是模型预测能力。

---

### 2.2 statistical_audit_results.json

**生成脚本**: `deep_handoff_audit/statistical_audit_064.py:12-91`

**关键代码**:
```python
def run_statistical_audit():
    for run_i in range(n_runs):
        for seg_i in range(n_segments):
            s = s0.copy()
            for step in range(n_steps):
                tau = tau_func(step, s)
                s_real = real_step(s, tau)
                err = abs(s_real[0] - s[0])  # ← 这是单步状态变化量！
                phi_errors.append(err)
                s = s_real
```

**实际测量**: `abs(s_real[0] - s[0])` — 真实动力学的单步状态变化量

**不是**: 模型预测误差

**为什么值很小 (~0.0003 rad)**: 这是 LQR 控制下真实系统每步的横滚角变化量，不是预测误差。

**结论**: 这个文件测量的是真实动力学的单步变化幅度，不是模型预测能力。

---

### 2.3 0.066 结果

**生成脚本**: `test_ensemble_optimization.py`

**实际测量**: GP+NN+DAgger 模型在 Mode C 下 500 步的预测误差

**公式**: `abs(s_ensemble[0] - s_real[0])` 在第 500 步

**配置**: 7 模型 + 5 轮 DAgger + residual_scale=0.3

**结论**: 这是真正的模型预测误差。

---

### 2.4 0.15 结果

**生成脚本**: `test_ensemble.py`

**实际测量**: GP+NN+DAgger 模型在 Mode C 下 500 步的预测误差

**配置**: 5 模型 + 3 轮 DAgger + residual_scale=0.3

**结论**: 这是真正的模型预测误差。

---

## 3. 冲突原因总结

| 文件 | 实际测量 | 数量级 | 是否可比 |
|------|----------|--------|----------|
| evaluation_results.json | 最终横滚角 (稳定性) | 0.15 rad | 否 |
| statistical_audit_results.json | 单步状态变化量 | 0.0003 rad | 否 |
| test_ensemble.py 输出 | 500步预测误差 | 0.15 rad | 是 (标准配置) |
| test_ensemble_optimization.py 输出 | 500步预测误差 | 0.066 rad | 是 (优化配置) |

**核心问题**: 两个审计文件都不是在测量模型预测误差。它们测量的是完全不同的物理量。

---

## 4. 建议

1. **不要将 evaluation_results.json 和 statistical_audit_results.json 与 0.066/0.15 结果比较** — 它们测量的是不同的东西
2. **重新运行统计审计**，使用实际 GP+NN+DAgger 模型，测量真正的预测误差
3. **统一指标定义**: 所有评估必须使用相同的指标 (Mode C 500步预测误差)

---

## 5. 指标定义标准化

为避免未来混淆，统一定义：

| 指标 | 定义 | 公式 |
|------|------|------|
| 单步预测误差 | 模型单步预测 vs 真实下一状态 | `abs(model_predict(s, tau) - real_step(s, tau))` |
| 多步预测误差 (Mode C) | 模型多步 rollout vs 真实轨迹 | `abs(s_model[t] - s_real[t])` 在指定步数 |
| 多步预测误差 (Mode D) | 模型闭环 rollout vs 真实闭环 | `abs(s_model[t] - s_real[t])` 在指定步数 |
| 最终状态误差 | 最后一步的状态误差 | `abs(s_model[-1] - s_real[-1])` |
| 稳定性指标 | 最终横滚角绝对值 | `abs(s_real[-1][0])` |
| 单步变化量 | 真实动力学的单步状态变化 | `abs(s_real[t+1][0] - s_real[t][0])` |

---

*文档生成时间: 2026-06-24*
*审计工具: Claude Code*
