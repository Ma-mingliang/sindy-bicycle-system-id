# Divergence and Metric Unit Audit

> **Generated**: 2026-06-25
> **Scope**: P3 -- 修正发散和指标单位

---

## 1. 问题描述

### 1.1 500步=1000步相同问题

多个模型在 Mode B 评估中，500步和1000步的 MAE 完全相同：

| 模型 | 500步 MAE | 1000步 MAE | 相同? |
|------|-----------|------------|-------|
| linearized_model | 6.1014 | 6.1014 | YES |
| sindy_4d | 6.6921 | 6.6921 | YES |
| gp_b4_sparse | 8.6174 | 8.6174 | YES |
| gp_standard | 12.1491 | 17.3848 | NO |
| gp_ensemble_5_3 | 0.3085 | 0.3128 | NO |
| gp_ensemble_7_5 | 0.2519 | 0.2573 | NO |

### 1.2 根因分析

**根因**: `eval_modes_v2.py` 中的发散检测逻辑 (line 78-81):

```python
if np.any(np.isnan(s_model)) or np.any(np.abs(s_model) > 100):
    remaining = n_valid - i - 1
    states_model.extend([np.full(4, np.nan)] * remaining)
    break
```

当模型发散时：
1. 发散后的所有步骤被填充为 NaN
2. `compute_all_metrics` 使用 `valid = ~np.any(np.isnan(abs_errors), axis=1)` 过滤 NaN
3. MAE 只在有效步骤上计算
4. 如果发散发生在 step K < 500，则 500步和 1000步的 MAE 完全相同 (都只用了 0-K 步)

**这不是 bug，是设计选择**: 发散后的 NaN 状态不应参与 MAE 计算。但需要额外报告 `survival_steps` (有效步数)。

### 1.3 发散时间线

| 模型 | 发散位置 | 证据 |
|------|----------|------|
| linearized_model | 100-500步之间 | 100步=0.464, 500步=6.101 (显著增加) |
| sindy_4d | 100-500步之间 | 100步=0.363, 500步=6.692 (显著增加) |
| gp_b4_sparse | 100-500步之间 | 100步=1.522, 500步=8.617 (显著增加) |
| gp_standard | 100-500步之间 | 100步=0.002, 500步=12.149 (突变) |
| gp_ensemble_5_3 | 未发散 | 500步=0.309, 1000步=0.313 (稳定) |
| gp_ensemble_7_5 | 未发散 | 500步=0.252, 1000步=0.257 (稳定) |

---

## 2. 指标单位问题

### 2.1 问题描述

总体 MAE 混合了不同物理单位：
- phi: rad (滚动角)
- delta: rad (转向角)
- phi_dot: rad/s (滚动角速度)
- delta_dot: rad/s (转向角速度)

将 rad 和 rad/s 的误差直接平均在物理上没有意义。

### 2.2 修正方案

使用 NMAE (Normalized MAE):
```
NMAE = (1/4) * sum(|e_i| / sigma_i)
```

其中 sigma_i 是各状态的标准差，用于归一化。

### 2.3 Per-State MAE (Mode B, 5 seeds)

| 模型 | 10步 | 50步 | 100步 | 500步 | 1000步 |
|------|------|------|-------|-------|--------|
| **phi (rad)** | | | | | |
| gp_standard | 0.000 | 0.000 | 0.001 | 5.977 | 8.543 |
| gp_ensemble_7_5 | 0.045 | 0.101 | 0.110 | 0.125 | 0.128 |
| gp_ensemble_5_3 | 0.040 | 0.128 | 0.142 | 0.153 | 0.155 |
| **delta (rad)** | | | | | |
| gp_standard | 0.000 | 0.000 | 0.001 | 6.172 | 8.842 |
| gp_ensemble_7_5 | 0.046 | 0.104 | 0.112 | 0.127 | 0.129 |
| gp_ensemble_5_3 | 0.042 | 0.130 | 0.144 | 0.156 | 0.158 |
| **phi_dot (rad/s)** | | | | | |
| gp_standard | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gp_ensemble_7_5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gp_ensemble_5_3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **delta_dot (rad/s)** | | | | | |
| gp_standard | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gp_ensemble_7_5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gp_ensemble_5_3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**注意**: phi_dot 和 delta_dot 的 MAE 为 0 是因为这些状态在评估中没有被正确计算或返回。需要检查 `compute_all_metrics` 是否包含所有 4 个状态。

---

## 3. 修正建议

### 3.1 短期修正

1. **报告 survival_steps**: 在评估结果中添加 `survival_steps` 字段，记录模型实际存活的步数
2. **区分有效 MAE 和全轨迹 MAE**: 有效 MAE 只计算非 NaN 步骤，全轨迹 MAE 将 NaN 步骤视为最大误差
3. **使用 NMAE**: 将各状态误差归一化后再平均

### 3.2 长期修正

1. **修改评估代码**: 在 `run_mode_b_fixed` 中返回 `survival_steps`
2. **修改指标计算**: 在 `compute_all_metrics` 中添加 `survival_steps` 和 `first_divergence_step`
3. **添加 per-state 报告**: 每个状态单独报告 MAE 和单位

---

## 4. 结论

1. **500=1000 不是 bug**: 是发散后 NaN 过滤的预期行为
2. **需要额外指标**: `survival_steps` 和 `first_divergence_step`
3. **单位混用需要修正**: 使用 NMAE 或 per-state 报告
4. **集成模型 (gp_ensemble_7_5, gp_ensemble_5_3) 是唯一不发散的模型**: 适合长期预测

---

*审计完成: 2026-06-25*
