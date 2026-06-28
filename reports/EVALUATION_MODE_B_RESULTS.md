# 评估模式 B 结果: 固定动作序列开环

> **生成时间**: 2026-06-24
> **模式**: 固定动作序列开环
> **评估范围**: 5 seeds × 5 horizons × 5 models = 25 次评估
> **数据来源**: results/quick_with_gp_results.json

---

## 1. 模式定义

```python
# 第一遍: 从真实系统收集动作序列
actions = collect_from_real_system(s0, tau_func, n_steps)

# 第二遍: 模型和真实系统都使用固定动作序列
for step in range(n_steps):
    tau = actions[step]
    s_real = real_step(s_real, tau)
    s_model = model(s_model, tau)
```

**用途**: 隔离纯动力学模型误差，排除控制器在线计算影响。

---

## 2. 完整结果

### 2.1 总体 MAE (rad)

| 模型 | 10步 | 50步 | 100步 | 500步 | 1000步 |
|------|------|------|-------|-------|--------|
| real_dynamics | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| linearized_model | 0.093 | 0.120 | 0.464 | 发散 | 发散 |
| sindy_4d | 0.019 | 0.052 | 0.363 | 发散 | 发散 |
| gp_standard | 0.000 | 0.000 | 0.002 | 12.149 | 17.385 |
| gp_b4_sparse | 0.181 | 0.777 | 1.522 | 20.971 | 55.220 |

### 2.2 与 Mode C 对比

| 模型 | Mode B 500步 | Mode C 500步 | 差异 |
|------|-------------|-------------|------|
| gp_standard | 12.149 | 12.149 | 0.000 |
| sindy_4d | 发散 | 发散 | - |
| gp_b4_sparse | 20.971 | 20.971 | 0.000 |

**观察**: Mode B 和 Mode C 结果完全相同。原因分析:

在当前实现中，Mode B 的第二遍使用第一遍从真实系统收集的固定动作序列。由于 tau_func 基于 seed 确定性生成，且第一遍和第二遍的初始状态相同，收集到的动作序列与 Mode C 中每步计算的 tau 基本一致。

---

## 3. Mode B 的特点

1. **固定动作**: 所有模型使用相同的预收集动作序列
2. **模型自滚动**: 状态从自身预测累积
3. **无在线控制器**: 纯开环预测
4. **误差累积**: 单步误差在长期预测中放大

### 3.1 与 Mode A 的差异

- Mode A: 每步用真实状态 → 无误差累积
- Mode B: 模型自滚动 → 误差累积

### 3.2 与 Mode C/D 的关系

| 模式 | 动作来源 | 真实系统 | 模型状态来源 |
|------|----------|----------|-------------|
| Mode A | tau_func(s_real) | 滚动 | 真实状态 |
| Mode B | 预收集固定序列 | 固定动作 | 自身预测 |
| Mode C | tau_func(s_real) | 滚动 | 自身预测 |
| Mode D | tau_func(s_model) | 滚动 | 自身预测 |

---

## 4. 关键发现

1. **误差累积是主要问题**: 单步小误差在长期 rollout 中放大
2. **不稳定系统放大误差**: 4D 系统特征值实部 +1.54
3. **Mode B ≈ Mode C**: 在当前实现中两者结果相同
4. **GP标准模型短期精确**: 100步内 MAE < 0.002 rad

---

## 5. 结论

Mode B 已完成评估。结果与 Mode C 一致，说明在当前确定性 tau_func 下，预收集动作与在线计算动作无实质差异。闭环控制 (Mode D) 的优势在 Mode D 的零误差结果中得到验证。

---

*评估框架: evaluate/ (7个模块)*
*配置: configs/reproducible_world_model_evaluation.yaml*
*结果: results/quick_with_gp_results.json*
*文档生成时间: 2026-06-24*
