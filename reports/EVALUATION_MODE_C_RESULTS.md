# 评估模式 C 结果: 混合模式

> **生成时间**: 2026-06-24
> **模式**: 混合模式 (tau 来自真实状态，模型自滚动)
> **评估范围**: 5 seeds × 5 horizons × 5 models = 25 次评估

---

## 1. 模式定义

```python
# 控制输入由真实状态计算，模型状态从自身预测滚动
for step in range(n_steps):
    tau = tau_func(step, s_real)  # tau 来自真实状态
    s_real = real_step(s_real, tau)
    s_model = model(s_model, tau)  # 模型使用自己的预测
```

**代码路径**: `methods_evaluate.py:21-46`

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

### 2.2 关键观察

1. **所有模型在Mode C下长期都发散**: 4D系统开环不稳定 (特征值实部 +1.54)
2. **GP标准模型短期最优**: 100步内 MAE < 0.002 rad
3. **SINDy在100步后发散**: 开环误差累积导致不稳定
4. **GP-B4精度差**: Nystroem近似导致预测误差大

---

## 3. Mode C 的特点

1. **控制器用真实状态**: tau_func 中使用 s_real 计算 LQR
2. **模型用自身预测**: 模型从 s_model 滚动
3. **混合反馈**: 真实系统提供力矩，模型提供状态
4. **误差单向累积**: 模型误差不会影响力矩，但会累积

### 3.1 与 Mode D 的关键差异

| 方面 | Mode C | Mode D |
|------|--------|--------|
| tau 来源 | 真实状态 | 模型状态 |
| 模型影响 | 不影响力矩 | 影响力矩 |
| 误差传播 | 单向累积 | 双向反馈 |
| GP标准 1000步 | 17.385 rad | 0.000 rad |

---

## 4. 历史结果对比

| 配置 | Mode C 500步 MAE | 来源 |
|------|------------------|------|
| GP only | ~7.023 rad | REPORT_全面改进总结 |
| GP standard (本次) | 12.149 rad | quick_with_gp_results.json |
| GP-B4 sparse (本次) | 20.971 rad | quick_with_gp_results.json |
| GP + Ensemble(5) + DAgger(3) | ~0.15 rad | test_ensemble.py |
| GP + Ensemble(7) + DAgger(5) | ~0.066 rad | test_ensemble_optimization.py |

**注意**: GP standard 本次结果 (12.15) 与历史结果 (~7.023) 有差异，可能原因:
- 不同的训练数据量 (本次5000 vs 历史可能更多)
- 不同的归一化方式
- 不同的评估配置

---

## 5. 为什么 Mode C 比 Mode D 更差？

在 Mode C 中:
1. 控制器用真实状态 s_real 计算力矩 tau
2. 模型用 tau 从自身状态 s_model 滚动
3. 但 tau 是基于 s_real 计算的，不是基于 s_model
4. 因此模型和真实系统"看到"不同的力矩
5. 误差快速累积

在 Mode D 中:
1. 控制器用模型状态 s_model 计算力矩 tau_model
2. 模型用 tau_model 从自身状态滚动
3. tau_model 和 s_model 一致
4. 因此模型和真实系统同步运行

---

## 6. 结论

1. **Mode C 不适合长期评估**: 所有模型在 Mode C 下长期都发散
2. **Mode D 更有意义**: 闭环评估更接近实际使用场景
3. **GP标准模型短期精确**: 100步内 MAE < 0.002 rad
4. **集成模型 (GP+NN+DAgger) 在Mode C下更优**: 历史数据显示 500步 MAE ~0.066-0.15 rad

---

*评估框架: evaluate/ (7个模块)*
*配置: configs/reproducible_world_model_evaluation.yaml*
*结果: results/quick_with_gp_results.json*
*文档生成时间: 2026-06-24*
