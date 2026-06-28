# DAgger Oracle 依赖与迁移分析

> **生成时间**: 2026-06-25

---

## 1. DAgger 实现分析

### 1.1 数据收集流程

```python
# evaluate/eval_models.py:352-379
for step in range(500):
    tau = tau_func(step, s_real)           # 动作来自真实状态
    s_real = real_step(s_real, tau)        # ← Oracle: 真实仿真器
    s_norm = s_base / state_std
    a_norm = tau / action_std
    delta_nn = predict_ensemble_mean(models, s_norm, a_norm)
    s_base = baseline.predict(s_base, tau) + delta_nn * delta_std * residual_scale

    # 残差标签
    real_delta = s_real - (s_base - delta_nn * delta_std * residual_scale)
    base_delta = baseline.predict(s_base_without_nn, tau) - s_base_without_nn
    residual = (real_delta - base_delta) / delta_std
```

### 1.2 Oracle 依赖点

| 代码行 | Oracle调用 | 说明 |
|--------|-----------|------|
| L365 | `s_real = real_step(s_real, tau)` | 仿真器前向传播 |
| L377 | `real_delta = s_real - ...` | 用真实状态计算残差标签 |

**结论**: DAgger 依赖真实仿真器 `real_step` 在模型预测状态附近查询真实下一状态。

---

## 2. Oracle 依赖程度

### 2.1 是否在模型预测状态上调用仿真器？

**否**。DAgger 使用的是 `s_real`（真实系统轨迹），不是 `s_model`（模型预测轨迹）。

具体流程:
1. 真实系统独立运行: `s_real = real_step(s_real, tau)`
2. 模型独立运行: `s_base = baseline.predict(s_base, tau) + nn_correction`
3. 残差标签: `residual = s_real - s_base`

**关键**: 仿真器只在真实轨迹上查询，不在模型预测的状态上查询。

### 2.2 是否属于仿真器代理模型学习？

**部分是**。DAgger 学习的是:
- 输入: 模型预测状态 `s_base` 和动作 `tau`
- 输出: 修正残差 `residual = s_real - s_base`
- 目标: 让模型预测逼近真实轨迹

这不是标准的"仿真器代理学习"（在任意状态查询仿真器），而是"轨迹跟踪学习"（在真实轨迹上学习修正）。

---

## 3. 迁移到真实自行车的可行性

### 3.1 可以迁移的部分

| 组件 | 可迁移性 | 说明 |
|------|----------|------|
| GP基线 | ✅ 可迁移 | 纯数据驱动，无Oracle依赖 |
| NN残差 | ⚠️ 部分可迁移 | 需要真实轨迹数据 |
| DAgger数据收集 | ❌ 需要替代方案 | 依赖仿真器 |

### 3.2 真实系统中的替代方案

| 方案 | 可行性 | 说明 |
|------|--------|------|
| 真实轨迹数据 | ✅ | 在真实自行车上收集轨迹，用GP+NN预测vs真实计算残差 |
| 安全探索 | ⚠️ | 在安全范围内让自行车运行，收集修正数据 |
| 模型集成 | ✅ | 用多个模型的分歧估计不确定性，减少对Oracle的依赖 |
| 离线RL | ⚠️ | 用离线数据集训练，不需要在线仿真器查询 |

### 3.3 推荐迁移策略

1. **阶段1**: 在仿真器中训练GP基线 (无Oracle依赖)
2. **阶段2**: 在真实自行车上收集少量轨迹数据
3. **阶段3**: 用真实轨迹数据训练NN残差 (替代DAgger)
4. **阶段4**: 在线微调 (安全探索 + 增量学习)

---

## 4. 结论

1. **DAgger 依赖仿真器**: 但只在真实轨迹上查询，不在任意状态查询
2. **不是标准仿真器代理学习**: 而是轨迹跟踪学习
3. **可以迁移到真实系统**: 需要用真实轨迹数据替代仿真器
4. **GP基线可直接迁移**: 无Oracle依赖
5. **NN残差需要真实数据**: 但数据量要求不大 (~1500样本/轮)

---

*分析时间: 2026-06-25*
