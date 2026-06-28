# DAgger 定向改进报告

> **生成时间**: 2026-06-24
> **目的**: 评估DAgger对各状态维度的定向改进效果

---

## 1. DAgger 原理

DAgger (Dataset Aggregation) 通过以下步骤减少分布偏移：

1. **初始训练**: 用离线数据训练基线模型
2. **在线收集**: 用模型自身预测运行轨迹，收集新数据
3. **数据聚合**: 将新数据加入训练集
4. **重新训练**: 用聚合数据重新训练模型
5. **重复**: 多轮迭代

---

## 2. 当前配置

### 2.1 标准配置 (5模型 + 3轮DAgger)

```python
n_models = 5
dagger_rounds = 3
residual_scale = 0.3
n_epochs = 100
# 每轮收集: 3段 × 500步 = 1500样本
```

### 2.2 优化配置 (7模型 + 5轮DAgger)

```python
n_models = 7
dagger_rounds = 5
residual_scale = 0.3
n_epochs = 100
use_scheduler = True  # CosineAnnealingLR
```

---

## 3. 已知结果

| 配置 | Mode C 500步 MAE | 来源 |
|------|------------------|------|
| GP only (无NN) | ~7.023 rad | REPORT_全面改进总结 |
| GP + Ensemble(5) 无DAgger | ~0.67-1.01 rad | fast_audit |
| GP + Ensemble(5) + DAgger(2) | ~0.78-1.06 rad | fast_audit |
| GP + Ensemble(5) + DAgger(3) | ~0.15 rad | test_ensemble.py |
| GP + Ensemble(7) + DAgger(5) | ~0.066 rad | test_ensemble_optimization.py |

**DAgger 效果**: 3轮DAgger将MAE从 ~1.0 降至 ~0.15 (6.7倍改进)

---

## 4. 逐状态分析 (待完整评估)

### 4.1 预期分布偏移

| 状态 | 训练范围 | 典型运行范围 | 偏移风险 |
|------|----------|-------------|----------|
| phi | [-0.5, 0.5] | [-0.25, 0.25] | 低 |
| delta | [-0.3, 0.3] | [-0.1, 0.1] | 低 |
| phi_dot | [-2.0, 2.0] | [-0.5, 0.5] | 中 |
| delta_dot | [-1.0, 1.0] | [-3, 3] | **高** |

**delta_dot** 的运行时范围可能超出训练范围，是最需要DAgger改进的状态。

### 4.2 DAgger 改进方向

1. **收集更多 delta_dot 极端值**: 在 DAgger 数据收集中，优先收集 delta_dot 大值的样本
2. **加权损失**: 对 delta_dot 的损失给予更高权重
3. **域随机化**: 在训练数据生成时扩大 delta_dot 范围

---

## 5. 建议

1. **增加DAgger轮次**: 从3轮增加到5轮可进一步降低MAE
2. **增加集成大小**: 7模型比5模型更稳定
3. **定向收集**: 在DAgger数据收集中关注 delta_dot 大值
4. **使用学习率调度**: CosineAnnealingLR 可改善训练稳定性

---

*完整评估结果待更新*
*文档生成时间: 2026-06-24*
