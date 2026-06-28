# 不确定性与OOD检测报告

> **生成时间**: 2026-06-24
> **目的**: 评估集成模型的不确定性和分布外检测能力

---

## 1. 不确定性来源

### 1.1 GP 不确定性

GP 提供解析的预测方差：

```
Var[f(x*)] = k(x*, x*) - k(x*, X) K(X,X)^{-1} k(X, x*)
```

- 在训练数据密集区域：方差小
- 在训练数据稀疏区域：方差大
- **天然OOD检测能力**

### 1.2 集成不确定性

NN集成通过模型间分歧估计不确定性：

```
std_ensemble = std([model_1(x), model_2(x), ..., model_n(x)])
```

- 所有模型一致：不确定性低
- 模型分歧大：不确定性高
- **无需额外计算成本**

---

## 2. 当前实现

### 2.1 GP+NN集成不确定性

```python
# predict_with_uncertainty()
s_next_base = baseline.predict(s, tau)  # GP预测
mean_nn, std_nn = ensemble.predict(s_norm, a_norm)  # NN集成
s_next = s_next_base + mean_nn * delta_std * residual_scale
uncertainty = std_nn * delta_std * residual_scale  # 不确定性
```

### 2.2 不确定性指标

- **ensemble_disagreement**: 集成标准差
- **uncertainty_error_correlation**: 不确定性与实际误差的相关性
- **mean_uncertainty**: 平均不确定性 (逐状态)

---

## 3. OOD 检测策略

### 3.1 基于不确定性的OOD检测

```python
def is_ood(uncertainty, threshold=0.1):
    """如果不确定性超过阈值，判定为OOD。"""
    return np.any(uncertainty > threshold)
```

### 3.2 基于状态范围的OOD检测

```python
def is_state_ood(s, state_std, n_sigma=3):
    """如果状态超出训练范围的n倍标准差，判定为OOD。"""
    return np.any(np.abs(s) > n_sigma * state_std)
```

### 3.3 基于GP方差的OOD检测

```python
def is_gp_ood(gp, s, tau, threshold=0.5):
    """如果GP预测方差超过阈值，判定为OOD。"""
    _, std = gp.predict(s, tau, return_std=True)
    return np.any(std > threshold)
```

---

## 4. 预期结果 (待完整评估)

### 4.1 不确定性与误差相关性

| 模型 | phi相关性 | delta相关性 | phi_dot相关性 | delta_dot相关性 |
|------|----------|------------|--------------|----------------|
| gp_standard | 待测 | 待测 | 待测 | 待测 |
| gp_ensemble_5_3 | 待测 | 待测 | 待测 | 待测 |
| gp_ensemble_7_5 | 待测 | 待测 | 待测 | 待测 |

### 4.2 OOD检测准确率

| 阈值策略 | 精确率 | 召回率 | F1 |
|----------|--------|--------|-----|
| uncertainty > 0.1 | 待测 | 待测 | 待测 |
| uncertainty > 0.05 | 待测 | 待测 | 待测 |
| state > 3σ | 待测 | 待测 | 待测 |

---

## 5. 应用场景

1. **安全监控**: 当不确定性过高时，切换到安全控制器
2. **主动学习**: 在OOD区域收集新数据
3. **模型选择**: 选择不确定性最低的模型
4. **自适应预测时域**: 根据不确定性动态调整预测步数

---

## 6. 建议

1. **校准不确定性**: 确保不确定性与实际误差匹配
2. **多阈值策略**: 使用多个阈值进行分级OOD检测
3. **结合GP和集成**: 同时利用GP方差和集成分歧
4. **实时监控**: 在控制循环中实时检测OOD

---

*完整评估结果待更新*
*文档生成时间: 2026-06-24*
