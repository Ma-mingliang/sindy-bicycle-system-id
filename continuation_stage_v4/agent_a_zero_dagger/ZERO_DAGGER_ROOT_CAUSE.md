# gp_ensemble_5_0 MAE=0 根因分析报告

> **状态**: CONFIRMED
> **日期**: 2026-06-25
> **严重性**: CRITICAL

---

## 1. 问题描述

`gp_ensemble_5_0`（5个NN模型，0轮DAgger）在 Mode A 和 Mode B 的所有评估步数中 MAE=0.000，明显异常。作为对比，`gp_standard` 在 500 步 Mode B 中 MAE=12.1。

## 2. 根因确认

### 2.1 训练循环永不执行 [CONFIRMED]

**位置**: `evaluate/eval_models.py`, GPEEnsemble.train(), Line 305

```python
self._ensemble_models = []          # Line 303: 初始化为空列表
for round_i in range(self.dagger_rounds):  # Line 305: range(0) 永远为空
    ...
    self._ensemble_models = models  # Line 341: 永远不会到达
```

当 `dagger_rounds=0` 时，`range(0)` 不产生任何迭代，训练循环体永不执行。`_ensemble_models` 保持为空列表 `[]`。

### 2.2 空集成返回 NaN [CONFIRMED]

**位置**: `evaluate/eval_models.py`, _predict_ensemble_mean(), Lines 381-392

```python
def _predict_ensemble_mean(self, models, s_norm, a_norm):
    preds = []
    for model in models:  # models = []，循环不执行
        ...
    return np.mean(preds, axis=0)  # np.mean([]) = nan
```

NumPy `np.mean([])` 对空数组返回 `nan`。

### 2.3 NaN 传播到 predict() [CONFIRMED]

**位置**: `evaluate/eval_models.py`, predict(), Lines 408-413

```python
def predict(self, s, tau):
    s_next_base = self._baseline.predict(s, tau)
    ...
    delta_nn = self._predict_ensemble_mean(...)  # 返回 nan
    return s_next_base + nan * delta_std * residual_scale  # 整个输出是 nan
```

### 2.4 NaN 被过滤，只剩初始状态 [CONFIRMED]

**位置**: `evaluate/eval_metrics.py`, _compute_overall_metrics(), Lines 60-81

```python
valid = ~np.any(np.isnan(abs_errors), axis=1)  # 所有 step>=1 被过滤
abs_err_valid = abs_errors[valid]               # 只剩 step 0
mae = float(np.mean(abs_err_valid))             # step 0 误差=0，MAE=0.0
```

## 3. 因果链

```
dagger_rounds=0
  -> range(0) 不执行
  -> _ensemble_models = []
  -> _predict_ensemble_mean 返回 np.mean([]) = nan
  -> predict() 返回 nan
  -> 误差 = nan
  -> valid mask 过滤所有 NaN 行
  -> 只剩 step 0（初始状态，误差=0）
  -> MAE = mean(0) = 0.0
```

## 4. 模式影响

| 模式 | 行为 | MAE 结果 |
|------|------|----------|
| A (teacher forcing) | NaN 被过滤，只剩 step 0 | **0.0 (假阳性)** |
| B (open loop) | NaN 被过滤，只剩 step 0 | **0.0 (假阳性)** |
| C (hybrid) | NaN 传播到 GP 基线 | sklearn 异常 |
| D (closed loop) | NaN 传播到 GP 基线 | sklearn 异常 |

## 5. 修复方案

在 `GPEEnsemble.train()` 中将 Line 305 从：

```python
for round_i in range(self.dagger_rounds):
```

修改为：

```python
for round_i in range(max(1, self.dagger_rounds)):
```

并保持 DAgger 数据收集守卫不变：

```python
if round_i < self.dagger_rounds - 1:
    new_inputs, new_residuals = self._collect_dagger_data(...)
```

此修复确保 NN 模型至少在初始残差数据上训练一次。

## 6. 附加问题

- 无验证检查 `_ensemble_models` 非空
- `_predict_ensemble_mean()` 和 `predict_with_uncertainty()` 无空模型保护
- 应在 `predict()` 中添加断言

---

*分析者: 零DAgger基线与集成训练修复子代理*
*验证状态: 全部 CONFIRMED*
