# Historical 0.064/0.066 Result Code Trace

> **Generated**: 2026-06-25
> **Scope**: P4 -- 追踪历史 0.066 结果来源

---

## 1. 问题描述

项目中存在三个不同的性能指标：
- 0.064 (某次运行)
- 0.066 (另一某次运行)
- 0.252 (评估v2 Mode B 100步)

需要确认这三个数字的来源和可比性。

---

## 2. 代码追踪

### 2.1 0.066 来源

**文件**: `test_ensemble_optimization.py`

**评估模式**: Mode C (混合模式)
- tau 来自真实状态: `tau = tau_func(i, s_real)`
- 模型自滚动: `s_model = model.predict(s_model, tau)`

**指标**: phi-only endpoint error at step 500
```python
# Line ~260-270
endpoint_error = abs(states_model[-1][0] - states_real[-1][0])  # phi only
```

**配置**:
- 7 NN models + 5 DAgger rounds + residual_scale=0.3
- 等效于 gp_ensemble_7_5

**初始状态**:
- `phi_init` 未设置随机种子 (unseeded)
- 每次运行产生不同的初始状态

### 2.2 0.064 来源

与 0.066 相同的代码路径，只是随机种子不同导致的初始状态差异。

### 2.3 0.252 来源

**文件**: `continuation_stage/evaluate/run_full_evaluation_v2.py`

**评估模式**: Mode B (固定动作开环)
- 动作序列从参考轨迹冻结
- 模型和真实系统使用相同固定动作

**指标**: 全轨迹 MAE (所有4个状态)
```python
mae = mean(|s_model - s_real|)  # over all 4 states, all steps
```

**配置**:
- gp_ensemble_7_5 (同上)
- 5 seeds, 100步

---

## 3. 差异分析

| 属性 | 0.066 | 0.252 |
|------|-------|-------|
| 评估模式 | Mode C (混合) | Mode B (固定动作) |
| 指标 | phi-only endpoint error | 全轨迹 MAE (4 states) |
| 步数 | 500 | 100 |
| 初始状态 | 未固定 (unseeded) | 固定 (seed 42-46) |
| 可重现性 | 否 | 是 |

**结论**: 0.066 和 0.252 不可直接比较：
1. 不同评估模式 (Mode C vs Mode B)
2. 不同指标 (phi-only endpoint vs 全轨迹 MAE)
3. 不同步数 (500 vs 100)
4. 不同初始状态 (unseeded vs seeded)

---

## 4. 修正后的比较

| 指标 | gp_ensemble_7_5 | 说明 |
|------|-----------------|------|
| Mode B 100步 MAE | 0.222 ± 0.108 | 评估v2, 5 seeds |
| Mode B 500步 MAE | 0.252 ± 0.117 | 评估v2, 5 seeds |
| Mode B 1000步 MAE | 0.257 ± 0.122 | 评估v2, 5 seeds |
| Mode C 500步 phi-only | ~0.066 | 历史, unseeded |
| Mode D 1000步 MAE | 0.074 ± 0.003 | 评估v2, 5 seeds |

**关键发现**:
- Mode C 的 phi-only endpoint error (0.066) 远小于 Mode B 的全轨迹 MAE (0.252)
- 这是因为 Mode C 的 tau 来自真实状态，模型误差不会通过动作放大
- Mode D 的 MAE (0.074) 也小于 Mode B，因为 LQR 控制器补偿了模型误差

---

## 5. 建议

1. **统一评估标准**: 使用 Mode B 作为世界模型评估的主要依据
2. **固定随机种子**: 所有评估必须使用固定种子以确保可重现性
3. **区分指标**: endpoint error 和全轨迹 MAE 是不同的指标，不可混用
4. **记录完整配置**: 包括评估模式、指标定义、步数、种子数

---

*审计完成: 2026-06-25*
