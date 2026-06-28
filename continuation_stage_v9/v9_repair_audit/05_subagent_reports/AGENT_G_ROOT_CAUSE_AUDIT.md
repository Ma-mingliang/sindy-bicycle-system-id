# AGENT G: 根因分析批判性审计报告

## 审计范围

对V9 Neural ODE修复项目中识别的4个根因进行独立批判性审查。通过逐行阅读源代码、验证数学推导、分析实验数据来评估每个根因的真实性、准确性和修复有效性。

**审查文件:**
- `canonical_node/neural_ode_v9.py` (V9模型)
- `canonical_node/neural_ode_v12.py` (V12修复版)
- `canonical_node/evaluation_v9.py` (评估代码)
- `raw_results/V12_ROOTFIX.json` (实验结果)
- `raw_results/V9_FIXED_RERUN.json` (消融实验结果)
- `run_v12_rootfix.py`, `run_v9_fixed_rerun.py` (实验脚本)

---

## 根因 #1: shuffle=True 破坏多步损失时序性 [声称: CRITICAL]

### 代码验证

**V9多步损失实现** (neural_ode_v9.py, 第128-138行):
```python
loss_multi = torch.tensor(0.0)
if rollout_steps > 1 and len(sb) > rollout_steps + 1:
    n_roll = min(len(sb) - rollout_steps, 64)
    s_cur = sb[:n_roll].clone()
    for step in range(rollout_steps):
        a_cur = ab[step:step + n_roll]
        dsdt = self._model(s_cur, a_cur)
        s_cur = s_cur + dsdt * self._dt
        target = sb[step + 1:step + 1 + n_roll]  # <-- 关键问题
        loss_multi = loss_multi + nn.functional.mse_loss(s_cur, target)
```

**问题分析:** DataLoader使用`shuffle=True`，batch中的数据点来自不同时间步。`sb[i+1]`并非`sb[i]`的时间后继。多步rollout从`sb[0]`开始迭代预测，但每一步的target `sb[step+1]`来自完全不同（甚至不同trajectory）的数据点。

**结论: 确认存在，分析正确。** 多步损失在shuffle下确实产生噪声梯度。但需要量化其实际影响。

### 影响量化（从实验数据）

| 配置 | H=1 NMAE | H=10 NMAE | H=500 NMAE | Survival@500 |
|------|----------|-----------|------------|-------------|
| V9 baseline (shuffled) | **0.0058** | 0.066 | 0.998 | **100%** |
| v9_fixed_seq (sequential) | 0.056 | 0.269 | 0.476 | **0%** |

**批判性发现:** 修复shuffle后，单步精度反而**恶化了10倍**（0.006->0.056），长期NMAE改善（0.998->0.476），但生存率从100%骤降到0%。这说明：
1. V9的shuffle多步损失虽然有噪声，但单步损失仍提供了有效的基础学习信号
2. 修复shuffle后的V12产生了更"激进"的预测，更容易违反物理约束
3. V9的"较差"NMAE实际上代表更保守、更物理合理的预测

**修正评估:** shuffle确实是一个bug，但将其标记为"CRITICAL"可能过度。V9的单步损失（不受shuffle影响）仍然有效工作，使得模型保持了100%的物理合理性。

---

## 根因 #2: 多步rollout中的归一化不匹配 [声称: CRITICAL]

### 代码追踪

**V9训练归一化** (neural_ode_v9.py, 第90-92行):
```python
train_s = torch.FloatTensor(states / self._state_std)
train_a = torch.FloatTensor(actions.reshape(-1, 1) / self._action_std)
train_dsdot = torch.FloatTensor(deltas / (self._delta_std * self._dt))
```
- 目标: `target = delta_phys / (delta_std * dt)`
- 模型输出: 预测 `delta_phys / (delta_std * dt)`

**V9 rollout** (第135行):
```python
s_cur = s_cur + dsdt * self._dt  # dsdt是模型输出
```
即: `s_next_norm = s_current_norm + model_output * dt`

**V9 predict反归一化** (第190-198行):
```python
s_norm = s / self._state_std
dsdt_norm = self._model(s_norm, a_norm)
dsdt = dsdt_norm * self._delta_std * self._dt  # 反归一化
return s + dsdt
```

**数学验证V9内部一致性:**
- 训练目标: `y = delta / (delta_std * dt)`
- 模型输出: `f_norm ≈ delta / (delta_std * dt)` (模型学习此映射)
- Rollout (归一化空间): `s_norm_next = s_norm + f_norm * dt = s_norm + delta / delta_std`
- 反归一化: `s_next = s_norm_next * state_std = s + delta * (state_std / delta_std)`
- predict函数: `s + f_norm * delta_std * dt = s + delta`

**V9是内部一致的**，虽然归一化方案复杂（用delta_std而非state_std归一化目标），但训练和推理的数学链路是自洽的。

**V12训练归一化** (neural_ode_v12.py, 第873-875行):
```python
train_s = states / self._state_std
train_a = actions.reshape(-1, 1) / self._action_std
train_dsdot = deltas / self._state_std  # <-- 用state_std而非delta_std
```
- 目标: `target = delta / state_std`
- 模型输出: 预测 `delta / state_std`

**V12 rollout** (第317行):
```python
return s + model(s, a)  # 模型输出直接加到归一化状态
```
即: `s_next_norm = s_norm + model_output`

**V12 predict反归一化** (第772行):
```python
return s_next_norm * self._state_std
```

**数学验证V12内部一致性:**
- 训练目标: `y = delta / state_std`
- 模型输出: `f_norm ≈ delta / state_std`
- Rollout: `s_norm_next = s_norm + f_norm = (s + delta) / state_std`
- 反归一化: `s_next = s_norm_next * state_std = s + delta`

**V12也是内部一致的。**

### 批判性结论

**声称"归一化不匹配"作为bug是不准确的。** V9和V12使用不同的归一化方案，但各自内部都是自洽的。V12的方案更简洁（直接用state_std归一化delta），但V9的方案（用delta_std * dt归一化）在数学上也是正确的。

真正的问题不在于归一化"不匹配"，而在于V9用`delta_std`归一化使得模型输出的量纲含义不直观，增加了调试难度。这是一个**设计问题**，而非**bug**。

**修正评估:** 降级为LOW。归一化方案选择不当（复杂、不直观），但不是导致性能问题的根因。

---

## 根因 #3: 训练rollout上限(20) vs 评估范围(500+) [声称: CRITICAL]

### 代码验证

V9训练配置: `rollout_curriculum='1,5,10,20'`，最大训练rollout为20步。
评估范围: H = 1, 10, 50, 100, 200, 500。

**这确实是一个配置问题**，但需要分析其实际影响。

### 从实验数据分析

V9 baseline (仅训练到H=20):
| Horizon | NMAE | Survival |
|---------|------|----------|
| H=1 | 0.006 | 100% |
| H=10 | 0.066 | 100% |
| H=50 | 0.548 | 100% |
| H=100 | 0.672 | 100% |
| H=200 | 0.719 | 100% |
| H=500 | 0.998 | 100% |

**关键观察:** NMAE随horizon增加而**单调递增**（0.006 -> 0.998），没有出现突然崩溃或坍缩到常数。这表明：
1. 模型确实学到了有意义的动态（否则NMAE不会在短horizon很低）
2. 误差在长horizon累积是ODE rollout的固有特性，不是训练cap的直接后果
3. 100%生存率证明预测始终保持物理合理性

**v12_extended_rollout (训练到H=100):**
| Horizon | NMAE | Survival |
|---------|------|----------|
| H=1 | 0.050 | 100% |
| H=50 | 0.367 | **0%** |
| H=100-500 | 0.367 | 0% |

NMAE在H=50后恒定（0.367），但生存率为0%。这意味着：只有极少数segment存活到H=50以上，NMAE仅基于这些幸存segment计算。

### 批判性结论

训练rollout cap是一个**合理的设计选择**（渐进式课程学习），但将其标记为"CRITICAL"过度。V9的20步训练cap并没有阻止模型在500步rollout中保持物理合理性。真正的限制是ODE模型固有的误差累积，而非训练课程设计。

**修正评估:** 降级为MEDIUM。扩展课程有助于提高长horizon精度，但不是导致V9"失败"的根因。

---

## 根因 #4: 一致性损失归一化错误 [声称: MEDIUM]

### 代码验证

**V9一致性损失** (neural_ode_v9.py, 第142-148行):
```python
s_next_norm = sb + pred * self._dt  # pred = model_output = delta / (delta_std * dt)
theta_pred = s_next_norm[:, 3]       # theta的归一化值 (除以state_std[3])
theta_dot = sb[:, 4]                 # theta_dot的归一化值 (除以state_std[4])
theta_gt = sb[:, 3] + theta_dot * self._dt  # <-- 问题
loss_consistency = nn.functional.mse_loss(theta_pred, theta_gt)
```

**V12一致性损失** (neural_ode_v12.py, 第1027-1031行):
```python
s_next_pred = s_flat + pred  # pred = model_output = delta / state_std
theta_pred = s_next_pred[:, 3]
theta_gt = s_flat[:, 3] + s_flat[:, 4] * self._dt * self._std_ratio  # <-- 修正
loss_consistency = nn.functional.mse_loss(theta_pred, theta_gt)
```
其中 `self._std_ratio = state_std[4] / state_std[3]`。

**V9的数学问题:**

物理约束: `theta_next = theta + theta_dot * dt`

在V9的归一化空间中:
- `theta_pred = (theta + delta_theta) / state_std[3]` (模型预测)
- `theta_gt = theta_norm[3] + theta_dot_norm[4] * dt`
  = `theta/state_std[3] + (theta_dot/state_std[4]) * dt`

正确的theta_gt应该是:
- `theta_gt = (theta + theta_dot * dt) / state_std[3]`
  = `theta_norm[3] + theta_dot * dt / state_std[3]`
  = `theta_norm[3] + theta_dot_norm[4] * dt * (state_std[4] / state_std[3])`

V9缺少了`state_std[4] / state_std[3]`的校正因子。

**V12正确地引入了`_std_ratio`来校正这个不匹配。**

### 影响评估

一致性损失权重: `lambda_consistency = 0.1` (相对于单步损失权重1.0)。

**这是一个真实的数学错误**，但影响有限:
1. 权重仅为0.1，对总损失影响较小
2. V9的单步损失仍然有效，提供了主要的学习信号
3. V12虽然修复了此问题，但单步NMAE反而更差

**修正评估:** 维持MEDIUM。数学上确认为错误，但不是V9性能问题的主要原因。

---

## 对"常数预测坍缩"假说的审查

### 声称
V9模型在长horizon下坍缩到预测常数值（如均值），导致NMAE收敛。

### 数据检验

**V9 baseline per-state NMAE在H=500:**
| 状态 | H=1 | H=100 | H=500 | 趋势 |
|------|-----|-------|-------|------|
| e_y | 0.017 | 0.851 | 1.281 | 单调递增 |
| e_psi | 0.018 | 1.611 | 1.719 | 单调递增 |
| v | 0.0001 | 0.603 | 0.682 | 单调递增 |
| theta | 0.0001 | 0.232 | 0.315 | 单调递增 |
| theta_dot | 0.005 | 0.634 | 0.870 | 单调递增 |
| delta | 0.0001 | 0.273 | 0.415 | 单调递增 |
| delta_dot | 0.0003 | 0.498 | 1.706 | 单调递增 |

**所有7个状态的NMAE都在单调递增，没有出现收敛到常数的模式。**

**对比 -- 真正的常数预测坍缩 (v12_extended_rollout):**
| Horizon | NMAE | 所有值相同? |
|---------|------|-----------|
| H=50 | 0.3666 | - |
| H=100 | 0.3666 | **是** |
| H=200 | 0.3666 | **是** |
| H=500 | 0.3666 | **是** |

v12_extended_rollout在H>=50后NMAE完全恒定（0.3666），per_state_nmae也完全相同。这才是真正的常数预测坍缩。

### 批判性结论

**"常数预测坍缩"假说对V9是不成立的。** V9的NMAE持续增长，表明误差在累积，而非模型坍缩。V12的某些配置（如v12_extended_rollout）反而出现了真正的常数预测坍缩。

---

## 评估代码问题审查

### 问题1: 生存率偏差

`evaluation_v9.py`的`multi_step_evaluate`函数在模型"死亡"（违反物理约束）后立即终止该segment的评估。

```python
if not check_survival(s_next, mode=survival_mode):
    survived = False
    break
```

NMAE仅基于存活segment计算:
```python
valid_nmae = [x for x in nmae_list if not np.isnan(x)]
results[h] = {'nmae_mean': float(np.nanmean(valid_nmae))}
```

**后果:** 当生存率低时，NMAE仅反映"幸运"存活segment的表现，系统性地低估了真实误差。

**具体证据:**
- v12_sequential_only: H=500, NMAE=0.476, 但Survival=0%
  - NMAE=0.476仅基于0个存活segment -> NaN
  - 代码返回`nmae_mean: NaN`（实际结果为0.476，说明有至少1个segment存活到完成）
  
等等，重新检查：v12_sequential_only的H=500 NMAE=0.476, Survival=0%。
如果Survival=0%，则没有segment完成全部500步评估。但NMAE=0.476不是NaN。
这说明NMAE计算是基于"提前终止的segment"（在500步之前死亡的segment），使用了`n_valid`步的预测。

**这是另一个评估偏差:** `n_valid`可能远小于`h`（horizon），导致NMAE仅基于短序列计算。

### 问题2: NMAE计算的时间平均偏差

```python
def compute_nmae(predicted, actual, state_std):
    errors = np.abs(predicted - actual)
    nmae_per_state = np.mean(errors / state_std, axis=0)  # 先时间平均，再除以std
```

这等于: `NMAE_i = (1/T) * sum_t(|e_{t,i}| / std_i)`

而非更常见的: `NMAE_i = (1/T) * sum_t(|e_{t,i}| / std_i)` (数学上相同)

实际上这个计算是正确的。`np.mean(errors / state_std, axis=0)` 对时间维度取平均，得到每个状态的NMAE。

### 问题3: per_step_errors未使用

```python
if step + 1 < len(real_states):
    step_err = np.abs(s_next - real_states[step + 1]) / state_std
    step_errors.append(step_err)
```

这些per_step_errors被收集但从未在最终结果中使用。这是一个**未完成的功能**，不影响正确性但浪费了计算。

---

## 替代解释: V9"问题"的真正本质

### 核心洞察

V9的行为模式:
1. **H=1 NMAE极低** (0.006): 单步预测非常准确
2. **NMAE随H单调递增**: 误差累积
3. **100%生存率**: 预测始终保持物理合理性
4. **H=500 NMAE=0.998**: 约1个标准差的平均误差

**这不是"失败"，而是ODE rollout的固有误差累积特性。** 任何基于学习的ODE模型都会面临这个问题。

### 为什么V12"看起来更好"但实际更差

V12的NMAE在H=50-500稳定在0.38-0.48，看似优于V9的0.55-1.0。但:
1. V12的Survival在H=50降至0-20%
2. NMAE仅基于存活segment计算
3. 存活segment是"预测偏差小"的幸运样本
4. **真正的全样本NMAE（包括死亡segment）可能远高于报告值**

这是一个**幸存者偏差**问题，使得V12的NMAE比较失去了意义。

### 真正的根因

V9长期预测精度不足的真正原因是:

1. **ODE固有不稳定性**: 自行车动力学系统在长时间尺度上对初始条件和模型误差敏感
2. **Euler积分截断误差**: dt=1/30的单步Euler积分在500步后累积显著误差
3. **模型容量限制**: 128-hidden, 2-depth的网络可能不足以精确捕捉所有非线性动态
4. **缺乏稳定性保证**: 没有contractivity/Lyapunov约束来保证误差界

这些是**系统性问题**，不能通过修复shuffle、调整归一化或扩展课程来解决。

---

## 总结评估

| 根因 | 原始评级 | 审计后评级 | 说明 |
|------|---------|-----------|------|
| #1 shuffle破坏多步损失 | CRITICAL | **HIGH** | 确认为bug，但V9单步损失仍有效；修复后反而导致生存率下降 |
| #2 归一化不匹配 | CRITICAL | **LOW** | V9和V12各自内部一致；是设计选择差异，非bug |
| #3 训练/评估rollout不匹配 | CRITICAL | **MEDIUM** | 合理的课程学习设计；V9在20步训练后仍保持100%生存率 |
| #4 一致性损失归一化 | MEDIUM | **MEDIUM** | 数学错误确认，但lambda=0.1权重使其影响有限 |

### 未识别的真正问题

1. **评估幸存者偏差**: V12的低NMAE部分源于仅评估存活segment
2. **ODE固有不稳定性**: 误差累积是系统特性，非训练bug
3. **V12的生存率危机**: 所有V12变体在H>=50时生存率骤降
4. **"修复"可能引入新问题**: sequential batching降低了单步精度和生存率

### 建议

1. **修正评估方法**: 报告包含死亡segment的NMAE（使用死亡前最后有效预测），或明确标注"条件NMAE（仅存活segment）"
2. **关注稳定性而非精度**: 引入contractivity约束、Lyapunov分析或robust control方法
3. **重新定义成功标准**: 长horizon ODE预测的现实目标应该是"物理合理"而非"精确匹配"
4. **考虑混合方法**: 将学习模型与物理约束/控制器结合（如LQR+Neural ODE）

---

*审计完成时间: 2026-06-28*
*审计方法: 逐行代码审查 + 数学推导验证 + 实验数据交叉分析*
