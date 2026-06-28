# Agent A: V9 Neural ODE Code Audit Report

**Date**: 2026-06-28
**Scope**: V9 Neural ODE training, evaluation, data loading, and configuration code
**Goal**: Identify issues that could degrade long-horizon prediction performance

---

## Files Read

| File | Lines | Role |
|------|-------|------|
| `continuation_stage_v9/canonical_node/neural_ode_v9.py` | 241 | Neural ODE model definition & training |
| `continuation_stage_v9/run_v9_experiments.py` | 914 | Main experiment orchestration script |
| `continuation_stage_v9/canonical_node/evaluation_v9.py` | 130 | Multi-step rollout evaluation & NMAE |
| `continuation_stage_v9/canonical_node/data_loader_v9.py` | 19 | Data loading (delegates to V8) |
| `continuation_stage_v9/canonical_node/config_v9.py` | 43 | Configuration & hyperparameters |
| `continuation_stage_v8/canonical_7d/data_loader.py` | 141 | V8 data loading (upstream) |
| `continuation_stage_v8/canonical_7d/config.py` | 84 | V8 configuration (upstream) |
| `continuation_stage_v8/canonical_7d/extended_models.py` | 286 | V8 Neural ODE baseline |

---

## Issues Found (by severity)

### CRITICAL-1: Multi-Step Rollout Loss Uses Shuffled Data -- Loss is Meaningless

**File**: `neural_ode_v9.py`, lines 96-97 and 128-138

**Phenomenon**: The DataLoader is created with `shuffle=True` (line 97). The multi-step rollout loss (lines 128-138) takes `sb[:n_roll]` as the starting states and then iterates `s_cur = s_cur + dsdt * dt`, comparing against `sb[step+1:step+1+n_roll]`. However, because the data is shuffled, consecutive indices in a batch do NOT correspond to temporally consecutive timesteps. `sb[0]` and `sb[1]` are from completely different trajectories/timesteps.

**Root Cause**: The multi-step rollout requires contiguous trajectory segments, but the DataLoader shuffles all (state, action, delta) tuples independently. There is no mechanism to sample contiguous subsequences.

**Impact on Long-Horizon Prediction**: **Severe**. The multi-step loss, which is supposed to train the model for accurate multi-step rollout (the key to long-horizon performance), is effectively computing MSE between random states. It provides no useful gradient signal for multi-step accuracy. Worse, the `lambda_multi=0.3` weight means this noise contributes 30% to the total loss, actively degrading single-step training. The curriculum schedule (rollout_steps ramping from 1 to 20) is entirely wasted.

**Evidence**:
```python
# Line 97: shuffle destroys temporal contiguity
loader = torch.utils.data.DataLoader(ds, batch_size=self.config.batch_size, shuffle=True)

# Lines 131-137: assumes sb[i] and sb[i+1] are consecutive timesteps
s_cur = sb[:n_roll].clone()
for step in range(rollout_steps):
    a_cur = ab[step:step + n_roll]  # random actions, not contiguous
    dsdt = self._model(s_cur, a_cur)
    s_cur = s_cur + dsdt * self._dt
    target = sb[step + 1:step + 1 + n_roll]  # random targets
    loss_multi = loss_multi + nn.functional.mse_loss(s_cur, target)
```

**Suggested Fix**: Either (a) create a custom sampler that yields contiguous trajectory windows of length `rollout_steps+1`, or (b) pre-group data into episodes and sample rollout windows from within episodes before batching. The simplest fix: disable shuffle and restructure data as contiguous trajectories, or build a separate dataset for multi-step that explicitly stores (s_t, a_t, s_{t+1}, ..., s_{t+H}) tuples.

**Confidence**: 99%. This is definitively a bug. The shuffle + contiguous-indexing assumption is incompatible.

---

### CRITICAL-2: Consistency Loss is Dimensionally Incorrect in Normalized Space

**File**: `neural_ode_v9.py`, lines 141-148

**Phenomenon**: The consistency loss enforces `theta_next_norm = theta_norm + theta_dot_norm * dt`. This equation is only valid in physical (unnormalized) space: `theta_next = theta + theta_dot * dt`. In normalized space, the correct relationship depends on the ratio of `std_theta` to `std_theta_dot`.

**Root Cause**: The code operates in normalized space (states divided by `state_std`), but applies the kinematic equation as if normalization preserves the linear relationship. Specifically:

```python
# Line 145-148: all quantities are in normalized space
theta_pred = s_next_norm[:, 3]          # theta_next / std_theta
theta_dot = sb[:, 4]                     # theta_dot / std_theta_dot
theta_gt = sb[:, 3] + theta_dot * self._dt  # theta/std_theta + (theta_dot/std_theta_dot)*dt
```

The correct relationship in normalized space is:
```
theta_next_norm = theta_norm + (theta_dot_physical * dt) / std_theta
                = theta_norm + (theta_dot_norm * std_theta_dot * dt) / std_theta
```

So `theta_gt` should be `sb[:, 3] + sb[:, 4] * self._dt * (delta_std[4] / state_std[3])` (using appropriate std ratios), or the comparison should be done in physical space.

**Impact**: The consistency loss applies incorrect gradients to the theta dimension. If `std_theta` and `std_theta_dot` are very different (which they typically are -- angle vs angular velocity), the loss pushes theta predictions in the wrong direction or with wrong magnitude. This could destabilize theta prediction, which is critical for heading accuracy in long rollouts.

**Suggested Fix**: Compute the consistency loss in physical space:
```python
s_next_physical = sb * self._state_std + pred * self._delta_std * self._dt
theta_pred_phys = s_next_physical[:, 3]
theta_gt_phys = sb[:, 3] * self._state_std[3] + sb[:, 4] * self._state_std[4] * self._dt
loss_consistency = mse(theta_pred_phys / self._state_std[3], theta_gt_phys / self._state_std[3])
```

**Confidence**: 95%. The dimensional analysis is clear. The only uncertainty is whether the std ratio happens to be close to 1.0 by coincidence, which would make the error small.

---

### HIGH-1: Phase 3F Planning Evaluation is Circular (Self-Consistency, Not Ground Truth)

**File**: `run_v9_experiments.py`, lines 634-656

**Phenomenon**: The planning evaluation computes "true costs for candidates" using `base_model.predict()` -- the same Neural ODE model being evaluated. This means the evaluation measures how well the model's ranked predictions agree with the model's own rollout, not with actual physics.

**Root Cause**: Ground-truth future states from real data are only available for the single action sequence that was actually executed. For arbitrary candidate action sequences, there is no ground truth. The code uses the model itself as a proxy for "true dynamics."

**Impact**: The Spearman correlation, top-3 overlap, and regret metrics are all measuring self-consistency. A model with systematic biases will show perfect self-consistency (high Spearman, low regret) while being completely wrong in absolute terms. This makes the planning evaluation unreliable for assessing real-world planning quality.

**Suggested Fix**: Acknowledge this limitation explicitly in results. For a meaningful evaluation, either (a) use a high-fidelity simulator as ground truth, or (b) evaluate planning on the single executed trajectory only (comparing predicted vs actual cost along the realized action sequence).

**Confidence**: 99%. This is a design limitation, not a bug per se, but it invalidates the planning metrics as indicators of real planning quality.

---

### HIGH-2: Euler Integration with dt=1/30 Accumulates Error Over Long Horizons

**File**: `neural_ode_v9.py`, line 135 (`s_cur = s_cur + dsdt * self._dt`) and line 198 (`return s + dsdt`)

**Phenomenon**: Both training rollout and inference use simple Euler integration: `s_{t+1} = s_t + f(s_t, a_t) * dt`. With `dt = 1/30`, each step introduces O(dt^2) local truncation error, accumulating to O(dt) global error over the trajectory.

**Root Cause**: The model is called "Neural ODE" but uses the simplest possible integrator. The V8 baseline also uses Euler (despite the docstring claiming RK4). For 1000-step rollouts, the cumulative integration error could be significant, especially for fast-changing states like `theta_dot` and `delta_dot`.

**Impact**: Integration error compounds over long horizons. For H=1000 steps, the accumulated integration error could dominate the model prediction error, making it impossible to distinguish whether poor long-horizon performance is due to model inaccuracy or integration error. The Jacobian regularization (which penalizes large derivatives) partially compensates by encouraging smooth dynamics, but does not fix the fundamental integrator limitation.

**Suggested Fix**: Use at least RK2 (midpoint method) or RK4 for inference rollouts. The training loss can remain Euler-based (it's differentiable and cheap), but evaluation should use a higher-order integrator. RK4 would reduce integration error from O(dt) to O(dt^4) over the full trajectory.

**Confidence**: 85%. The actual impact depends on the dynamics -- if f(s,a) is smooth and small, Euler may be adequate. But for a bicycle with oscillatory modes (theta_dot, delta_dot), this is likely significant at H=500+.

---

### HIGH-3: No Validation-Based Early Stopping or Model Selection

**File**: `neural_ode_v9.py`, lines 74-75 (val_states/val_actions/val_deltas accepted but never used)

**Phenomenon**: The `train()` method accepts validation data parameters (`val_states`, `val_actions`, `val_deltas`) but never uses them. Training runs for a fixed 200 epochs with no early stopping, no validation loss monitoring, and no model selection based on generalization.

**Root Cause**: The validation parameters are accepted in the signature but the training loop has no validation logic.

**Impact**: The model may overfit to training data over 200 epochs, especially with the small network (128 hidden, 2 layers) and the potentially noisy multi-step loss. Without validation monitoring, there is no way to detect or prevent overfitting. The cosine annealing schedule helps somewhat by reducing LR, but does not replace early stopping.

**Suggested Fix**: Add validation loss computation every N epochs, track the best validation checkpoint, and return the best model rather than the last model.

**Confidence**: 90%. Overfitting risk is real but severity depends on dataset size and model capacity.

---

### MEDIUM-1: NMAE Surviving-Trajectory Bias

**File**: `evaluation_v9.py`, lines 99-107

**Phenomenon**: When a trajectory dies (survival=False), the NMAE for that segment is computed only over the surviving prefix. Dead trajectories contribute partial NMAE values that are typically lower (since errors grow with horizon). The survival flag at line 107 requires `survived and n_valid >= n - 1`, meaning partially surviving segments still contribute their (low) partial NMAE to the mean.

**Root Cause**: The NMAE averaging does not distinguish between "completed all H steps with low error" and "died at step 10 with low error on those 10 steps."

**Impact**: The reported NMAE at long horizons (H=500, H=1000) may be artificially low because dead trajectories contribute short-horizon NMAE values. A segment that dies at step 50 with NMAE=0.2 will lower the H=500 NMAE average, masking the fact that the model cannot actually survive to H=500.

**Suggested Fix**: Either (a) only include segments that survive the full horizon in the NMAE calculation, or (b) report NMAE conditioned on survival, or (c) pad dead trajectories with the last valid state and compute NMAE over the full horizon (which would inflate NMAE for dead trajectories).

**Confidence**: 85%. This is a reporting issue that could mislead model selection.

---

### MEDIUM-2: Jacobian Regularization is Computationally Expensive and Potentially Destabilizing

**File**: `neural_ode_v9.py`, lines 152-163

**Phenomenon**: The Jacobian regularization computes the full Jacobian norm via `torch.autograd.grad` with `create_graph=True`, enabling second-order gradients. This is applied to a mini-batch of up to 32 samples, computing 7 gradient calls (one per state dimension) per batch.

**Root Cause**: The intent is to penalize large Jacobian norms for stability, but the implementation creates a computational graph through the Jacobian computation, which is expensive and can cause gradient instability.

**Impact**: (a) Training speed is significantly reduced (each Jacobian computation requires backprop through the entire network for each of 7 output dimensions). (b) The `lambda_jacobian=0.01` weight is small, so the actual effect on the loss may be negligible. (c) With `create_graph=True`, second-order gradient issues (exploding/vanishing) can destabilize training.

**Suggested Fix**: Consider using Hutchinson's trace estimator for a stochastic Jacobian norm approximation (single random projection instead of 7 exact gradients), or reduce the frequency of Jacobian loss computation (e.g., every K-th batch).

**Confidence**: 75%. The impact depends on whether the Jacobian loss actually contributes meaningfully at lambda=0.01.

---

### MEDIUM-3: Physical Survival Limits May Be Too Generous for Bicycle Dynamics

**File**: `config_v9.py`, lines 19-23

**Phenomenon**: The physical limits are:
- `e_y`: 5.0m (lateral error)
- `v`: 5.0 m/s (velocity)
- `delta`: pi/2 = 90 degrees (steering angle)
- `delta_dot`: 10.0 rad/s (steering rate)

**Root Cause**: These limits are set to be permissive to avoid false death classifications, but they are physically unrealistic for a bicycle. A steering angle of 90 degrees would cause an immediate fall. A lateral error of 5m means the bicycle is completely off track.

**Impact**: The survival rate metric may be artificially inflated. A model that predicts delta=80 degrees (physically impossible) would still "survive" the check. This masks model divergence that manifests as physically impossible but numerically finite predictions.

**Suggested Fix**: Tighten limits to physically meaningful ranges (e.g., delta < pi/4, e_y < 2.0m) and report both "numerical survival" and "physical survival" separately.

**Confidence**: 70%. Depends on the actual scale of the data and whether these limits are appropriate for the specific bicycle model.

---

### MEDIUM-4: `predict()` Creates New Tensors on Every Call (Performance)

**File**: `neural_ode_v9.py`, lines 190-198

**Phenomenon**: Each call to `predict()` allocates new `torch.FloatTensor` objects for state and action, runs a forward pass, and converts back to numpy. For a 1000-step rollout, this means 1000 tensor allocations and numpy conversions.

**Impact**: While not a correctness issue, this makes evaluation slow. The planning phase (`phase_3f_planning`) calls `predict()` for 200 candidates x 50 steps x 10 segments = 100,000 calls. This is a practical bottleneck.

**Suggested Fix**: Use `predict_batch()` (which already exists at line 200) for rollout evaluation, or keep tensors on GPU/device during rollouts.

**Confidence**: 95%. Purely performance, not correctness.

---

### LOW-1: Gradient Clipping Threshold (1.0) is Arbitrary

**File**: `neural_ode_v9.py`, line 172

**Phenomenon**: `clip_grad_norm_(self._model.parameters(), 1.0)` clips the total gradient norm to 1.0. This is a reasonable default but may be too aggressive for the Jacobian loss (which naturally has larger gradients) or too lenient for the multi-step loss.

**Impact**: May slow convergence or limit the model's ability to learn from the multi-step and Jacobian losses.

**Confidence**: 60%. Would need gradient monitoring to confirm.

---

### LOW-2: Hardcoded Paths in Configuration

**File**: `config_v9.py`, lines 9-11

**Phenomenon**: `DATA_PATH`, `V8_PATH`, `V9_PATH` are hardcoded absolute paths starting with `D:/`.

**Impact**: Not portable. Not a performance issue.

**Confidence**: 100%.

---

## Self-Questioning

### Q1: Is the shuffled multi-step loss actually harmful, or just wasted compute?

**Analysis**: It is harmful, not just wasted. The multi-step loss with shuffled data computes MSE between `s_cur` (derived from random state + model prediction) and `sb[step+1]` (a completely unrelated random state). The gradient from this loss pushes the model to predict dynamics that transform random states into other random states, which is noise. With `lambda_multi=0.3`, this noise gradient is 30% of the total gradient, actively degrading single-step learning. The only saving grace is that when `rollout_steps=1` (epochs 0-39), `loss_multi` is zero (line 129 condition `rollout_steps > 1`), so the first 20% of training is unaffected.

### Q2: Could the consistency loss error be masked by the small lambda?

**Analysis**: With `lambda_consistency=0.1`, the consistency loss contributes ~10% of the total gradient. If the dimensional error produces a loss value that is small compared to `loss_single`, the gradient contribution may be negligible. However, the consistency loss specifically targets theta (index 3), and if the std ratio `std_theta_dot * dt / std_theta` is far from 1.0, the gradient direction for theta will be systematically wrong. For a bicycle, `theta` (heading angle) has std on the order of radians, while `theta_dot` (angular velocity) has std on the order of rad/s. With dt=1/30, the ratio is approximately `std_theta_dot / (30 * std_theta)`, which could be anywhere from 0.01 to 1.0 depending on the data. If it is far from 1.0, the error is significant even at lambda=0.1.

### Q3: Does the lack of RK4 matter if the model is trained with Euler integration?

**Analysis**: The model is trained with Euler integration (single-step loss targets are consistent with Euler). If we evaluate with RK4, the predicted next states would differ from what the model was trained to produce. However, the "true" dynamics are continuous, and Euler with dt=1/30 introduces systematic bias. Using RK4 for evaluation would actually give more accurate results because it better approximates the continuous integral of the learned dynamics. The model learns `f(s,a) = ds/dt`, and integrating this with a better integrator gives a better approximation of `s(t+dt)`. So switching to RK4 for evaluation is strictly beneficial, even if training used Euler.

### Q4: Is the V8 baseline also affected by these issues?

**Analysis**: The V8 `NeuralODEModel` (from `extended_models.py`) uses only single-step MSE loss with no multi-step, no consistency, and no Jacobian regularization. It does NOT have the shuffled multi-step bug because it doesn't use multi-step training at all. So V8 is not affected by CRITICAL-1 or CRITICAL-2. V8 does use Euler integration (same as V9), so HIGH-2 applies to both. This means the V8 baseline is a cleaner single-step model, and V9's additional losses (multi-step, consistency, Jacobian) may actually be hurting rather than helping due to the bugs identified.

### Q5: Could the per_step_errors in evaluation be incorrectly computed for dead trajectories?

**Analysis**: In `evaluation_v9.py`, when `survived=False` due to NaN/Inf (line 76-78), the loop breaks and `step_errors` contains only errors up to (but not including) the failing step. The NMAE is then computed over this partial trajectory (line 100). The `n_valid` count (line 97) will be less than `n-1`, so `survival_list` at line 107 will mark it as `False`. However, the NMAE value (computed from the partial prefix) is still included in `nmae_list` and averaged at line 115. This means dead trajectories contribute their (typically lower) partial NMAE to the average, biasing the result downward. This is the MEDIUM-1 issue confirmed.

---

## Summary Table

| ID | Severity | Issue | Impact on Long-Horizon |
|----|----------|-------|----------------------|
| CRITICAL-1 | CRITICAL | Multi-step loss uses shuffled data | Multi-step training completely broken |
| CRITICAL-2 | CRITICAL | Consistency loss dimensionally wrong | Wrong gradients on theta dimension |
| HIGH-1 | HIGH | Planning eval is self-referential | Planning metrics meaningless |
| HIGH-2 | HIGH | Euler integration error compounds | Accuracy degrades at H>200 |
| HIGH-3 | HIGH | No validation/early stopping | Potential overfitting |
| MEDIUM-1 | MEDIUM | NMAE biased by dead trajectories | Metrics misleading |
| MEDIUM-2 | MEDIUM | Jacobian reg expensive/unstable | Training instability risk |
| MEDIUM-3 | MEDIUM | Physical limits too generous | Survival rate inflated |
| MEDIUM-4 | MEDIUM | Tensor allocation per predict call | Slow evaluation |
| LOW-1 | LOW | Gradient clip threshold arbitrary | Minor convergence impact |
| LOW-2 | LOW | Hardcoded paths | Portability only |

---

## Final Conclusion

**标签: 已确认 (Confirmed)**

V9 Neural ODE 代码存在两个关键缺陷，直接破坏了其相对于 V8 基线的核心改进：

1. **多步损失因数据洗牌而完全失效**（CRITICAL-1）。V9 相对于 V8 的主要改进是引入了多步 rollout 训练和课程学习，但由于 DataLoader 的 `shuffle=True` 与多步展开的时序连续性假设不兼容，多步损失实际上在计算随机状态之间的 MSE，产生噪声梯度。这意味着 V9 的多步训练策略不仅无效，还可能因噪声梯度（权重 0.3）而损害单步训练质量。

2. **一致性损失在归一化空间中维度错误**（CRITICAL-2）。运动学方程 `theta_next = theta + theta_dot * dt` 仅在物理空间成立，在归一化空间中需要乘以 std 比率修正。当前实现对 theta 维度施加了错误的梯度方向。

3. **规划评估循环论证**（HIGH-1）。Phase 3F 使用模型自身作为"真实动力学"来评估规划质量，得到的 Spearman 相关和 regret 指标仅反映自洽性，不反映真实规划能力。

修复 CRITICAL-1 后重新训练，V9 的长时域预测性能应有显著提升，因为多步 rollout 训练是其设计上对抗误差累积的核心机制。修复 CRITICAL-2 可进一步改善 theta（航向角）维度的预测精度。
