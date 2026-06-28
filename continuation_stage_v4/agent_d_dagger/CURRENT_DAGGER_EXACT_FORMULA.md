# Current DAgger Exact Formula

**Date:** 2026-06-25
**Source File:** evaluate/eval_models.py
**Class:** GPEEnsemble
**Methods:** train() lines 274-350, _collect_dagger_data() lines 352-379

---

## Executive Summary

The current DAgger implementation uses **two different residual formulas** across rounds:
- **Round 0**: Pure local dynamics residual (Version L) at expert states
- **Round 1+**: Two-step GP trajectory comparison at model states

These are semantically inconsistent training objectives. The Round 1+ formula is NOT a standard DAgger formulation.

---

## Round 0: Pure Local Dynamics Residual (Version L)

### Code (lines 288-292)

```python
residuals = np.empty_like(deltas)
for i in range(len(states)):
    s_next_base = self._baseline.predict(states[i], actions[i])
    residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std
```

### Mathematical Formula

```
r_k = (delta_k - delta_GP_k) / delta_std
    = ((s_{k+1}^real - s_k^real) - (GP(s_k^real, u_k) - s_k^real)) / delta_std
    = (s_{k+1}^real - GP(s_k^real, u_k)) / delta_std
    = (F_real(s_k^real, u_k) - GP(s_k^real, u_k)) / delta_std
```

### Semantic Classification: Version L (Local Dynamics Residual)

- Both real dynamics and GP evaluated at the **same state**: s_k^real (expert state)
- Training input: (s_k^real, u_k) -- expert state, expert action
- Training label: normalized prediction error of GP at expert state
- **Confidence: CONFIRMED** -- directly from code

---

## Round 1+: Two-Step GP Trajectory Comparison

### Code Trace (lines 352-379)

**Step k execution order:**

```python
# Line 361-362: Initialize
s_base = s0.copy()    # model state
s_real = s0.copy()    # real state

# Line 364: Control from REAL state
tau = tau_func(step, s_real)        # tau computed from s_real[k]

# Line 365: Advance REAL trajectory
s_real = real_step(s_real, tau)     # s_real = s_real[k+1]

# Line 366: Normalize MODEL state (BEFORE update)
s_norm = s_base / state_std        # s_norm = s_base[k] / state_std

# Lines 367-368: NN prediction
a_norm = tau / action_std
delta_nn = self._predict_ensemble_mean(models, s_norm, a_norm)

# Line 369: Advance MODEL trajectory
s_base = self._baseline.predict(s_base, tau) + delta_nn * delta_std * self.residual_scale
# After: s_base = s_base[k+1] = GP(s_base[k], tau) + scaled_NN_correction
```

**CRITICAL: Residual computation (lines 372-378):**

```python
# Line 372: Store input (model state BEFORE update)
new_inputs.append(np.concatenate([s_norm, [a_norm]]))
# Input = (s_base[k] / state_std, tau / action_std)

# Lines 373-374: Re-evaluate GP at the GP-predicted state
s_next_base = self._baseline.predict(
    s_base - delta_nn * delta_std * self.residual_scale,  # = GP(s_base[k], tau)
    tau
)
# s_next_base = GP(GP(s_base[k], tau), tau) = two-step GP prediction

# Line 376: GP delta at GP-predicted state
base_delta = s_next_base - (s_base - delta_nn * delta_std * self.residual_scale)
# = GP(GP(s_base[k],tau), tau) - GP(s_base[k], tau)

# Line 377: Real delta from GP-predicted state
real_delta = s_real - (s_base - delta_nn * delta_std * self.residual_scale)
# = s_real[k+1] - GP(s_base[k], tau)

# Line 378: Final residual
new_residuals.append((real_delta - base_delta) / delta_std)
```

### Mathematical Formula

Let `gp_k = GP(s_base[k], u_k)` (first GP call at model state).
Let `gp_k1 = GP(gp_k, u_k)` (second GP call at GP-predicted state).

```
base_delta  = gp_k1 - gp_k
real_delta  = s_real[k+1] - gp_k

r_k = (real_delta - base_delta) / delta_std
    = (s_real[k+1] - gp_k - (gp_k1 - gp_k)) / delta_std
    = (s_real[k+1] - gp_k1) / delta_std
    = (s_real[k+1] - GP(GP(s_base[k], u_k), u_k)) / delta_std
```

### Semantic Classification: Trajectory-Level Comparison (NOT Version L, NOT standard Version T)

| Aspect | Value |
|--------|-------|
| Real dynamics state | s_real[k] (for computing tau) then s_real[k+1] (result) |
| GP evaluation state | s_base[k] first call, then GP(s_base[k],tau) second call |
| Comparison type | Real next state vs TWO-STEP GP trajectory |
| Input state | s_base[k] (model state) |
| **Confidence** | **CONFIRMED -- derived from code trace** |

### Why This Is Neither Version L Nor Standard Version T

**Not Version L (Local Dynamics):**
- Version L requires both real dynamics and GP at the SAME state: `F(s,u) - GP(s,u)`
- Current: real dynamics at s_real[k], GP at GP(s_base[k],tau) -- different states

**Not Standard Version T (Trajectory Synchronization):**
- Standard Version T: `r_k = s_real[k+1] - s_next_base[k]`
  where `s_next_base[k] = GP(s_base[k], tau)`
- Current: `r_k = s_real[k+1] - GP(GP(s_base[k], tau), tau)`
  GP is called TWICE, not once

**What It Actually Is:**
- A multi-step trajectory comparison that measures the gap between real next state and a two-step-ahead GP prediction
- The GP is evaluated at its own predicted intermediate state, introducing a "hallucinated" state into the residual computation

---

## Inconsistency Summary

| Round | Formula | State Alignment | Semantic |
|-------|---------|----------------|----------|
| 0 | (F_real(s,u) - GP(s,u)) / std | Same state (expert) | Version L |
| 1+ | (s_real[k+1] - GP(GP(s_base[k],tau),tau)) / std | Different states | Two-step GP trajectory |

**This inconsistency means the NN is trained on two fundamentally different objectives across rounds.**

---

## Predict() Function (Inference)

```python
# Line 408-413
def predict(self, s, tau):
    s_next_base = self._baseline.predict(s, tau)
    s_norm = s / self._state_std
    a_norm = tau / self._action_std
    delta_nn = self._predict_ensemble_mean(self._ensemble_models, s_norm, a_norm)
    return s_next_base + delta_nn * self._delta_std * self.residual_scale
```

At inference, the composite model is:
```
s_next = GP(s, u) + residual_scale * NN(s, u) * delta_std
```

This is a standard residual correction structure. The NN predicts a correction to the GP prediction, scaled by `residual_scale`.

---

## Evidence Table

| Claim | Evidence | Confidence |
|-------|----------|------------|
| Round 0 is Version L | Lines 288-292: `(deltas[i] - (s_next_base - states[i])) / delta_std` | CONFIRMED |
| Round 1+ uses two-step GP | Lines 373-374: `s_next_base = self._baseline.predict(s_base - delta_nn * ..., tau)` where argument = GP(s_base[k],tau) | CONFIRMED |
| Round 1+ residual formula | Lines 376-378: expanded to `(s_real[k+1] - GP(GP(s_base[k],tau),tau)) / delta_std` | CONFIRMED |
| DAgger semantics audit is WRONG | Audit claims "pure local dynamics residual" but code shows two-step GP comparison | CONFIRMED (audit refuted) |
| residual_scale = 0.3 | Default parameter, line 262 | CONFIRMED |
| NN input is model state | Line 372: `s_norm = s_base / state_std` before update | CONFIRMED |
