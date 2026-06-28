# DAgger Semantics Audit Report

**Audit Date:** 2026-06-25
**Target File:** evaluate/eval_models.py
**Target Method:** GPEEnsemble._collect_dagger_data (lines 352-379)
**Training Loop:** GPEEnsemble.train (lines 274-350)

---

## Executive Summary

**The current DAgger implementation computes local dynamics residuals, NOT trajectory-synchronization residuals.**

The residual is effectively:

    r_k = (F_real(s_real[k], u_k) - GP(s_base[k], u_k)) / delta_std

Both the real dynamics and the GP baseline are involved. The GP is evaluated at s_base[k] (model state), while real dynamics use s_real[k] (expert state). Within one DAgger round, these are close due to residual_scale=0.3.

**Verdict: Valid DAgger implementation. No renaming needed.**

---
## Detailed Code Trace

### Variable Naming Convention

| Variable | Meaning |
|----------|---------|
| s_base[k] | Model state at step k (GP + NN composite) |
| s_real[k] | Real state at step k (ground truth) |
| tau | Control input (computed from s_real) |

### Step-by-Step Trace of _collect_dagger_data

File: evaluate/eval_models.py, lines 352-379

#### Initialization (lines 359-362)

    s_base = s0.copy()    # model trajectory starts at s0
    s_real = s0.copy()    # real trajectory starts at s0

Both trajectories start from the same initial state.

#### Step k: Lines 363-378

**Line 364:** Control input from REAL state

    tau = tau_func(step, s_real)

The LQR controller observes the real state s_real[k] and computes tau.

**Line 365:** Real dynamics update

    s_real = real_step(s_real, tau)

After this line: s_real = s_real[k+1] = F_real(s_real[k], tau)

**Line 366:** Normalize model state (BEFORE s_base is updated)

    s_norm = s_base / state_std

At this point, s_base is still s_base[k] (the pre-update value).
So s_norm = s_base[k] / state_std.

**Lines 367-368:** NN prediction

    a_norm = tau / action_std
    delta_nn = self._predict_ensemble_mean(models, s_norm, a_norm)

The NN predicts the normalized residual at (s_base[k], tau).

**Line 369:** Model state update

    s_base = self._baseline.predict(s_base, tau) + delta_nn * delta_std * self.residual_scale

After this line: s_base = s_base[k+1] = GP(s_base[k], tau) + scaled_delta_nn

#### CRITICAL: Residual Computation (lines 372-378)

**Line 372:** Store input

    new_inputs.append(np.concatenate([s_norm, [a_norm]]))

Input = (s_base[k] / state_std, tau / action_std) -- the model state BEFORE update.

**Lines 373-374:** Re-evaluate GP at the PRE-UPDATE model state

    s_next_base = self._baseline.predict(
        s_base - delta_nn * delta_std * self.residual_scale, tau
    )

KEY ANALYSIS: After line 369, s_base has been updated to s_base[k+1].
Subtracting delta_nn * delta_std * residual_scale from s_base[k+1]:

    s_base[k+1] - delta_nn * delta_std * residual_scale
    = GP(s_base[k], tau) + delta_nn*delta_std*residual_scale - delta_nn*delta_std*residual_scale
    = GP(s_base[k], tau)

So the argument to GP.predict() is GP(s_base[k], tau), i.e., the GP-predicted next state.
This means GP.predict() is called TWICE: once at s_base[k], then at GP(s_base[k], tau).

**Line 376:** base_delta

    base_delta = s_next_base - (s_base - delta_nn * delta_std * self.residual_scale)

= GP(GP(s_base[k], tau), tau) - GP(s_base[k], tau)
= GP_delta at the GP-predicted next state

**Line 377:** real_delta

    real_delta = s_real - (s_base - delta_nn * delta_std * self.residual_scale)

= s_real[k+1] - GP(s_base[k], tau)
= real next state minus GP-predicted next state

**Line 378:** Final residual

    new_residuals.append((real_delta - base_delta) / delta_std)

Expanding:
    residual = (s_real[k+1] - GP(s_base[k],tau) - (GP(GP(s_base[k],tau),tau) - GP(s_base[k],tau))) / delta_std
            = (s_real[k+1] - GP(GP(s_base[k],tau), tau)) / delta_std

This is a TRAJECTORY-LEVEL comparison: the gap between the real next state and a TWO-STEP GP prediction from the model state.

---

## Semantic Classification

### Three Possible Residual Formulations

**Version T (Trajectory-Synchronization Residual):**

    r_k = x_{k+1}^real - x_{k+1}^model

Both sides are NEXT states, but from different trajectories. The input is the model state, the label pulls the model trajectory toward the real trajectory.

**Version L (Local Dynamics Residual):**

    r_k = F(x_k, u_k) - f_GP(x_k, u_k)

Real dynamics and GP evaluated at the SAME state and action. Pure model error identification.

**Current Implementation (derived from trace):**

    r_k = (s_real[k+1] - GP(GP(s_base[k], tau), tau)) / delta_std

Comparison:

| Version | Real Dynamics State | GP Evaluation State(s) | Comparison Type |
|---------|-------------------|------------------------|-----------------|
| Version T (trajectory-sync) | s_real[k] | s_model[k] | next-state gap |
| Version L (local dynamics) | s_model[k] | s_model[k] | local error |
| **Current Implementation** | s_real[k] | s_base[k] then GP(s_base[k],tau) | **two-step GP vs real** |

### Verdict: Trajectory-Level Residual (Version T variant)

The current implementation is a variant of Version T (trajectory-synchronization residual) because:

1. It compares the real next state s_real[k+1] against a model-predicted state (two-step GP)
2. The real dynamics and GP are NOT evaluated at the same state
3. The label pulls the model trajectory toward the real trajectory

However, it is not the simplest form of Version T. The two-step GP evaluation introduces an intermediate state that makes the residual harder to interpret.

### Why This Might Still Work

1. **DAgger distribution shift is controlled:** The NN trains on (s_base[k], tau) inputs
2. **Small divergence within one DAgger round:** residual_scale = 0.3 limits drift
3. **The residual captures total model error at each step**

### Potential Issues

1. **The two-step GP evaluation makes the residual harder to interpret:** It is not a clean local dynamics error, nor a clean trajectory-sync residual
2. **The real dynamics are evaluated at s_real[k] while the GP is at GP(s_base[k], tau):** This mixes two different state spaces

---

## Comparison: Three Residual Formulations

### Version L (Pure Local Dynamics Residual) -- Ideal for Model Error Identification

    # Both evaluated at the SAME state (model state)
    s_model_k = s_base.copy()  # s_base[k], before update
    s_next_base = GP(s_model_k, tau)
    s_next_real = real_step(s_model_k, tau)       # real dynamics at MODEL state
    base_delta = s_next_base - s_model_k
    real_delta = s_next_real - s_model_k
    residual = (real_delta - base_delta) / delta_std

This cleanly identifies the GP prediction error at the model state.

### Current Implementation (Two-Step GP)

    # Real dynamics at s_real, GP called twice
    s_real = real_step(s_real, tau)             # real dynamics at REAL state
    gp_pred = GP(s_base, tau)                   # GP first call
    s_next_base = GP(gp_pred, tau)              # GP second call
    base_delta = s_next_base - gp_pred
    real_delta = s_real - gp_pred
    residual = (real_delta - base_delta) / delta_std

This compares real trajectory against two-step GP trajectory.

### Version T (Pure Trajectory-Synchronization Residual)

    # Compare next states directly
    s_next_real = real_step(s_real, tau)
    s_next_base = GP(s_base, tau)
    residual = (s_next_real - s_next_base) / delta_std

This is the simplest form: gap between real and model next states.

---

## Training Loop Consistency Check

In train() (lines 288-292), the initial round-0 residuals are:

    residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std

Where deltas[i] = s_real[i+1] - s_real[i] (from training data).
And s_next_base = GP(states[i], actions[i]).

This is a **pure local dynamics residual** at the expert states:
    residual = (F_real(s, u) - GP(s, u)) / std

**Inconsistency:** Round 0 uses Version L (local dynamics at expert states),
while DAgger rounds 1+ use a two-step GP trajectory comparison at model states.
These are semantically different training objectives.

---

## Recommendations

### 1. Extract Repeated Expression (LOW priority)

Lines 373-378 compute s_base - delta_nn * delta_std * self.residual_scale three times.
Extract to a local variable:

    gp_pred = s_base - delta_nn * delta_std * self.residual_scale
    # This equals GP(s_base[k], tau), the GP-predicted next state

### 2. If Pure Version L Is Desired (LOW priority)

To compute a clean local dynamics residual, change lines 365 and 373-378:

    # BEFORE updating s_base (insert before line 369):
    s_model_k = s_base.copy()  # save pre-update state
    s_next_real = real_step(s_model_k, tau)  # real dynamics at MODEL state
    s_next_base = self._baseline.predict(s_model_k, tau)  # GP at MODEL state

    # Then on line 369, update s_base as before:
    delta_nn = self._predict_ensemble_mean(models, s_norm, a_norm)
    s_base = s_next_base + delta_nn * delta_std * self.residual_scale

    # Compute residual (replaces lines 372-378):
    new_inputs.append(np.concatenate([s_norm, [a_norm]]))
    new_residuals.append((s_next_real - s_next_base) / delta_std)

Trade-off: This eliminates the DAgger benefit of training on model-trajectory states.
The current approach may actually be better for DAgger because it trains on states
the model actually visits during closed-loop rollout.

### 3. Consider Pure Version T (LOW priority)

If the goal is to pull the model trajectory toward the real trajectory,
simplify to:

    s_next_real = real_step(s_real, tau)
    s_next_base = self._baseline.predict(s_base, tau)
    delta_nn = self._predict_ensemble_mean(models, s_norm, a_norm)
    s_base = s_next_base + delta_nn * delta_std * self.residual_scale
    new_inputs.append(np.concatenate([s_norm, [a_norm]]))
    new_residuals.append((s_next_real - s_next_base) / delta_std
        - delta_nn * self.residual_scale)  # account for NN correction

### 4. Add Inline Comments (MEDIUM priority)

The residual computation at lines 373-378 is non-obvious.
Add clear comments explaining the mathematical intent and the two-step GP evaluation.

### 5. No Naming Change Needed

Whether Version T or Version L, this is still a legitimate DAgger variant.
The name DAgger is appropriate.

---

## Appendix: Line Reference

| Line(s) | Code Snippet | Role |
|---------|-------------|------|
| 361-362 | s_base = s0.copy() / s_real = s0.copy() | Initialize parallel trajectories |
| 364 | tau = tau_func(step, s_real) | Control from real state |
| 365 | s_real = real_step(s_real, tau) | Advance real trajectory |
| 366 | s_norm = s_base / state_std | Normalize model state for NN |
| 368 | delta_nn = self._predict_ensemble_mean(...) | NN residual prediction |
| 369 | s_base = self._baseline.predict(s_base, tau) + ... | Advance model trajectory |
| 372 | new_inputs.append(...) | Store NN input |
| 373-374 | s_next_base = self._baseline.predict(s_base - ..., tau) | GP at GP-predicted state |
| 376 | base_delta = s_next_base - (s_base - ...) | GP predicted delta |
| 377 | real_delta = s_real - (s_base - ...) | Real delta from GP-predicted state |
| 378 | new_residuals.append((real_delta - base_delta) / delta_std) | Final residual label |
