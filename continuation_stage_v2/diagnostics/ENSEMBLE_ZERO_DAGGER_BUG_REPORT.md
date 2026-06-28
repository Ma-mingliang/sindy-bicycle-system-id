# gp_ensemble_5_0 MAE=0 Bug Report

> **Severity**: CRITICAL  
> **Date**: 2026-06-25  
> **Model**: gp_ensemble_5_0 (5 NN models, 0 DAgger rounds)  
> **Symptom**: MAE=0.000 across all horizons in Modes A and B, while gp_standard shows MAE>12 at 500 steps  

---

## Root Cause

When dagger_rounds=0, the DAgger training loop in GPEEnsemble.train() never executes, leaving self._ensemble_models as an empty list []. The _predict_ensemble_mean() method returns np.mean([]) which equals nan. This makes predict() return nan for all state dimensions.

The metrics computation then silently masks this as MAE=0.0:

1. compute_all_metrics() computes errors = states_model - states_real
2. NaN predictions produce NaN errors from step 1 onward
3. valid = ~np.any(np.isnan(abs_errors), axis=1) filters out all NaN rows
4. Only step 0 (the initial state, always error=0) remains valid
5. MAE computed over just step 0 is exactly 0.0

## Affected Files

### evaluate/eval_models.py

**Bug Location 1**: GPEEnsemble.train() -- Lines 303-350

Line 303: self._ensemble_models = [] (initialized empty)
Line 305: for round_i in range(self.dagger_rounds): -- range(0) is empty, loop never executes
Line 341: self._ensemble_models = models -- never reached when dagger_rounds=0

**Bug Location 2**: _predict_ensemble_mean() -- Lines 381-392

Line 387: for model in models: -- models is [], loop does not execute
Line 392: return np.mean(preds, axis=0) -- preds is [], np.mean([]) = nan

**Bug Location 3**: predict() -- Lines 408-413

Line 412: delta_nn = self._predict_ensemble_mean(...) -- returns nan
Line 413: return s_next_base + nan * delta_std * residual_scale -- entire output is nan

### evaluate/eval_metrics.py

**Masking Location**: _compute_overall_metrics() -- Lines 60-81

Line 63: valid = ~np.any(np.isnan(abs_errors), axis=1) -- filters NaN rows
Line 70: abs_err_valid = abs_errors[valid] -- only step 0 remains valid
Line 74: mae = float(np.mean(abs_err_valid)) -- mean over step 0 = 0.0
Line 79: endpoint_error = float(np.mean(abs_errors[-1])) -- NOT filtered, = nan

## Why MAE=0.0 but endpoint_error=NaN

The endpoint_error is computed from abs_errors[-1] (the last row) WITHOUT the valid filter. Since the last row is NaN, endpoint_error=NaN.

All other metrics (MAE, RMSE, etc.) use abs_err_valid which only contains step 0. Step 0 error is always 0 (initial state), so MAE=0.0.

## Mode-by-Mode Impact

| Mode | Behavior | MAE Result |
|------|----------|------------|
| A (teacher forcing) | NaN masked, only step 0 counted | **0.0 (false positive)** |
| B (open loop) | NaN masked, only step 0 counted | **0.0 (false positive)** |
| C (hybrid) | NaN propagates to GP baseline at step 2 | **Exception: sklearn NaN error** |
| D (closed loop) | NaN propagates to GP baseline at step 2 | **Exception: sklearn NaN error** |

## Evidence from full_evaluation_v2.json

gp_ensemble_5_0 Mode A: mae=0.0, endpoint_error=NaN, survival_steps=1000
gp_ensemble_5_0 Mode B: mae=0.0, endpoint_error=NaN, survival_steps=1000
gp_ensemble_5_0 Mode C: error=Input X contains NaN (sklearn GP failure)
gp_ensemble_5_0 Mode D: error=Input X contains NaN (sklearn GP failure)

## Why gp_standard Shows MAE>12 at 500 Steps

gp_standard is a pure GP model making single-step predictions. Over 500 open-loop steps, GP prediction errors compound, reaching MAE=12.1. This is expected behavior.

gp_ensemble_5_0 should behave similarly (or better with DAgger), but instead returns NaN because the ensemble component is never trained.

## Fix Recommendation

**Primary Fix** -- In GPEEnsemble.train(), change line 305 from:

    for round_i in range(self.dagger_rounds):

to:

    for round_i in range(max(1, self.dagger_rounds)):

And keep the DAgger data collection guard at line 344:

    if round_i < self.dagger_rounds - 1:
        new_inputs, new_residuals = self._collect_dagger_data(...)

This ensures NN models are always trained at least once on the initial residual data.

**Alternative Fix** -- Add a guard in _predict_ensemble_mean():

    def _predict_ensemble_mean(self, models, s_norm, a_norm):
        if not models:
            return np.zeros(4)
        ...

**Additional Issue**: No validation that _ensemble_models is non-empty before prediction. Add a check in predict() and predict_with_uncertainty().

## Test Results

See ENSEMBLE_EQUIVALENCE_TEST.json for quantitative test results.
