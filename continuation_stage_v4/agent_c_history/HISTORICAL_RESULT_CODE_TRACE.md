# Historical Result Code Trace: 0.064, 0.066, 0.252

> **Generated**: 2026-06-25
> **Method**: Direct code tracing through source files, NOT numerical inference
> **Confidence levels**: CONFIRMED (code path verified), INFERRED (indirect evidence), UNKNOWN (insufficient data)

---

## 1. Result 0.064

### 1.1 Report Location

Referenced in: `REPORT_集成方法.md`, `GP_TARGETED_AUDIT.md`, `test_ensemble_optimization.py` header ("在最佳结果(0.064)基础上尝试改进"), `reproduce_064.py` docstring.

### 1.2 Script Chain

```
test_ensemble.py __main__
  -> test_ensemble("GP + 集成(5模型)", GPMethod, n_models=5, residual_scale=0.3)
    -> train_ensemble(GPMethod, n_models=5, n_epochs=100, dagger_rounds=3, residual_scale=0.3)
      -> generate_training_data(30000)
      -> GPMethod().train(states, actions, deltas, state_std, action_std, delta_std)
        -> GPStandard(max_samples=5000, n_restarts=2)
      -> EnsembleResidualNet(n_models=5).train(train_inputs, train_residuals, n_epochs=100)
        -> 5x ResidualNet with torch.manual_seed(i*42), np.random.seed(i*42)
        -> Adam lr=1e-3, MSE loss, batch_size=256
        -> NO LR scheduler
      -> DAgger: 3 rounds, round 0 uses base residuals, rounds 1+ collect new data
        -> DAgger data: 3 segments, tau_func(100+seg_i, 500), phi_init UNSEEDED
    -> evaluate loop (5 segments, 500 steps each)
```

### 1.3 Function: `test_ensemble()` Evaluation Loop (lines 196-208)

```python
for seg_i in range(n_segments):         # 5 segments
    tau_func = make_tau_func(seg_i, 500)
    phi_init = np.random.uniform(-0.25, 0.25)   # UNSEEDED
    s0 = np.array([phi_init, 0.0, 0.0, 0.0])
    s_real = s0.copy()
    s_ensemble = s0.copy()

    for step in range(500):
        tau = tau_func(step, s_real)              # tau from REAL state
        s_real = real_step(s_real, tau)           # real system advances
        s_ensemble, uncertainty = ensemble_predict_with_uncertainty(
            baseline, ensemble, s_ensemble, tau,  # model self-rolls
            state_std, action_std, delta_std, residual_scale
        )
        if abs(s_real[0]) > math.pi / 3:
            break
        if step + 1 in eval_steps:
            all_errors[step + 1].append(abs(s_ensemble[0] - s_real[0]))  # PHI-ONLY
```

### 1.4 Model Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Baseline | GPStandard | `methods_classic.GPMethod` |
| GP max_samples | 5000 | hardcoded in GPMethod |
| GP n_restarts | 2 | hardcoded in GPMethod |
| GP kernel | ConstantKernel(1.0) * RBF(1.0) | line 118 eval_models.py |
| GP alpha | 1e-6 | hardcoded |
| NN count | 5 | n_models=5 argument |
| NN architecture | ResidualNet (128 hidden, SiLU) | methods_nn.py |
| DAgger rounds | 3 | dagger_rounds=3 argument |
| residual_scale | 0.3 | hardcoded |
| n_epochs | 100 | hardcoded |
| LR scheduler | **NONE** | No scheduler in EnsembleResidualNet.train() |
| Training data | 30000 samples | generate_training_data(30000) |

### 1.5 Evaluation Configuration

| Parameter | Value | Evidence |
|-----------|-------|----------|
| Evaluation mode | Mode C style | tau = tau_func(step, s_real), model self-rolls |
| Error metric | phi-only endpoint error | `abs(s_ensemble[0] - s_real[0])` at step 500 |
| State channels | phi only (index 0) | `[0]` indexing |
| Unit | rad | phi is roll angle in radians |
| Time domain | 500 steps | range(500), eval at step 500 |
| Segments | 5 | n_segments=5 |
| Initial state | UNSEEDED | `np.random.uniform(-0.25, 0.25)` |
| Divergence guard | abs(s_real[0]) > pi/3 | Early termination |

### 1.6 Status: CONFIRMED

All attributes verified by direct code reading of `test_ensemble.py`.

### 1.7 Critical Finding

The 0.064 result uses **phi-only endpoint error** (single scalar at final step), NOT full-trajectory MAE across all states. This is fundamentally different from the 0.252 metric.

---

## 2. Result 0.066

### 2.1 Report Location

Referenced in: `continuation_stage_v2/audit/HISTORICAL_0064_0066_CODE_TRACE.md`, `CURRENT_TASK_COMPLETION_FACTS_V2.json` (ISSUE_4).

### 2.2 Script Chain

```
test_ensemble_optimization.py __main__
  -> test_optimization_params(GPMethod)
    -> config "7模型+5轮DAgger":
       n_models=7, dagger_rounds=5, residual_scale=0.3, n_train=30000, use_scheduler=False
    -> train_optimized_ensemble(GPMethod, n_models=7, dagger_rounds=5, residual_scale=0.3, use_scheduler=False)
      -> generate_training_data(30000)
      -> GPMethod().train(...)
      -> OptimizedEnsembleResidualNet(n_models=7, lr=1e-3, use_scheduler=False).train(...)
        -> 7x ResidualNet with torch.manual_seed(i*42), np.random.seed(i*42)
        -> Adam lr=1e-3, MSE loss, batch_size=256
        -> NO LR scheduler (use_scheduler=False)
      -> DAgger: 5 rounds
    -> evaluation loop (5 segments, 500 steps each)
```

### 2.3 Function: `test_optimization_params()` Evaluation Loop (lines 261-281)

```python
for seg_i in range(n_segments):         # 5 segments
    tau_func = make_tau_func(seg_i, 500)
    phi_init = np.random.uniform(-0.25, 0.25)   # UNSEEDED
    s0 = np.array([phi_init, 0.0, 0.0, 0.0])
    s_real = s0.copy()
    s_ensemble = s0.copy()

    for step in range(500):
        tau = tau_func(step, s_real)              # tau from REAL state
        s_real = real_step(s_real, tau)
        s_ensemble = optimized_ensemble_predict(  # model self-rolls
            baseline, ensemble, s_ensemble, tau,
            state_std, action_std, delta_std, config['residual_scale']
        )
        if abs(s_real[0]) > math.pi / 3:
            break
        if step == 499:
            errors_500.append(abs(s_ensemble[0] - s_real[0]))  # PHI-ONLY
```

### 2.4 Model Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Baseline | GPStandard | Same as 0.064 |
| NN count | **7** | n_models=7 |
| DAgger rounds | **5** | dagger_rounds=5 |
| residual_scale | 0.3 | config |
| n_epochs | 100 | hardcoded |
| LR scheduler | **NONE** | use_scheduler=False in config |
| Training data | 30000 samples | config |

### 2.5 Evaluation Configuration

| Parameter | Value | Evidence |
|-----------|-------|----------|
| Evaluation mode | Mode C style | Same as 0.064 |
| Error metric | phi-only endpoint error | `abs(s_ensemble[0] - s_real[0])` at step 499 |
| State channels | phi only | `[0]` indexing |
| Unit | rad | |
| Time domain | 500 steps | step == 499 check |
| Segments | 5 | n_segments=5 |
| Initial state | UNSEEDED | `np.random.uniform(-0.25, 0.25)` |

### 2.6 Status: CONFIRMED

All attributes verified by direct code reading of `test_ensemble_optimization.py`.

### 2.7 Relationship to 0.064

0.066 uses the SAME evaluation code path as 0.064, with two training differences:
- 7 NN models (vs 5)
- 5 DAgger rounds (vs 3)

The header of test_ensemble_optimization.py confirms: "在最佳结果(0.064)基础上尝试改进" -- 0.064 was the known best, 0.066 is a slightly worse attempt with more models/DAgger.

---

## 3. Result 0.252

### 3.1 Report Location

Referenced in: `FINAL_HANDOVER_V3.md` table 3.1, `key_results_summary.md` table Mode B, `full_evaluation_v2.json`.

### 3.2 Script Chain

```
continuation_stage/evaluate/run_full_evaluation_v2.py __main__
  -> main()
    -> seeds = [42, 43, 44, 45, 46]
    -> horizons = [10, 50, 100, 500, 1000]
    -> generate_training_data(30000)
    -> GPEEnsemble(n_models=7, dagger_rounds=5, residual_scale=0.3, n_epochs=100, use_scheduler=True)
       .train(states, actions, deltas, s_std, a_std, d_std)
    -> For each seed:
       rng = np.random.RandomState(seed)
       phi_init = rng.uniform(-0.25, 0.25)    # SEEDED
       tau_func = me.make_tau_func(seed - 42, 1000)
       ref = generate_reference_trajectory(s0, tau_func, 1000, mc.real_step)
    -> For each horizon=500:
       run_mode_b_fixed(model, s0, ref, 500, mc.real_step)
       compute_all_metrics(states_model, states_real, ...)
    -> mean across 5 seeds -> 0.251891 ≈ 0.252
```

### 3.3 Function: `run_mode_b_fixed()` (eval_modes_v2.py lines 44-92)

```python
def run_mode_b_fixed(model, s0, reference, n_steps, real_dynamics):
    actions = reference['actions']     # FROZEN from reference trajectory
    n_valid = min(n_steps, len(actions))

    # Real system: from s0, using frozen actions
    states_real = [s0.copy()]
    s_real = s0.copy()
    for i in range(n_valid):
        s_real = real_dynamics(s_real, actions[i])    # real uses frozen actions
        states_real.append(s_real.copy())

    # Model: from s0, using SAME frozen actions
    states_model = [s0.copy()]
    s_model = s0.copy()
    for i in range(n_valid):
        tau = actions[i]
        s_next, unc = model.predict_with_uncertainty(s_model, tau)
        s_model = s_next
        states_model.append(s_model.copy())
```

### 3.4 Function: `compute_all_metrics()` (eval_metrics.py lines 60-81)

```python
def _compute_overall_metrics(abs_errors, errors, state_std):
    valid = ~np.any(np.isnan(abs_errors), axis=1)
    abs_err_valid = abs_errors[valid]
    return {
        'mae': float(np.mean(abs_err_valid)),           # ALL 4 states, ALL steps
        'endpoint_error': float(np.mean(abs_errors[-1])),
        ...
    }
```

**MAE = mean of |s_model - s_real| over ALL 4 state channels and ALL valid time steps.**

### 3.5 Model Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Model class | GPEEnsemble | eval_models.py |
| Baseline | GPStandard(max_samples=5000) | GPEEnsemble.train() line 285 |
| GP kernel | ConstantKernel(1.0) * RBF(1.0) | GPStandard.train() line 118 |
| GP n_restarts | 2 | GPStandard default |
| NN count | **7** | n_models=7 |
| NN architecture | ResidualNet (128 hidden, SiLU) | methods_nn.py |
| DAgger rounds | **5** | dagger_rounds=5 |
| residual_scale | 0.3 | hardcoded |
| n_epochs | 100 | hardcoded |
| LR scheduler | **CosineAnnealingLR(T_max=100)** | use_scheduler=True |
| Training data | 30000 samples | generate_training_data(30000) |

### 3.6 Evaluation Configuration

| Parameter | Value | Evidence |
|-----------|-------|----------|
| Evaluation mode | **Mode B** (fixed action open-loop) | run_mode_b_fixed() |
| Error metric | **Full-trajectory MAE** (all 4 states) | compute_all_metrics() -> overall.mae |
| State channels | All 4: phi, delta, phi_dot, delta_dot | abs_errors has shape (n+1, 4) |
| Unit | **Mixed**: rad (phi, delta) + rad/s (phi_dot, delta_dot) | No unit normalization |
| Time domain | 500 steps | horizon=500 |
| Seeds | **5 fixed seeds** (42-46) | seeds=[42,43,44,45,46] |
| Initial state | SEEDED | rng.uniform(-0.25, 0.25) with RandomState(seed) |
| Aggregation | Mean across 5 seeds | np.mean(mae_values) |
| Divergence guard | NaN or abs > 100 | eval_modes_v2.py line 78 |

### 3.7 Per-Seed Values (CONFIRMED from full_evaluation_v2.json)

| Seed | phi_init | MAE (500-step) |
|------|----------|----------------|
| 42 | from RandomState(42) | 0.350003 |
| 43 | from RandomState(43) | 0.047898 |
| 44 | from RandomState(44) | 0.375250 |
| 45 | from RandomState(45) | 0.216319 |
| 46 | from RandomState(46) | 0.269986 |
| **Mean** | | **0.251891** |
| **Std** | | **0.116661** |

### 3.8 Status: CONFIRMED

All attributes verified by direct code reading of `run_full_evaluation_v2.py`, `eval_modes_v2.py`, `eval_metrics.py`, and `eval_models.py`. Per-seed values verified from `full_evaluation_v2.json`.

---

## 4. Critical Differences Between Results

### 4.1 Summary Table

| Attribute | 0.064 | 0.066 | 0.252 |
|-----------|-------|-------|-------|
| **Source script** | test_ensemble.py | test_ensemble_optimization.py | run_full_evaluation_v2.py |
| **Eval mode** | Mode C style | Mode C style | **Mode B** (fixed action) |
| **Action source** | Real state (tau_func) | Real state (tau_func) | **Frozen reference** |
| **Error metric** | **phi-only endpoint** | **phi-only endpoint** | **Full-trajectory MAE (4 states)** |
| **State channels** | phi only | phi only | **All 4** |
| **Unit** | rad | rad | **Mixed (rad + rad/s)** |
| **Steps** | 500 | 500 | 500 |
| **Initial state** | **UNSEEDED** | **UNSEEDED** | **SEEDED** (42-46) |
| **Reproducible** | NO | NO | YES |
| **NN count** | 5 | 7 | 7 |
| **DAgger rounds** | 3 | 5 | 5 |
| **LR scheduler** | **NONE** | **NONE** | **CosineAnnealing** |
| **Aggregation** | mean(5 segments) | mean(5 segments) | **mean(5 seeds)** |
| **Eval segments** | 5 | 5 | 1 per seed |
| **result file** | stdout only | stdout only | full_evaluation_v2.json |

### 4.2 Why They Are NOT Comparable

1. **Different metrics**: phi-only endpoint error (0.064/0.066) vs full-trajectory 4-state MAE (0.252). The full-trajectory MAE includes 3 additional state channels (delta, phi_dot, delta_dot) which tend to have larger errors, inflating the overall MAE.

2. **Different evaluation modes**: Mode C style (0.064/0.066) vs Mode B (0.252). In Mode C, actions come from real state at each step, so the action sequence adapts to the real trajectory. In Mode B, actions are frozen from a reference, so both real and model use the same fixed sequence. For a non-diverging model, these produce the same action sequence (since the real trajectory is the same), but the semantics differ.

3. **Different training pipelines**: 0.064/0.066 use `EnsembleResidualNet` (no scheduler), while 0.252 uses `GPEEnsemble` (with CosineAnnealingLR scheduler). The scheduler changes the training dynamics, producing different trained models.

4. **Different aggregation**: 0.064/0.066 average over 5 unseeded segments (random phi_init). 0.252 averages over 5 seeded evaluations (deterministic phi_init per seed).

### 4.3 Old Results Bug: Mode B == Mode C

The old `ensemble_evaluation_results.json` (generated by `evaluate/run_ensemble_evaluation.py`) shows **identical MAE for Mode B and Mode C** for all models. This is because the old `evaluate/eval_modes.py` has `run_mode_b` and `run_mode_c` that produce identical action sequences:
- Both compute `tau = tau_func(i, s_real)` from the same real trajectory
- The real trajectory is identical because the same actions are applied
- The model trajectory is identical because the same actions are used

This bug was fixed in `eval_modes_v2.py` by using a pre-generated reference trajectory for Mode B.

---

## 5. Additional Trace: old ensemble_evaluation_results.json

### 5.1 Source

Generated by: `evaluate/run_ensemble_evaluation.py`
Uses: `evaluate.eval_modes` (old, buggy Mode B/C)
Models: gp_ensemble_5_3, gp_ensemble_7_5
Training: GPEEnsemble with use_scheduler=True
Seeds: from config file (5 seeds)
Horizons: [10, 50, 100, 500, 1000]

### 5.2 Key Values (gp_ensemble_7_5 Mode B 500-step)

| Seed | MAE |
|------|-----|
| 42 | 0.350003 |
| 43 | 0.047898 |
| 44 | 0.375250 |
| 45 | 0.216319 |
| 46 | 0.269986 |
| **Mean** | **0.251891** |

These values are IDENTICAL to the new full_evaluation_v2.json, confirming the same model training and evaluation code paths (despite the Mode B/C bug in old eval_modes.py, Mode B still produces correct results for non-diverging models).

---

*Trace complete: 2026-06-25*
*All findings based on direct code reading, no numerical inference*
