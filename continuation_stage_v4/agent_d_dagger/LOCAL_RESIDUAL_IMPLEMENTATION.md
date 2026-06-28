# Version L: Local Dynamics Residual Implementation

**File:** `version_l_local_residual.py`
**Date:** 2026-06-25

---

## Residual Formula

```
r_k = (F_real(s_model_k, u_k) - GP(s_model_k, u_k)) / delta_std
```

Both real dynamics and GP are evaluated at the **same state**: the model state s_model[k].

---

## Key Design Decisions

### 1. Real Dynamics at Model State

In `_collect_dagger_data_vL()`, line 250:
```python
s_next_real = real_step(s_model_k, tau)  # real dynamics at MODEL state
```

This is the critical difference from Version T. The real dynamics are evaluated at the model's current state, NOT at the real trajectory's state. This gives a clean local error identification: "how wrong is the GP at this specific state?"

### 2. DAgger Distribution

The NN trains on states the model actually visits during rollout (model trajectory states), which is the standard DAgger property. The residual label at each such state is the local dynamics error.

### 3. Round 0 Consistency

Round 0 residuals are computed at expert states (same as original code):
```python
residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std
```
At expert states, s_model = s_real, so Version L and Version T are equivalent in Round 0.

---

## Algorithm

```
For each DAgger round r = 0, 1, ..., R-1:
    1. Train NN ensemble on accumulated (input, residual) pairs
    2. For each rollout step k:
        a. tau = LQR(s_real[k])  # control from real state
        b. s_next_gp = GP(s_model[k], tau)
        c. s_next_real = real_step(s_model[k], tau)  # KEY: at model state
        d. residual = (s_next_real - s_next_gp) / std
        e. delta_nn = NN(s_model[k], tau)
        f. s_model[k+1] = s_next_gp + scale * delta_nn * std
```

---

## When Version L Is Better

Based on experimental results (LOCAL_VS_TRAJECTORY_RESULTS.json):

| Condition | L MAE | T MAE | Winner |
|-----------|-------|-------|--------|
| Mode A (Teacher Forcing) | **0.0023** | 0.0320 | L (14x) |
| Mode D (Closed-loop) | **0.0039** | 0.1294 | L (33x) |
| OOD extreme_phi | **0.0038** | 0.0409 | L (11x) |
| OOD extreme_delta | **0.0019** | 0.0173 | L (9x) |
| Action: small_sine | **0.0238** | 0.2020 | L (8.5x) |

Version L excels when:
- Single-step prediction accuracy matters (Mode A)
- A feedback controller compensates for errors (Mode D)
- The system stays near training distribution
- Actions are smooth and bounded

---

## When Version L Is Worse

| Condition | L MAE | T MAE | Winner |
|-----------|-------|-------|--------|
| Mode B (Open-loop 100 steps) | 0.777 | **0.458** | T (40%) |
| Initial: large_phi | 1.796 | **0.517** | T (3.5x) |
| Initial: high_vel | 3.070 | **0.683** | T (4.5x) |
| Action: large_sine | 0.210 | **0.734** | T (worse) |
| Per-state delta_dot | 1.219 | **0.465** | T (2.6x) |

Version L struggles when:
- Multi-step open-loop rollout is needed
- Initial conditions are far from training
- Large control inputs cause trajectory divergence

---

## Confidence Level

**CONFIRMED**: Implementation directly follows from the mathematical formula.
Code trace available in `version_l_local_residual.py` lines 196-230.
