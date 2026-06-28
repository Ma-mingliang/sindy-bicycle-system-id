# Version T: Trajectory-Synchronization Residual Implementation

**File:** `version_t_trajectory_residual.py`
**Date:** 2026-06-25

---

## Residual Formula

```
r_k = (s_real[k+1] - GP(s_model[k], u_k)) / delta_std
```

Real dynamics evaluated at s_real[k] producing s_real[k+1].
GP evaluated at s_model[k].
The residual is the next-state gap between the real trajectory and the GP prediction.

---

## Key Design Decisions

### 1. Real Dynamics at Real State

In `_collect_dagger_data_vT()`, line 251:
```python
s_real_next = real_step(s_real, tau)  # real dynamics at REAL state
```

The real dynamics advance along the REAL trajectory. The residual measures how far the GP's prediction (from the model state) is from where the real system actually went.

### 2. Trajectory-Level Comparison

Unlike Version L which compares at a single state, Version T compares across trajectories:
- GP starts from s_model[k]
- Real dynamics start from s_real[k]
- The residual measures the trajectory gap

### 3. DAgger Benefit

Even though the residual compares across trajectories, the NN still trains on model-trajectory states (standard DAgger). The key insight is that the label provides information about "where the real system would be" relative to "where the model thinks it is."

---

## Algorithm

```
For each DAgger round r = 0, 1, ..., R-1:
    1. Train NN ensemble on accumulated (input, residual) pairs
    2. For each rollout step k:
        a. tau = LQR(s_real[k])  # control from real state
        b. s_real[k+1] = real_step(s_real[k], tau)
        c. s_next_gp = GP(s_model[k], tau)
        d. residual = (s_real[k+1] - s_next_gp) / std  # trajectory gap
        e. delta_nn = NN(s_model[k], tau)
        f. s_model[k+1] = s_next_gp + scale * delta_nn * std
```

---

## When Version T Is Better

Based on experimental results (LOCAL_VS_TRAJECTORY_RESULTS.json):

| Condition | L MAE | T MAE | Winner |
|-----------|-------|-------|--------|
| Mode B (Open-loop 100 steps) | 0.777 | **0.458** | T (40%) |
| Mode C (Mixed 100 steps) | 0.777 | **0.458** | T (40%) |
| Initial: large_phi | 1.796 | **0.517** | T (3.5x) |
| Initial: high_vel | 3.070 | **0.683** | T (4.5x) |
| Initial: combined | 2.661 | **0.651** | T (4.1x) |
| Per-state delta | 0.512 | **0.197** | T (2.6x) |
| Per-state delta_dot | 1.219 | **0.465** | T (2.6x) |

Version T excels when:
- Multi-step open-loop prediction is needed
- Initial conditions vary significantly
- The model must capture long-horizon dynamics
- Steering-related states (delta, delta_dot) are important

---

## When Version T Is Worse

| Condition | L MAE | T MAE | Winner |
|-----------|-------|-------|--------|
| Mode A (Teacher Forcing) | **0.0023** | 0.0320 | L (14x) |
| Mode D (Closed-loop) | **0.0039** | 0.1294 | L (33x) |
| OOD extreme_phi | **0.0038** | 0.0409 | L (11x) |
| Action: small_sine | **0.0238** | 0.2020 | L (8.5x) |

Version T struggles when:
- Single-step prediction accuracy is paramount
- A feedback controller handles error correction
- The system stays within training distribution
- Actions are smooth and small

---

## Relationship to Original Code

The original `GPEEnsemble._collect_dagger_data()` (eval_models.py lines 352-379) was a corrupted variant:
```
r_k = (s_real[k+1] - GP(GP(s_model[k], tau), tau)) / delta_std
```

Version T simplifies this to the clean standard form:
```
r_k = (s_real[k+1] - GP(s_model[k], tau)) / delta_std
```

The two-step GP evaluation in the original is removed, making the residual interpretable and well-defined.

---

## Confidence Level

**CONFIRMED**: Implementation directly follows from the mathematical formula.
Code trace available in `version_t_trajectory_residual.py` lines 196-230.
