# GP Improvement Report

> **Generated**: 2026-06-24
> **Project**: sindy_bicycle (D:\系统辨识作业\sindy_bicycle)
> **Purpose**: GP-based improvement methods for bicycle dynamics identification

---

## 1. Executive Summary

This report documents GP-based improvement methods for the sindy_bicycle project. Key findings:

1. **The 0.064 rad result is reproducible** with the correct hyperparameters (7 models, 5 DAgger rounds)
2. **Pure GP methods diverge** after ~100 steps - NN residual ensemble is essential
3. **DAgger is critical** for reducing distribution shift and improving long-term accuracy

---

## 2. Baseline Results

### 2.1 GP-Only Baseline (GP-B0) - CONFIRMED

| Step | Mean MAE (rad) | Std |
|------|----------------|-----|
| 1 | 0.00000 | 0.00000 |
| 10 | 0.00003 | 0.00000 |
| 50 | 0.00040 | 0.00004 |
| 100 | 0.00422 | 0.00027 |
| 200 | 0.69306 | 0.09572 |
| **500** | **4.81 ± 1.10** | Diverges |

**Conclusion**: Pure GP methods cannot maintain long-term stability. Accurate for short-term (step 100: 0.004 rad) but diverges catastrophically after ~150 steps.

### 2.2 All GP Methods Comparison

| Method | 500-step MAE (rad) | Training Time | Status |
|--------|-------------------|---------------|--------|
| GP-B0: Standard GP | 4.81 ± 1.10 | ~4000s | CONFIRMED |
| GP-B1: ARD GP | 14.66 ± 2.07 | ~5400s | CONFIRMED |
| GP-B2: Composite Kernel | 298.81 ± 29.00 | ~2200s | CONFIRMED |
| GP-B3: Physics Residual | 2709 ± 925 | ~670s | CONFIRMED |
| **GP-B4: Sparse/Local** | **0.42 ± 0.08** | **~12s** | **CONFIRMED** |
| GP-B5: Bootstrap Ensemble | 6.21 ± 0.26 | ~5000s | CONFIRMED |
| GP-B6: Uncertainty-Gated | 2.28 ± 0.31 | ~1700s | CONFIRMED |

### 2.3 GP + NN Residual Ensemble

| Configuration | 500-step MAE | Notes |
|---------------|--------------|-------|
| 5 models, 3 DAgger | 0.15-0.25 rad | Standard config |
| 7 models, 3 DAgger | 0.37 rad | More models, worse |
| 9 models, 3 DAgger | 0.18 rad | Diminishing returns |
| 5 models, 5 DAgger | 0.11 rad | More DAgger helps |
| **7 models, 5 DAgger** | **0.066 rad** | **Best config** |

**Source**: test_ensemble_optimization.py output

---

## 3. Key Findings

### 3.1 The 0.064 Result Reproduction

The reported 0.064 rad MAE was achieved with:
- **n_models = 7** (not the standard 5)
- **dagger_rounds = 5** (not the standard 3)
- **residual_scale = 0.3** (standard)

This configuration achieves **0.06600 rad**, which matches the reported 0.064 within experimental variance.

### 3.2 Why Pure GP Diverges

GP-only methods diverge because:
1. GP extrapolates poorly outside training distribution
2. Small errors accumulate over 500 steps
3. No mechanism to correct drift

### 3.3 Why NN Residual Ensemble Alone Doesn't Work

GP+NN Ensemble without DAgger also diverges (5.41 rad) because:
1. Training data is from random uniform sampling of state space
2. Evaluation data is from closed-loop trajectories
3. **Distribution shift** causes NN to fail on evaluation data

### 3.4 Why DAgger is Critical

DAgger fixes the distribution shift problem:
1. Collects training data from model's own predictions
2. NN learns to correct residuals in the distribution it will see at test time
3. Each DAgger round reduces the train-test distribution gap
4. More rounds = better alignment = better performance

---

## 4. Improvement Methods Tested

### 4.1 GP-B0: Baseline GP
- Standard GP with RBF kernel
- Result: Diverges after ~100 steps

### 4.2 GP-B1: ARD GP - CONFIRMED
- Automatic Relevance Determination
- Separate length scale per dimension
- Result: **14.66 ± 2.07 rad** (worse than standard GP)
- ARD downweights delta (56,390) and delta_dot (23,643), hurting performance
- Standard GP with uniform length scale is better

### 4.3 GP-B2: Composite Kernel - CONFIRMED
- RBF + Matérn + WhiteKernel
- Result: **298.81 ± 29.00 rad** (catastrophic divergence)
- Matérn kernel learns very large length scales (~100,000), effectively becoming constant
- Composite kernel overfits training data and fails to generalize

### 4.4 GP-B3: Physics Residual - CONFIRMED
- Uses Meijaard dynamics as prior
- GP learns only the residual
- Result: **2709 ± 925 rad** (complete divergence)
- Physics prior introduces systematic errors that GP residual cannot correct
- Errors accumulate exponentially over 500 steps

### 4.5 GP-B4: Sparse/Local GP - CONFIRMED
- Nystroem approximation (500 components)
- Result: **0.42 ± 0.08 rad** (best pure GP method!)
- 10x better than standard GP baseline (4.81 rad)
- 100x faster training (~12s vs ~4000s)
- Saturates around step 50, doesn't diverge further

### 4.6 GP-B5: Bootstrap Ensemble - CONFIRMED
- Multiple GPs with bootstrap sampling
- Result: **6.21 ± 0.26 rad** (diverges)
- Uncertainty estimate: 0.053 rad (severely underestimated - 120x lower than actual error)

### 4.7 GP-B6: Uncertainty-Gated - CONFIRMED
- Gates residual by GP uncertainty
- Gate value: 1.0 → 0.018 as uncertainty increases
- Result: **2.28 ± 0.31 rad** (better than GP-only, worse than DAgger)
- The gate prevents divergence but limits accuracy

---

## 5. Recommendations

### 5.1 For Best Performance (GP + NN + DAgger)

Use the **optimized configuration**:
```python
n_models = 7
dagger_rounds = 5
residual_scale = 0.3
```

This achieves **0.066 rad** MAE at 500 steps.

### 5.2 For Best Pure GP Method

Use **GP-B4: Sparse/Local GP** with Nystroem approximation:
- **0.42 ± 0.08 rad** at 500 steps
- **12 seconds** training time (100x faster than standard GP)
- Best trade-off between accuracy and computational cost

### 5.3 For Faster Training (GP + NN)

Use the standard configuration:
```python
n_models = 5
dagger_rounds = 3
residual_scale = 0.3
```

This achieves **0.15-0.25 rad** MAE at 500 steps.

### 5.4 For Uncertainty Estimation

Use the bootstrap ensemble (GP-B5) for uncertainty estimates, but combine with NN residual for accuracy. Note: GP-B5 severely underestimates uncertainty (0.053 rad estimated vs 6.21 rad actual).

---

## 6. File Structure

```
gp_improvement/
├── gp_b0_baseline.py           # Baseline GP
├── gp_b1_ard.py                # ARD GP
├── gp_b2_composite_kernel.py   # Composite kernel GP
├── gp_b3_physics_residual.py   # Physics residual GP
├── gp_b4_sparse_local.py       # Sparse/Local GP
├── gp_b5_bootstrap_ensemble.py # Bootstrap ensemble GP
├── gp_b6_uncertainty_gated.py  # Uncertainty-gated GP
├── run_all_gp_improvements.py  # Runner script
├── GP_IMPROVEMENT_REPORT.md    # This report
└── GP_IMPROVEMENT_RESULTS.json # Raw results
```

---

## 7. Conclusion

1. **The 0.064 result is reproducible** with 7 models + 5 DAgger rounds
2. **Pure GP methods are insufficient** for long-term stability - best pure GP (Sparse/Local) achieves only 0.42 rad
3. **DAgger is critical** for long-term accuracy - without it, GP+NN diverges (5.41 rad)
4. **The key insight**: GP provides good short-term predictions, NN residual corrects for long-term drift
5. **GP-B4 (Sparse/Local) is the best pure GP method** - 10x better than standard GP, 100x faster
6. **Physics-informed GP (B3) and Composite Kernel (B2) fail catastrophically** - do not use

---

*Report generated by Claude Code on 2026-06-24*
