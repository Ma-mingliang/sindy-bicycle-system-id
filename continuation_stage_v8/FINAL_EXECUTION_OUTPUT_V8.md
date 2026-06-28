# V8 7D Bicycle World Model — Final Execution Output

**Date**: 2026-06-26
**Version**: V8 (7D real bicycle data)
**State**: [e_y, e_psi, v, theta, theta_dot, delta, delta_dot] — NO kappa
**Data**: stage2_dataset_150k.npz (150k samples, 59 episodes, real bicycle sensor data)

---

## 1. Executive Summary

V8 builds a 7D world model on real bicycle sensor data. **12 methods** were trained and evaluated across single-step accuracy, multi-step rollout stability, uncertainty calibration, and MPC planning.

**Key findings:**
- **Best single-step**: GP (NMAE=0.0017), but results may be inflated by suspicious zero errors on e_y/e_psi
- **Best multi-step**: Neural ODE (NMAE=0.407 at H=100, 5x better than GP)
- **GP Ensemble**: Significant multi-step improvement (GP Ens-5: NMAE=0.632 at H=100 vs GP: 1.980)
- **MPC Planning**: Not viable — near-zero ranking correlation (Spearman rho=0.006)
- **Uncertainty**: 51x too small, requires post-hoc calibration
- **V8 vs V6**: 83-413x worse at H=50 (real data vs simulated data)

---

## 2. Data Audit (Agent A)

| Metric | Value |
|--------|-------|
| Total samples | 150,000 |
| State dimension | 7D (kappa dropped — constant zero) |
| Episodes | 59 actual (declared 148 — metadata discrepancy) |
| Train/Test split | 106,214 / 43,786 (71%/29%) |
| NaN/Inf values | None |
| Continuity errors | 89 at segment boundaries (max diff ~1.03) |

**State statistics:**

| State | Std | Range | Notes |
|-------|-----|-------|-------|
| e_y (m) | 0.241 | [-1.5, 1.5] | Lateral error |
| e_psi (rad) | 0.537 | [-1, 1] | Heading error, concentrated at 0 |
| v (m/s) | 0.013 | [0.54, 0.66] | Very low variance — trivially predictable |
| theta (rad) | 0.569 | [-2.5, 0.5] | Road inclination |
| theta_dot (rad/s) | 0.193 | [-1, 1] | Angular velocity |
| delta (rad) | 0.572 | [-1.5, 0.5] | Steering angle |
| delta_dot (rad/s) | 0.566 | [-3, 2] | Steering rate |

**Concerns:**
- Speed (v) has extremely low variance (std=0.013), making it trivially predictable and inflating overall NMAE
- 89 continuity errors at segment boundaries may cause incorrect transition learning
- Episode count discrepancy (148 declared vs 59 actual) does not affect training

---

## 3. Interface Verification (Agent B)

**Result: PASS (10/10 tests)**

All 6 baseline models (GP, E1, RDE-L, RDE-T, RDE-M, SINDy) produce correct (7,) predictions. Extended models (GPEnsemble, AdaptiveResidualGP, NeuralODE, NeuralODEMultiStep) also verified.

---

## 4. Single-Step Accuracy

### 4.1 Core Methods (Agent C, 3000 eval samples)

| Method | NMAE | MAE | Train (s) | Notes |
|--------|------|-----|-----------|-------|
| GP | 0.0321 | 0.0131 | 16.0 | Best overall |
| E1 (GP+NN) | 0.0309 | 0.0123 | 218.4 | Best residual |
| RDE-L (local) | 0.0310 | 0.0123 | 190.7 | ≈ E1 |
| RDE-T (trajectory) | 0.0986 | 0.0413 | 216.0 | Worse — trajectory sync hurts |
| RDE-M (DAgger) | NaN | NaN | 242.7 | **FAILED** — diverged |
| SINDy | 0.0401 | 0.0161 | 0.23 | Fastest training |

**Per-state NMAE (Agent C):**

| State | GP | E1 | SINDy | Hardest? |
|-------|----|----|-------|----------|
| e_y | 0.0836 | 0.0906 | 0.1344 | Yes |
| e_psi | 0.1240 | 0.1138 | 0.1310 | Yes |
| v | 1.55e-6 | 4.29e-5 | 1.51e-6 | Trivial |
| theta | 0.000247 | 0.000130 | 0.001278 | No |
| theta_dot | 0.01213 | 0.01110 | 0.006297 | No |
| delta | 0.000497 | 0.000295 | 0.002671 | No |
| delta_dot | 0.004249 | 0.000544 | 0.005141 | No |

### 4.2 Extended Methods (Agent H, 10000 eval samples)

| Method | NMAE | MAE | Train (s) |
|--------|------|-----|-----------|
| GP (2000) | 0.00166 | 0.000805 | 218.5 |
| GP Ensemble 5 | 0.00180 | 0.000839 | 1383.8 |
| GP Ensemble 7 | 0.00174 | 0.000816 | 2674.6 |
| Adaptive Residual | 0.00748 | 0.003176 | 904.9 |
| E1 | 0.00998 | 0.004138 | 685.4 |
| RDE-L | 0.00998 | 0.004138 | 759.1 |
| Neural ODE | 0.02769 | 0.011379 | 144.2 |
| Neural ODE MultiStep | 0.03602 | 0.014345 | 308.6 |

**Note:** Agent C and Agent H report different NMAE for GP (0.0321 vs 0.00166) — 20x discrepancy likely due to different evaluation subsets. Agent H's GP shows e_y=0.0 and e_psi=0.0 NMAE which is suspicious.

### 4.3 GP Sample Size Comparison

| Config | NMAE | Train (s) | Notes |
|--------|------|-----------|-------|
| GP (2000 samples) | 0.00166 | 218 | Current default |
| GP (5000 samples) | 0.00147 | 1294 | 11% better, 6x slower |

**Conclusion:** GP 5000 is not worth the 6x cost for 11% improvement. Use GP 2000.

---

## 5. Multi-Step Rollout (Mode B — Open Loop)

### 5.1 Core Methods (Agent E, 5 segments x 200 steps)

| Method | H=1 | H=5 | H=10 | H=20 | H=50 | H=100 | H=200 |
|--------|-----|-----|------|------|------|-------|-------|
| GP | 0.0141 | 0.0435 | 0.0834 | 0.3265 | 1.1155 | 1.4605 | 1.6411 |
| E1 | 0.0159 | 0.0489 | 0.0907 | 0.5053 | 1.0477 | 1.4540 | 1.6973 |
| RDE-L | 0.0159 | 0.0489 | 0.0907 | 0.5053 | 1.0477 | 1.4540 | 1.6973 |

### 5.2 Extended Methods (Agent H, 3 segments x 200 steps)

| Method | H=1 | H=5 | H=10 | H=20 | H=50 | H=100 |
|--------|-----|-----|------|------|------|-------|
| GP | 0.0131 | 0.0427 | 0.0890 | 0.1895 | 1.3273 | 1.9800 |
| GP Ensemble 5 | 0.0131 | 0.0426 | 0.0886 | 0.1640 | **0.5073** | **0.6318** |
| GP Ensemble 7 | 0.0130 | 0.0426 | 0.0887 | 0.1641 | 0.5665 | 0.7238 |
| Adaptive Residual | 0.0098 | 0.0301 | 0.0656 | 0.2867 | 1.0072 | 1.2833 |
| E1 | 0.0093 | 0.0316 | 0.0693 | 0.1702 | 1.1690 | 3.0328 |
| RDE-L | 0.0093 | 0.0316 | 0.0693 | 0.1702 | 1.1690 | 3.0328 |
| **Neural ODE** | **0.0030** | **0.0106** | **0.0237** | **0.1195** | **0.3012** | **0.4071** |
| Neural ODE MS | 0.0041 | 0.0229 | 0.0557 | 0.2015 | 0.5481 | 0.7335 |

### 5.3 Multi-Step Rankings by Horizon

**Short horizon (H=1-10):** Neural ODE > Neural ODE MS > Adaptive Residual > E1/RDE-L > GP

**Medium horizon (H=20-50):** Neural ODE > GP Ensemble 5 > GP Ensemble 7 > Neural ODE MS > GP > Adaptive Residual > E1

**Long horizon (H=100):** Neural ODE (0.407) > GP Ensemble 5 (0.632) > GP Ensemble 7 (0.724) > Neural ODE MS (0.734) > Adaptive Residual (1.283) > GP (1.980) > E1 (3.033)

### 5.4 V8 7D vs V6 4D Comparison

| Horizon | V8 7D GP | V6 4D GP | Ratio |
|---------|----------|----------|-------|
| H=1 | 0.0141 | 0.0001 | 141x |
| H=10 | 0.0834 | 0.0004 | 209x |
| H=50 | 1.1155 | 0.0027 | 413x |
| H=100 | 1.4605 | 0.0195 | 75x |
| H=200 | 1.6411 | 1.77 | 0.93x |

V8 7D is 83-413x worse than V6 4D at short/medium horizons. At H=200, both diverge similarly.

---

## 6. Uncertainty Calibration (Agent D)

**Model evaluated:** E1 (GP + offline NN residual), 5-model ensemble

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Pearson r (overall) | 0.413 | Moderate correlation — direction informative |
| Optimal scalar scale | 51.0 | Uncertainties are **51x too small** |
| Coverage at scale=51 | 0.798 | After scaling, 80% coverage |
| ECE (overall) | 0.163 | Poor calibration (ideal=0) |
| MACD | 0.980 | High calibration error |

**Per-state calibration:**

| State | Pearson r | Optimal Scale | ECE |
|-------|-----------|---------------|-----|
| e_y | 0.510 | 13.9 | 0.074 |
| e_psi | 0.292 | 22.8 | 0.129 |
| v | 0.021 | 7.4 | 0.328 |
| theta | 0.397 | 4.0 | 0.078 |
| theta_dot | 0.674 | 89.7 | 0.393 |
| delta | 0.661 | 5.7 | 0.063 |
| delta_dot | 0.487 | 10.2 | 0.079 |

**OOD Detection:** BROKEN — 100% of test samples classified as OOD (criterion too aggressive).

---

## 7. MPC Planning (Agent F)

| Metric | H=10 | H=20 |
|--------|------|------|
| Spearman rho | 0.006 | -0.004 |
| Kendall tau | 0.003 | -0.007 |
| Top-K overlap | 0.08 | 0.12 |
| MAE ratio | 0.353 | 0.438 |

**Conclusion:** Planning is **NOT viable**. Near-zero ranking correlation means the model cannot distinguish good trajectories from bad ones. Root cause: multi-step prediction errors compound too rapidly (NMAE=0.09 at H=10, 0.33 at H=20).

---

## 8. Extended Methods Analysis (Agent H)

### 8.1 GP Ensemble (Bootstrap)

| Config | Single-step NMAE | H=50 NMAE | H=100 NMAE | Train (s) |
|--------|-----------------|-----------|------------|-----------|
| GP (single) | 0.00166 | 1.327 | 1.980 | 218 |
| GP Ensemble 5 | 0.00180 | **0.507** | **0.632** | 1384 |
| GP Ensemble 7 | 0.00174 | 0.567 | 0.724 | 2675 |

**Finding:** Ensembles provide **no single-step improvement** but **significant multi-step improvement** (2.6-3.1x at H=50-100). Worth the cost for multi-step applications.

### 8.2 Adaptive Residual GP (OOD-aware)

- Single-step NMAE: 0.00748 (best non-GP method)
- H=50 NMAE: 1.007 (similar to GP)
- Verdict: Marginal improvement over simpler methods

### 8.3 Neural ODE

- Single-step NMAE: 0.02769 (16.7x worse than GP)
- H=100 NMAE: **0.407** (4.9x better than GP)
- Train time: 144s (fastest NN method)
- **Verdict: Best method for multi-step prediction**

### 8.4 Neural ODE MultiStep

- Single-step NMAE: 0.03602 (worse than standard Neural ODE)
- H=100 NMAE: 0.734 (worse than standard Neural ODE)
- **Verdict: Multi-step training objective does NOT help — standard Neural ODE is better at all horizons**

---

## 9. Issues Found (Agent G Review)

### CRITICAL

1. **E1 and RDE-L produce identical results** — RDE-L with dagger_rounds=0 is mathematically equivalent to E1. Invalidates E1 vs RDE-L comparison.

2. **GP reports e_y=0.0, e_psi=0.0 NMAE** — Physically implausible. Likely data leakage or evaluation bug in Agent H.

### HIGH

3. **Agent C vs Agent H NMAE discrepancy** — GP NMAE=0.0321 (C) vs 0.00166 (H), 20x difference. Different evaluation protocols.

4. **RDE-M diverged to NaN** — DAgger training instability. One of 6 baseline models non-functional.

5. **OOD detection broken** — 100% test samples classified as OOD. Criterion too aggressive.

### MEDIUM

6. Uncertainties 51x too small — requires post-hoc calibration
7. MPC planning not viable — Spearman rho=0.006
8. Neural ODE MultiStep degrades vs standard Neural ODE

---

## 10. Final Rankings

### Single-Step Accuracy

| Rank | Method | NMAE | Train (s) |
|------|--------|------|-----------|
| 1 | GP (2000) | 0.0017* | 218 |
| 2 | GP Ensemble 7 | 0.0017 | 2675 |
| 3 | GP Ensemble 5 | 0.0018 | 1384 |
| 4 | Adaptive Residual | 0.0075 | 905 |
| 5 | E1 / RDE-L | 0.0100 | 685 |
| 6 | Neural ODE | 0.0277 | 144 |
| 7 | SINDy | 0.0401 | 0.2 |

*Results may be inflated — see issues section

### Multi-Step Stability (H=100)

| Rank | Method | NMAE | Notes |
|------|--------|------|-------|
| 1 | **Neural ODE** | **0.407** | Clear winner |
| 2 | GP Ensemble 5 | 0.632 | Best GP variant |
| 3 | GP Ensemble 7 | 0.724 | |
| 4 | Neural ODE MS | 0.734 | |
| 5 | Adaptive Residual | 1.283 | |
| 6 | GP | 1.980 | Diverged |
| 7 | E1 / RDE-L | 3.033 | Worst divergence |

### Practical Recommendation

| Use Case | Recommended Method | Reason |
|----------|-------------------|--------|
| Single-step prediction | GP (2000 samples) | Best accuracy, reasonable speed |
| Multi-step rollout (H≥5) | Neural ODE | 5x better stability, fast training |
| Multi-step with uncertainty | GP Ensemble 5 | Best GP stability + uncertainty |
| Fast prototyping | SINDy | 0.2s training, interpretable |
| MPC planning | Not viable with current models | All methods have insufficient ranking correlation |

---

## 11. Generated Artifacts

| File | Description |
|------|-------------|
| `canonical_7d/config.py` | 7D configuration (frozen dataclasses) |
| `canonical_7d/data_loader.py` | Data loading and episode splitting |
| `canonical_7d/models.py` | GP and SINDy models |
| `canonical_7d/residual_models.py` | E1, RDE-L, RDE-T, RDE-M |
| `canonical_7d/extended_models.py` | GPEnsemble, AdaptiveResidual, NeuralODE |
| `canonical_7d/evaluation_modes.py` | Mode A/B/C evaluation |
| `canonical_7d/metrics.py` | NMAE, cost computation |
| `canonical_7d/runner.py` | Training and evaluation pipeline |
| `raw_results/AGENT_A_DATA_AUDIT.json` | Data quality audit |
| `raw_results/AGENT_B_INTERFACE_VERIFY.json` | Interface test results |
| `raw_results/AGENT_C_TRAINING.json` | Core model training results |
| `raw_results/AGENT_D_UNCERTAINTY.json` | Uncertainty calibration |
| `raw_results/AGENT_E_MULTISTEP.json` | Multi-step rollout (core) |
| `raw_results/AGENT_F_PLANNING.json` | MPC planning evaluation |
| `raw_results/AGENT_G_REVIEW.json` | Independent review |
| `raw_results/AGENT_H_EXTENDED.json` | Extended methods comparison |

---

## 12. Confidence Assessment

| Finding | Confidence | Reason |
|---------|-----------|--------|
| Neural ODE best multi-step | HIGH | Consistent across all horizons |
| GP best single-step | MEDIUM | Suspicious zero errors, 20x cross-agent discrepancy |
| Ensembles help multi-step | HIGH | Clear 2.6-3.1x improvement at H=50-100 |
| Planning not viable | HIGH | Near-zero ranking correlation |
| Uncertainties poorly calibrated | HIGH | 51x scale factor consistently needed |
| E1=RDE-L identity | HIGH | Identical to 6 decimal places |
| V8 >> V6 difficulty | HIGH | 83-413x worse at H=50 |

**Overall confidence: MEDIUM** — Qualitative findings (Neural ODE best multi-step, ensembles help) are reliable. Quantitative rankings have inconsistencies that reduce precision.
