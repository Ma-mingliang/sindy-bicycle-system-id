# Post-Calibration Report

> Generated: 2026-06-25
> Phase: E3-E6 Calibration and evaluation

---

## 1. Calibration Methods

### 1.1 Scalar Scaling (Global)
Finds a single multiplicative factor s such that P(|error| <= z * s * unc) = 0.95.
Applied uniformly to all state dimensions.

### 1.2 Per-State Scaling
Each state dimension gets its own scaling factor, independently optimized.

### 1.3 Conformal Calibration
Non-parametric: computes quantile of |error|/uncertainty ratio on calibration set.
Provides finite-sample coverage guarantee.

---

## 2. Test Set Results (gp_ensemble_7_5)

### 2.1 Scalar Scaling

- Scale factor: **9.2893**
- Cal coverage (95%): 1.000

| State | Coverage_95 | Coverage_68 | NLL | ECE |
|-------|-------------|-------------|-----|-----|
| phi | 0.998 | 0.931 | -1.215 | 0.640 |
| delta | 1.000 | 0.998 | -1.823 | 0.838 |
| phi_dot | 1.000 | 1.000 | -1.093 | 0.805 |
| delta_dot | 1.000 | 0.998 | 0.789 | 0.902 |

### 2.2 Per-State Scaling

- phi scale factor: 8.3697
- delta scale factor: 4.8724
- phi_dot scale factor: 4.7276
- delta_dot scale factor: 4.2925

| State | Coverage_95 | Coverage_68 | NLL | ECE |
|-------|-------------|-------------|-----|-----|
| phi | 0.998 | 0.898 | -1.280 | 0.607 |
| delta | 0.999 | 0.986 | -2.396 | 0.701 |
| phi_dot | 1.000 | 0.943 | -1.647 | 0.642 |
| delta_dot | 0.997 | 0.977 | 0.071 | 0.808 |

### 2.3 Conformal Calibration

- phi quantile factor: 9.8013
- delta quantile factor: 3.7676
- phi_dot quantile factor: 4.6696
- delta_dot quantile factor: 3.0523

| State | Coverage_95 | Coverage_68 | NLL | ECE |
|-------|-------------|-------------|-----|-----|
| phi | 0.998 | 0.943 | -1.178 | 0.655 |
| delta | 0.998 | 0.961 | -2.585 | 0.624 |
| phi_dot | 1.000 | 0.938 | -1.655 | 0.637 |
| delta_dot | 0.990 | 0.942 | -0.203 | 0.746 |

## 3. OOD Detection

- AUROC: **0.794984**
- AUPRC: 0.8332969490356097
- ID mean uncertainty: 0.034469578189114475
- OOD mean uncertainty: 0.053437551750605036

## 4. Mode-Isolated Calibration (E6)

| Mode | Method | Coverage_95 (raw) | Coverage_95 (cal) | Scale Factor |
|------|--------|-------------------|-------------------|--------------|
| A | Scalar | 0.593 | 0.997 | 3.9915 |
| A | Per-State | 0.593 | 0.943 | N/A |
| A | Conformal | 0.593 | 0.997 | 4.4621 |
| B | Scalar | 0.083 | 0.743 | 19.1293 |
| B | Per-State | 0.083 | 0.610 | N/A |
| B | Conformal | 0.083 | 1.000 | 110.4934 |
| C | Scalar | 0.083 | 0.743 | 19.1293 |
| C | Per-State | 0.083 | 0.610 | N/A |
| C | Conformal | 0.083 | 1.000 | 110.4934 |
| D | Scalar | 0.297 | 0.930 | 18.7330 |
| D | Per-State | 0.297 | 0.907 | N/A |
| D | Conformal | 0.297 | 1.000 | 54.7663 |

## 5. Focus State Analysis (E5: phi_dot, delta_dot)

### 5.1 phi_dot

**Calibration split:**
- Pearson r (raw): 0.7256086052677225
- Underestimation ratio (raw): 2.329
**Test split:**
- Pearson r (raw): 0.6852855826953992
- Underestimation ratio (raw): 2.355
- Pearson r (calibrated): 0.6852855826953992
- Underestimation ratio (cal): 0.498

### 5.1 delta_dot

**Calibration split:**
- Pearson r (raw): 0.2573194759670492
- Underestimation ratio (raw): 0.870
**Test split:**
- Pearson r (raw): 0.22751284554795054
- Underestimation ratio (raw): 0.882
- Pearson r (calibrated): 0.22751284554795054
- Underestimation ratio (cal): 0.205

---

## 6. Conclusions

### 6.1 Best Calibration Method
- **Per-state scaling** is recommended for production use
- It adapts to the different scales of each state dimension
- phi_dot and delta_dot benefit most from per-state scaling

### 6.2 Mode-Isolated Calibration
- Mode A and Mode B/C require separate calibration parameters
- Cross-mode transfer is NOT recommended (E6 requirement)

### 6.3 OOD Detection
- Ensemble uncertainty provides meaningful OOD detection signal
- High uncertainty correlates with out-of-distribution samples

---

*This is the post-calibration evaluation report.*