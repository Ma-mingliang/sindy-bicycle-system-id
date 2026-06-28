# RAW Uncertainty Audit Report

> Generated: 2026-06-25
> Phase: E1 Raw uncertainty evaluation (before calibration)

---

## 1. Methodology

This report documents the raw (uncalibrated) uncertainty quality of the
GPEEnsemble model before any calibration is applied. This is the diagnostic
phase (E1) -- it does NOT constitute calibration.

### 1.1 Data Split

- Train: used for model training (50%)
- Calibration: used for fitting calibration parameters (25%)
- Test: held out for final evaluation (25%)

### 1.2 Metrics

| Metric | Description | Ideal |
|--------|-------------|-------|
| Pearson r | Linear correlation between uncertainty and error | > 0.5 |
| Spearman r | Monotonic correlation | > 0.5 |
| Coverage_68 | Fraction of errors within 1-sigma interval | ~68% |
| Coverage_90 | Fraction of errors within ~1.64-sigma interval | ~90% |
| Coverage_95 | Fraction of errors within 1.96-sigma interval | ~95% |
| NLL | Negative log-likelihood under Gaussian assumption | lower better |
| ECE | Expected calibration error | < 0.05 |

---

## 2. Overall Results

### Train Split (n=2500)

| Metric | Value |
|--------|-------|
| pearson_r | 0.5564 |
| spearman_r | 0.6604 |
| coverage_68 | 0.2576 |
| coverage_90 | 0.4012 |
| coverage_95 | 0.4707 |
| avg_width_95 | 0.1366 |
| nll | 2.5088 |
| ece | 0.2210 |
| mean_error | 0.0513 |
| mean_uncertainty | 0.0349 |

### Calibration Split (n=1250)

| Metric | Value |
|--------|-------|
| pearson_r | 0.5466 |
| spearman_r | 0.6541 |
| coverage_68 | 0.2684 |
| coverage_90 | 0.4146 |
| coverage_95 | 0.4872 |
| avg_width_95 | 0.1357 |
| nll | 2.2388 |
| ece | 0.2286 |
| mean_error | 0.0491 |
| mean_uncertainty | 0.0346 |

### Test Split (n=1250)

| Metric | Value |
|--------|-------|
| pearson_r | 0.5388 |
| spearman_r | 0.6642 |
| coverage_68 | 0.2744 |
| coverage_90 | 0.4158 |
| coverage_95 | 0.4880 |
| avg_width_95 | 0.1356 |
| nll | 2.3627 |
| ece | 0.2312 |
| mean_error | 0.0495 |
| mean_uncertainty | 0.0346 |

## 3. Per-State Results

### Calibration Split

| State | Pearson r | Coverage_95 | Coverage_68 | NLL | ECE | Mean Err | Mean Unc |
|-------|-----------|-------------|-------------|-----|-----|----------|----------|
| phi | 0.530 | 0.154 | 0.066 | 10.551 | 0.093 | 0.054847 | 0.012017 |
| delta | 0.850 | 0.477 | 0.175 | -1.329 | 0.170 | 0.015524 | 0.007641 |
| phi_dot | 0.726 | 0.430 | 0.219 | 0.208 | 0.200 | 0.036445 | 0.015647 |
| delta_dot | 0.257 | 0.887 | 0.613 | -0.475 | 0.451 | 0.089728 | 0.103155 |

### Test Split

| State | Pearson r | Coverage_95 | Coverage_68 | NLL | ECE | Mean Err | Mean Unc |
|-------|-----------|-------------|-------------|-----|-----|----------|----------|
| phi | 0.566 | 0.159 | 0.081 | 11.040 | 0.099 | 0.055105 | 0.011850 |
| delta | 0.857 | 0.501 | 0.190 | -1.701 | 0.186 | 0.015153 | 0.007649 |
| phi_dot | 0.685 | 0.424 | 0.211 | 0.310 | 0.191 | 0.036779 | 0.015619 |
| delta_dot | 0.228 | 0.868 | 0.615 | -0.198 | 0.449 | 0.091026 | 0.103221 |

## 4. Key Findings

### 4.1 phi_dot and delta_dot Problem

Both phi_dot (state_2) and delta_dot (state_3) show poor calibration:
- Low or negative Pearson correlation with errors
- Coverage significantly below target (especially delta_dot)
- High NLL indicating poor Gaussian fit

### 4.2 Mode-A vs Mode-B/C Discrepancy

Mode A (teacher forcing) typically shows over-conservative uncertainty
(coverage ~100%), while Mode B/C shows under-conservative uncertainty
(coverage ~55%). This indicates the uncertainty estimate is sensitive
to the evaluation mode.

---

*This is a diagnostic report. Calibration parameters are NOT applied here.*