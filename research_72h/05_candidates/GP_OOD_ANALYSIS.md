# GP OOD Analysis Report

**Generated**: 2026-06-28 20:44:45
**Experiment**: GP Out-of-Distribution Generalization Test
**Dataset**: stage2_dataset_150k.npz (150k samples, 58 episodes)

---

## Executive Summary

Key findings:

- **GP ID NMAE (H=100)**: 0.2598
- **GP OOD NMAE (H=100)**: 0.6090
- **OOD Degradation**: 134.4%
- **OOD Detection F1**: 0.9656
- **GP Training Time**: 51.6s

---

## 1. OOD Detection Performance

Z-score based OOD detector evaluated on synthetic OOD samples (states perturbed beyond 3-sigma from training distribution).

| Metric | Value |
|--------|-------|
| True Positives | 1991 |
| False Positives | 133 |
| True Negatives | 1867 |
| False Negatives | 9 |
| **Precision** | 0.9374 |
| **Recall** | 0.9955 |
| **F1 Score** | 0.9656 |
| Accuracy | 0.9645 |

### Optimal Threshold

- Best F1 threshold: **3.34** (F1=0.9756, P=0.9731, R=0.9780)

---

## 2. GP Performance: In-Distribution vs Out-of-Distribution

| Horizon | GP-ID NMAE | GP-OOD NMAE | ID Survival | OOD Survival | Degradation |
|---------|-----------|------------|------------|-------------|-------------|
| H=1 | 0.0232 | 0.0312 | 100% | 100% | 34.4% |
| H=10 | 0.1418 | 0.1603 | 100% | 100% | 13.1% |
| H=50 | 0.2620 | 0.4229 | 100% | 100% | 61.4% |
| H=100 | 0.2598 | 0.6090 | 100% | 100% | 134.4% |
| H=200 | 0.2553 | 0.8170 | 100% | 100% | 220.1% |
| H=500 | 0.2553 | 0.8170 | 100% | 100% | 220.1% |

---

## 3. Synthetic Perturbation Robustness

GP evaluated on test segments with controlled perturbations at H=100.
Perturbations simulate distribution shift in different state dimensions.

| Perturbation | mag=0.1 | mag=0.5 | mag=1.0 | mag=2.0 |
|---|---|---|---|---|
| **speed** | 0.3538 (100%) | 0.6716 (100%) | 0.2570 (100%) | 0.3360 (100%) |
| **lateral** | 0.2570 (100%) | 0.2580 (100%) | 0.3355 (100%) | 1.5089 (100%) |
| **heading** | 0.2570 (100%) | 0.2570 (100%) | 0.2570 (100%) | 0.2572 (100%) |
| **action** | 0.2570 (100%) | 0.2570 (100%) | 0.2570 (100%) | 0.2570 (100%) |
| **lean** | 0.3339 (100%) | 0.3953 (0%) | 0.4989 (0%) | 1.0429 (0%) |
| **combined** | 0.2572 (100%) | 0.2590 (100%) | 0.2768 (100%) | 0.9512 (100%) |

### Perturbation Impact Analysis

- **Speed**: Tests robustness to velocity changes (scaled v)
- **Lateral**: Tests robustness to lateral offset changes (e_y shift)
- **Heading**: Tests robustness to heading changes (e_psi shift)
- **Action**: Tests robustness to larger steering inputs
- **Lean**: Tests robustness to lean angle changes (theta shift)
- **Combined**: Multi-dimensional perturbation (e_y + e_psi + v + action)

- **Most robust** to: speed (ratio: 0.9x)
- **Least robust** to: lateral (ratio: 5.9x)

---

## 4. GP Uncertainty Calibration

Evaluates how well GP's predicted uncertainty correlates with actual prediction error.
Well-calibrated uncertainty is critical for safe OOD detection.

| Metric | Value |
|--------|-------|
| Uncertainty-Error Correlation | 0.4051 |
| Calibration Ratio (error/unc) | 0.2727 |
| Mean Uncertainty | 0.103089 |
| Mean Error | 0.028110 |
| Sample Pairs | 500 |

### Binned Calibration

| Bin | Uncertainty Range | Mean Uncertainty | Mean Error | Ratio | Count |
|-----|-------------------|-----------------|------------|-------|-------|
| 0 | [0.1031, 0.1031] | 0.103088 | 0.001276 | 0.0124 | 100 |
| 1 | [0.1031, 0.1031] | 0.103089 | 0.001505 | 0.0146 | 100 |
| 2 | [0.1031, 0.1031] | 0.103089 | 0.031491 | 0.3055 | 100 |
| 3 | [0.1031, 0.1031] | 0.103089 | 0.028729 | 0.2787 | 100 |
| 4 | [0.1031, 0.1031] | 0.103090 | 0.074365 | 0.7214 | 99 |

**Interpretation**: Moderate correlation -- GP uncertainty provides useful but imperfect error signal.

---

## 5. GP vs Neural ODE on OOD Data

Neural ODE model not available for comparison (PyTorch load failed).
To enable this comparison, ensure PyTorch is installed and the v9 model exists at:
`D:/系统辨识作业/sindy_bicycle/research_72h/07_models/v9_seed42.pt`

---

## 6. Natural OOD Segments in Test Data

- Natural OOD segments found: **68**
- Natural ID segments found: **1267**
- Top OOD scores: 3.25, 3.21, 3.20, 3.17, 3.14

---

## 7. GP Generalization Assessment

### Strengths

- High OOD detection accuracy -- z-score detector works well for this state space
- High recall -- rarely misses OOD samples
- Robust to speed perturbation (< 3x degradation at 2x magnitude)
- Robust to heading perturbation (< 3x degradation at 2x magnitude)
- Robust to action perturbation (< 3x degradation at 2x magnitude)

### Weaknesses

- Moderate uncertainty correlation (0.41) -- GP uncertainty is useful but imperfect for OOD detection
- Significant OOD degradation: 134% NMAE increase on OOD data at H=100
- Moderate degradation on lateral perturbation (5.9x at 2x magnitude)
- Moderate degradation on lean perturbation (3.1x at 2x magnitude)
- Moderate degradation on combined perturbation (3.7x at 2x magnitude)

### Recommendations

1. **Use GP uncertainty for OOD gating**: When GP variance exceeds a calibrated threshold, fall back to a physics baseline or reduce residual contribution.
2. **Combine with physics constraints**: GP predictions outside the training distribution should be blended with known dynamics (e.g., Meijaard model).
3. **Monitor state-space coverage**: Track which regions of the state space are well-covered by training data and flag predictions in sparse regions.
4. **Adaptive threshold**: The optimal OOD detection threshold should be calibrated on a validation set and updated as more data is collected.
5. **Ensemble for robustness**: Use multiple GP models trained on different subsets to detect when predictions disagree (high variance = likely OOD).

---

## Appendix: Experimental Setup

- **GP kernel**: RBF (length_scale=1.0) with ConstantKernel
- **Training samples**: 1000 (random subset from 101k training samples)
- **GP dimensions**: 7D state (e_y, e_psi, v, theta, theta_dot, delta, delta_dot)
- **Action dimension**: 1D steering torque
- **GP training time**: 51.6s
- **Dataset**: stage2_dataset_150k.npz (150k samples, 58 episodes)
- **Train/test split**: 75%/25% by episodes
- **OOD detector**: Z-score based, threshold=3.0
