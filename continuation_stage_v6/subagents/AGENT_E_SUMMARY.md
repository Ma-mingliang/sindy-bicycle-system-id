# Agent E: Uncertainty Calibration and OOD Detection Summary

## Experimental Setup

- **Model**: RDE-M (5-NN ensemble, 2 DAgger rounds)
- **Data split**: train=2500, cal=1250, test=1250, OOD=1000
- **OOD definition**: states with |phi| > 0.8 (outside training range [-0.5, 0.5])
- **Total runtime**: ~223s

## 1. Raw Uncertainty Assessment

### Pearson Correlation (|error| vs uncertainty)

| State    | r      | p-value   | Interpretation          |
|----------|--------|-----------|------------------------|
| phi      | 0.2064 | 1.7e-13   | Weak positive          |
| delta    | 0.4977 | 3.4e-79   | Moderate positive      |
| phi_dot  | 0.1938 | 4.8e-12   | Weak positive          |
| delta_dot| 0.2833 | 1.7e-24   | Weak-moderate positive |

All correlations are statistically significant (p < 0.05). The ensemble uncertainty does track prediction error to some degree, but the relationships are weak. Delta has the best correlation (0.50), while phi_dot has the weakest (0.19).

### Raw Coverage (fraction of |error| < z_alpha * sigma)

| Confidence | phi   | delta | phi_dot | delta_dot | All   |
|------------|-------|-------|---------|-----------|-------|
| 68%        | 0.104 | 0.130 | 0.453   | 0.469     | 0.017 |
| 90%        | 0.170 | 0.217 | 0.674   | 0.726     | 0.065 |
| 95%        | 0.204 | 0.262 | 0.754   | 0.825     | 0.098 |
| 99%        | 0.274 | 0.352 | 0.848   | 0.917     | 0.188 |

**Critical finding**: Raw uncertainty is severely underestimated. At the 95% confidence level:
- phi only achieves 20% coverage (should be 95%)
- delta achieves 26% coverage
- phi_dot and delta_dot are closer (75%, 82%) but still far from target
- Simultaneous all-state coverage is only 10%

The ensemble std from 5 NN members is too small to represent true prediction uncertainty.

## 2. Calibration Results

### Scalar Calibration

- **Optimal scale factor**: s = 5.0 (search range capped at 5.0)
- **Gap from 95% target**: 0.058

The raw uncertainty must be multiplied by 5x to achieve approximately 95% coverage. This indicates the ensemble diversity is roughly 5x too narrow.

### Per-State Calibration

| State    | Scale Factor | Gap from 95% |
|----------|-------------|-------------|
| phi      | 5.000 (capped) | 0.0516 |
| delta    | 3.892        | 0.0004 |
| phi_dot  | 2.144        | 0.0004 |
| delta_dot| 1.553        | 0.0004 |

phi requires the largest correction (5x, capped at search limit), indicating it is the hardest state to predict. delta_dot requires the smallest correction (1.55x), suggesting the ensemble is most informative for this state.

### Calibrated Coverage (95% target)

**Scalar-calibrated (s=5.0):**

| Confidence | phi   | delta | phi_dot | delta_dot | All   |
|------------|-------|-------|---------|-----------|-------|
| 68%        | 0.534 | 0.763 | 0.972   | 0.990     | 0.485 |
| 90%        | 0.833 | 0.962 | 0.996   | 0.999     | 0.822 |
| 95%        | 0.888 | 0.985 | 0.998   | 1.000     | 0.880 |
| 99%        | 0.946 | 0.997 | 1.000   | 1.000     | 0.943 |

**Per-state-calibrated:**

| Confidence | phi   | delta | phi_dot | delta_dot | All   |
|------------|-------|-------|---------|-----------|-------|
| 68%        | 0.534 | 0.571 | 0.783   | 0.699     | 0.286 |
| 90%        | 0.833 | 0.896 | 0.923   | 0.915     | 0.688 |
| 95%        | 0.888 | 0.950 | 0.951   | 0.965     | 0.798 |
| 99%        | 0.946 | 0.986 | 0.982   | 0.984     | 0.902 |

Scalar calibration achieves 88% all-state coverage at the 95% level. Per-state calibration achieves 80% simultaneous coverage but with more balanced per-state coverage.

### ECE (Expected Calibration Error)

| Method    | Overall | phi   | delta | phi_dot | delta_dot |
|-----------|---------|-------|-------|---------|-----------|
| Raw       | 0.842   | 0.882 | 0.834 | 0.831   | 0.821     |
| Scalar    | 0.934   | 0.855 | 0.915 | 0.980   | 0.987     |
| Per-state | 0.885   | 0.855 | 0.870 | 0.915   | 0.901     |

The ECE values remain high after calibration. This indicates that while scalar/per-state scaling improves marginal coverage, the conditional calibration (reliability across different uncertainty levels) remains poor. The ensemble uncertainty does not cleanly separate "easy" from "hard" predictions.

## 3. OOD Detection Performance

| Metric | Value |
|--------|-------|
| AUROC | 0.7609 |
| AUPRC | 0.7903 |
| ID mean uncertainty | 0.0167 |
| OOD mean uncertainty | 0.0262 |
| OOD/ID ratio | 1.57x |

- AUROC of 0.76 indicates moderate discriminative ability. A random classifier achieves 0.50; a perfect classifier achieves 1.0.
- OOD samples have 57% higher mean uncertainty than ID samples.
- The separation is real but not dramatic -- there is significant overlap between ID and OOD uncertainty distributions.

## 4. Conclusions

### Is uncertainty reliable for decision-making?

**Partially, with significant caveats:**

1. **Raw uncertainty is unreliable.** The ensemble std underestimates true prediction error by approximately 5x. Using raw uncertainty directly would lead to severe overconfidence.

2. **Calibrated uncertainty is usable for marginal coverage.** After scaling (s ~ 5x), the model achieves approximately 88-95% coverage at the 95% level. This means calibrated uncertainty intervals contain the true state with reasonable frequency.

3. **Conditional calibration is poor.** The high ECE values indicate that the uncertainty does not reliably distinguish easy from hard predictions. A sample with 2x the uncertainty of another does not necessarily have 2x the error.

4. **OOD detection is moderate.** AUROC of 0.76 means uncertainty can partially flag out-of-distribution states, but is not reliable enough for safety-critical decisions. A threshold-based OOD detector would have non-trivial false positive and false negative rates.

5. **State-dependent performance.** phi (lean angle) is the hardest state to predict and calibrate. delta_dot is the easiest. Decision-making should account for this asymmetry.

### Recommended usage:
- Use calibrated uncertainty (scalar scale ~5x) for coverage guarantees
- Do not rely on raw ensemble std for uncertainty quantification
- Combine uncertainty with other OOD detection methods for safety-critical applications
- Consider phi uncertainty as the bottleneck for overall system reliability
