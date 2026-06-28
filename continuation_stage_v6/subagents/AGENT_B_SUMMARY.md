# Agent B Summary: RDE-L / RDE-T / RDE-M Independence Verification

## Verdict: CONFIRMED

All three RDE variants (RDE-L, RDE-T, RDE-M) are **truly independent** implementations with different training labels and different formulas.

---

## What Each Model Does

| Model | Label Formula | State Source | DAgger Rounds |
|-------|--------------|-------------|---------------|
| **RDE-L** | `(F(s_i, u_i) - GP(s_i, u_i)) / delta_std` | Oracle (real training data states) | 0 |
| **RDE-T** | `(s_real[k+1] - s_real[k]) - (GP(s_model[k], u_k) - s_model[k]) / delta_std` | Real trajectory delta vs GP from drifted model state | 1 |
| **RDE-M** | `(dynamics.step(s_model, u) - s_model) - (GP(s_model, u) - s_model) / delta_std` | True dynamics FROM model state (local residual) | 2 |

### Key Distinctions

1. **RDE-L** uses oracle states from the training dataset. No model drift involved. Pure offline residual.

2. **RDE-T** compares the REAL trajectory delta (`s_real[k+1] - s_real[k]`) against GP prediction from the DRIFTED model state. This captures cumulative model error.

3. **RDE-M** runs the true dynamics simulator FROM the model state (`dynamics.step(s_model, u) - s_model`). This gives the local residual at the model state, not the trajectory comparison. This is the key difference from RDE-T.

---

## Label Correlation Evidence

All pairwise label correlations are near zero or negative, confirming independence:

| Pair | Correlation | Max Absolute Difference | Mean Absolute Difference |
|------|------------|------------------------|-------------------------|
| RDE-L vs RDE-T | 0.039 | 17.78 | 1.21 |
| RDE-L vs RDE-M | 0.003 | 32.91 | 0.85 |
| RDE-T vs RDE-M | -0.112 | 33.33 | 1.99 |

- RDE-L labels are essentially uncorrelated with both RDE-T and RDE-M (correlations ~0)
- RDE-T and RDE-M are negatively correlated (corr=-0.112), confirming their labels are structurally different
- Maximum differences are very large (17-33x the label std), far beyond numerical noise

---

## Prediction Comparison Evidence

All three models produce different predictions on the same test states:

| Pair | Correlation | Max Absolute Difference | Mean Absolute Difference |
|------|------------|------------------------|-------------------------|
| RDE-L vs RDE-T | 0.9994 | 0.126 | 0.022 |
| RDE-L vs RDE-M | 0.9997 | 0.111 | 0.017 |
| RDE-T vs RDE-M | 0.9999 | 0.052 | 0.006 |

- High correlation because they all share the same GP baseline
- But predictions are measurably different (max diff 0.05-0.13 on 4D state)
- RDE-T and RDE-M are most similar (corr=0.9999) because their labels share the same GP baseline computation, but differ in what "true delta" means

---

## Uncertainty Comparison Evidence

| Pair | Correlation | Max Absolute Difference |
|------|------------|------------------------|
| RDE-L vs RDE-T | 0.628 | 0.043 |
| RDE-L vs RDE-M | 0.664 | 0.040 |
| RDE-T vs RDE-M | 0.880 | 0.013 |

- Uncertainty estimates differ across all pairs
- RDE-L has the most distinct uncertainty (lowest correlation with others)
- RDE-T and RDE-M are most similar in uncertainty (corr=0.880) but still distinct

---

## Formula Verification Results

### RDE-L
- Uses oracle state: YES
- Label mean (normalized): ~0 across all 4 dimensions (expected for well-trained GP)
- Label std: 3.5e-5 to 4.2e-4 (very small residuals, GP fits well on training data)

### RDE-T
- Uses model state for GP: YES
- Uses real trajectory delta: YES
- Label mean: [1.87, -0.26, 0.65, 0.08] (large non-zero mean due to model drift)
- Label std: [5.56, 2.09, 2.13, 0.29] (large variance from trajectory divergence)
- Model state drift: 2.62 (significant drift over 200 steps)

### RDE-M
- Uses model state for GP: YES
- Uses true dynamics FROM model state: YES
- Label mean: [0.10, -1.70, -0.33, -0.21] (different from RDE-T)
- Label std: [0.42, 6.55, 2.78, 0.76] (different distribution from RDE-T)

---

## Conclusion

The three RDE variants are confirmed as **truly independent** implementations:

1. **RDE-L** is fundamentally different (oracle states vs model states)
2. **RDE-T** and **RDE-M** both use model states but compute different "true delta":
   - RDE-T: real trajectory delta (what actually happened)
   - RDE-M: true dynamics from model state (what would happen IF we started from model state)
3. All three produce measurably different labels (correlations ~0 or negative) and different predictions
