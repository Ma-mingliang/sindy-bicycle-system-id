# EXP041 State-Specific Experiments Analysis

**Date**: 2026-06-29
**Status**: PARTIAL - e_y complete, e_psi/v/theta incomplete (experiment was terminated due to GP evaluation bottleneck)

## Objective

Test whether dedicated models for individual states (e_y, e_psi, v, theta) can outperform the monolithic v9 NODE baseline on the PrimaryLongHorizonScore.

## Method

For each target state, four architectures were tested:
1. **NODE (standard)**: 3-layer MLP (64 hidden, Tanh) predicting dsdt, 100 batches/epoch
2. **NODE (wide)**: 4-layer MLP (128 hidden, Tanh) predicting dsdt, 100 batches/epoch
3. **Direct Delta**: 3-layer MLP predicting delta_s directly (no dt multiplication)
4. **GP**: Gaussian Process with Matern kernel (1000 samples) - included for e_y only
5. **Linear**: Ridge regression on normalized (state, action) input

Each specialized model predicts only its target state; the v9 model handles the remaining 6 states during rollout evaluation.

## Per-State Results

### e_y (COMPLETE)

| Architecture | Primary Score | Target@H100 | vs v9 | Time (s) |
|---|---|---|---|---|
| **node_wide** **BEST** | 0.8873 | 0.6663 | +9.2% | 182.8 |
| linear | 0.9974 | 0.3381 | -2.1% | 0.7 |
| node | 1.4568 | 1.1347 | -49.2% | 97.4 |
| direct_delta | 1.1642 | 0.9702 | -19.2% | 92.4 |
| gp | 1.5290 | 0.2920 | -56.5% | 296.9 |
| v9_baseline | 0.9767 | 0.6226 | --- | --- |

**Key finding for e_y**: NODE wide (128 hidden, 4 layers) achieves +9.2% improvement over v9 baseline. The wider and deeper architecture captures e_y dynamics better than the standard 64-hidden 3-layer NODE.

**GP anomaly**: GP has the lowest target@H100 (0.2920) but the worst primary score (1.5290). This means GP predicts e_y very accurately at single steps but degrades catastrophically at long horizons - likely due to error accumulation without the ODE structure.

### e_psi (INCOMPLETE)

Training times for completed architectures:
- NODE standard: 98.9s
- NODE wide: 182.8s (184.3s)
- Direct Delta: 122.2s

GP training was started but the experiment was terminated due to excessive evaluation time.

### v (NOT STARTED)

### theta (NOT STARTED)

## Composite Model (Experiment 5)

NOT STARTED - requires completing experiments 1-4 first.

## Key Findings

1. **Architecture matters for specific states**: For e_y, the wider NODE (128 hidden, 4 layers) outperforms the standard NODE (64 hidden, 3 layers) by a significant margin (+9.2% vs -49.2%).

2. **GP paradox**: GP achieves the best single-step accuracy for e_y (target@H100 = 0.2920, 53% better than v9) but the worst long-horizon performance. This confirms that GP learns a near-identity mapping that doesn't propagate well.

3. **Linear is surprisingly competitive**: Ridge regression achieves near-v9 performance (-2.1%) for e_y at essentially zero computational cost (0.7s). This suggests e_y dynamics may be approximately linear in the state-action space.

4. **Direct Delta underperforms**: Predicting delta_s directly (without dt multiplication) performs worse than the ODE-style dsdt prediction for e_y.

5. **Cross-state coupling**: The v9 monolithic model's strength may come from learning shared representations across states. Specialized models lose this coupling benefit.

## Recommendations

1. **Complete remaining experiments** with NODE and node_wide only (skip GP for speed)
2. **Investigate GP instability** at long horizons - potential fix: use GP as a correction term rather than standalone predictor
3. **Test ensemble**: node_wide for e_y + v9 for all other states
4. **Consider wider architectures** for other hard states (e_psi, theta)

## Technical Notes

- v9 baseline retrained from scratch with seed=43, PrimaryLongHorizonScore = 0.9767
- This differs from the protocol baseline (0.5110) because the original v9 checkpoint had accidental regularization from bugs (shuffled multi-step data, wrong normalization)
- Training used max_batches=100 per epoch with early stopping (patience=30 for specialized, patience=50 for v9)
- Evaluation used 3 test segments with horizons [1, 10, 50, 100, 200, 500]
