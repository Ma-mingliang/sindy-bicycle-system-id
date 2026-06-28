# Agent F: 7D Route Baseline Summary

## Task
Establish a minimal 7D route baseline using real data.

## Data Availability

**Real data used**: `data/stage2_dataset_150k.npz` (150,000 samples, 58 episodes)

The original 8D state vector `[e_y, e_psi, v, theta, theta_dot, k, delta, delta_dot]` was reduced to 7D by dropping kappa (index 5), which is constant at zero across all episodes. This is consistent with the reclassification in `ROUTE8D_STATE_RECLASSIFICATION.md`.

### 7D State Vector
| State | Range | Std | Mean | Notes |
|-------|-------|-----|------|-------|
| e_y | [-0.50, 0.50] | 0.238 | +0.180 | Path lateral error |
| e_psi | [-1.00, 1.00] | 0.536 | -0.171 | Path heading error |
| v | [0.54, 0.66] | 0.013 | +0.600 | Speed (narrow range, nearly constant) |
| theta | [-1.00, 1.00] | 0.549 | -0.783 | Roll angle |
| theta_dot | [-0.47, 0.47] | 0.185 | -0.265 | Roll rate |
| delta | [-1.00, 1.00] | 0.551 | -0.784 | Steering angle |
| delta_dot | [-1.00, 1.00] | 0.546 | -0.781 | Steering rate |

- Action (tau): [-0.10, 0.10], std=0.058
- Episode lengths: 184 to 18,260 steps (mean: 2,536)

## GP Training

**Feasibility**: Yes, GP training succeeded.

- **Config**: 1,000 training samples, 1 restart, RBF kernel
- **Training time**: 47.1s (7 GPs, one per state dimension)
- **Train/test split**: 80/20 by episode (112,590 train / 34,498 test)

## Performance Metrics

### Single-Step Prediction (next-state from current state + action)

| State | MAE | RMSE |
|-------|-----|------|
| **Overall** | **0.01160** | -- |
| e_y | 0.02033 | 0.08396 |
| e_psi | 0.05747 | 0.19204 |
| v | 2.0e-08 | 2.5e-08 |
| theta | 6.4e-05 | 7.8e-04 |
| theta_dot | 0.00238 | 0.00335 |
| delta | 3.3e-04 | 0.00257 |
| delta_dot | 6.5e-04 | 0.00329 |

Key observations:
- **v** has near-zero error because it is nearly constant (std=0.013) -- GP learns a trivial constant predictor
- **theta, delta, delta_dot** have very small single-step errors (order 1e-4 to 1e-3)
- **e_psi** has the largest single-step error (0.057) -- most variable kinematic state

### Multi-Step Rollout (endpoint MAE at horizon h)

| Horizon | Endpoint MAE | Valid Episodes |
|---------|-------------|----------------|
| 1 | 0.00124 | 12 |
| 5 | 0.00597 | 12 |
| 10 | 0.02906 | 12 |
| 20 | 0.04021 | 12 |
| 50 | 0.08433 | 12 |
| 100 | 0.61007 | 12 |

Key observations:
- **Horizon 1-10**: GP performs well, endpoint MAE < 0.03
- **Horizon 20-50**: Moderate degradation, endpoint MAE 0.04-0.08
- **Horizon 100**: Significant divergence (MAE = 0.61), GP accumulates errors over long rollouts
- **delta_dot** is the primary source of error accumulation at long horizons

## Conclusion

The GP baseline establishes a strong single-step benchmark on real 7D data. Multi-step performance degrades gracefully up to horizon 20-50 but diverges at horizon 100. This confirms that:

1. **Real 7D data is available and usable** -- no synthetic data needed
2. **GP is feasible** for the 7D route problem (1,000 samples, 47s training)
3. **Single-step prediction is accurate** (overall MAE = 0.012)
4. **Multi-step rollouts degrade beyond horizon 50** -- more advanced models (NN, hybrid) needed for long-horizon prediction
5. **kappa removal is trivial** -- it was always zero in the dataset

## Files
- Script: `continuation_stage_v6/subagents/v6_agent_f.py`
- Results: `continuation_stage_v6/subagents/AGENT_F_RESULTS.json`
