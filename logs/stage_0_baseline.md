# Stage 0: Baseline Verification

## Date: 2026-06-16 21:35

## Configuration
- Environment: PathTrackingEnv (LQR+Stanley, zero residual)
- Seeds: 0-49 (50 episodes)
- Max steps: 1500

## Results

| Metric | Value |
|--------|-------|
| return_mean | 31.61 |
| return_std | 40.48 |
| length_mean | 1018.3 |
| length_std | 566.8 |
| early_term_count | 21/50 (42%) |
| ey_rms_mean | 2.984 |
| epsi_rms_mean | 1.331 |
| theta_rms_mean | 1.454 |
| theta_max_max | 1.570 (pi/2 = termination) |
| control_energy_mean | 6654 |
| u_lqr_mean | 68.76 |
| u_total_mean | 68.76 |

## Key Findings

1. **Baseline ET = 42%**: Previously reported ET=5/10 was seed selection bias. Full 50-seed baseline shows 21 early terminations.

2. **theta_max hits termination threshold**: theta_max_max = 1.57 = pi/2. The baseline itself is near the stability boundary.

3. **High return variance**: return_std = 40.48 (128% of mean). Some seeds perform very well, others terminate early.

4. **LQR dominates**: u_lqr_mean = 68.76 vs u_stanley_mean = 0.48. Stanley contribution is <1% of total control.

## Risk Assessment

- ET=21/50 is high. Any MPPI method must NOT increase ET above this baseline.
- theta_rms=1.45 is close to pi/2=1.57. Small perturbations could push over the edge.
- The return improvement from +26 (10-seed) to +31.6 (50-seed) confirms seed variance.

## Implications for Reference-Level Residual

1. Reference-level residual (B/C) showed return=+42 with ET=3/10 in Phase 4 diagnostic.
2. But that was with world model trained on A data — not a fair comparison.
3. Must re-evaluate B/C with properly trained world model on stanley_ref data.
4. Success criteria: ET must not exceed 21/50, theta_rms must not exceed 1.454.

## Next Action

Stage 1: Confirm injection modes work correctly. Begin data collection for stanley_ref mode.
