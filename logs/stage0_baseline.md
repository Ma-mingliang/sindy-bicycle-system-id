# Stage 0: Baseline Evaluation

**Timestamp**: 20260617_105429
**Config**: 50 seeds, zero-residual, stanley_ref injection

## Results

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Return | 31.61 | 40.48 | -19.87 | 79.42 |
| Episode length | 1018.3 | 566.8 | 273 | 1500 |
| ey_rms | 2.9841 | 0.0806 | 2.6912 | 3.1165 |
| epsi_rms | 1.3313 | 0.1766 | 1.1266 | 1.6241 |
| theta_rms | 1.4539 | 0.0934 | 1.1953 | 1.5384 |
| theta_max | 1.5700 | — | — | — |
| Control energy | 6654.0575 | 582.5354 | — | — |
| u_lqr_mean | 0.0000 | — | — | — |
| u_stanley_mean | 0.4820 | — | — | — |
| u_total_mean | 68.7600 | — | — | — |

**Early terminations**: 21/50 (42.0%)

## Next Steps

Stage 1: LQR Q/R re-tuning sweep to prioritize safety.
