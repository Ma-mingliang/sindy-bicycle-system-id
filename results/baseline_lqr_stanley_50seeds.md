# Baseline: LQR+Stanley Zero-Residual (50 Seeds)

**Timestamp**: 20260616_213543

## Summary

| Metric | Value |
|--------|-------|
| return_mean | 31.6139 |
| return_std | 40.4795 |
| length_mean | 1018.3200 |
| length_std | 566.7857 |
| early_term_count | 21.0000 |
| early_term_rate | 0.4200 |
| ey_rms_mean | 2.9841 |
| ey_rms_std | 0.0806 |
| epsi_rms_mean | 1.3313 |
| epsi_rms_std | 0.1766 |
| theta_rms_mean | 1.4539 |
| theta_rms_std | 0.0934 |
| theta_max_mean | 1.5700 |
| theta_max_max | 1.5700 |
| control_energy_mean | 6654.0575 |
| u_lqr_mean | 68.7600 |
| u_lqr_max | 265.7059 |
| u_stanley_mean | 0.4820 |
| u_stanley_max | 2.0440 |
| u_total_mean | 68.7600 |
| u_total_max | 265.7059 |

## Risk Assessment

- **WARNING**: ET=21/50 — baseline itself is unstable
- **WARNING**: theta_max_max=1.570 — close to termination (pi/2=1.57)
- **WARNING**: theta_rms_mean=1.454 — high roll angle

## Conclusion

Baseline return: 31.61 +/- 40.48
Baseline ET: 21/50
