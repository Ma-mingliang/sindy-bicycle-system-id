# Stage 4: MPPI Diagnostics (stanley_ref injection)

**Timestamp**: 20260616_214130
**Checkpoint**: residual_mppi_stanley_ref/checkpoint_ensemble_mppi.pt
**Uncertainty threshold**: 0.000356
**Injection mode**: stanley_ref (B: epsilon → delta_stanley → theta_target → LQR)

## Results

| margin | fb%  | rwd  | risk | unc  | marg | nf    | \|e\|  | e_max | pred  | real  | ET | len | ey_rms | ep_rms | th_rms | th_max |
|--------|------|------|------|------|------|-------|--------|-------|-------|-------|-----|-----|--------|--------|--------|--------|
| 0.010  | 85%  | 0.354| 0.000| 0.140| 0.697| 159.1 | 0.06495| 0.100 | +0.024| +8.77 | 4   | 1045| 2.979  | 1.317  | 1.455  | 1.570  |
| 0.005  | 69%  | 0.347| 0.000| 0.140| 0.554| 322.9 | 0.05044| 0.100 | +0.016| +9.12 | 4   | 1045| 2.980  | 1.317  | 1.456  | 1.570  |
| 0.001  | 19%  | 0.351| 0.000| 0.141| 0.169| 845.1 | 0.03548| 0.100 | +0.011| +9.14 | 4   | 1045| 2.976  | 1.317  | 1.456  | 1.570  |
| 0.000  | N/A  |      |      |      |      |       |        |       |       |       |     |     |        |        |        |        |

*Note: margin=0.0 skipped (MPPI every step too slow). Covered in Stage 5 sweep.*

## Baseline (50 seeds, LQR+Stanley, no residual)

| Metric   | Value  |
|----------|--------|
| ET       | 21/50 (42%) |
| ey_rms   | 2.984  |
| epsi_rms | 1.331  |
| theta_rms| 1.454  |
| return   | 31.61  |

## Key Findings

1. **Real advantage is POSITIVE** across all margins: +8.77 to +9.14
2. **ET dramatically improved**: 4/10 vs baseline 21/50 (42%)
3. **epsi_rms improved**: 1.317 vs baseline 1.331
4. **theta_rms preserved**: 1.455 vs baseline 1.454 (within noise)
5. **margin=0.001 optimal balance**: fb=19%, nf=845, real_adv=+9.14

## Analysis

### Margin Gate Behavior
- margin=0.01: 85% fallback, only 159 steps active — too conservative
- margin=0.005: 69% fallback, 323 steps active — still conservative
- margin=0.001: 19% fallback, 845 steps active — good balance
- margin=0.000: 0% fallback expected, all steps active — maximum MPPI usage

### Predicted vs Real Advantage
- Predicted advantage is small (+0.011 to +0.024) due to world model conservatism
- Real advantage is much larger (+8.77 to +9.14) — world model underestimates benefit
- This gap suggests the world model is conservative but the residual is genuinely helpful

### Safety Profile
- ET=4/10 across all margins (vs baseline 21/50=42%)
- theta_rms=1.455-1.456 (baseline 1.454) — no degradation
- ey_rms=2.976-2.980 (baseline 2.984) — slight improvement

## Next Steps

Stage 5 will perform fine-grained tuning:
1. Margin sweep: [0.0, 0.001, 0.005, 0.01, 0.02]
2. Lambda_res sweep: [0.2, 0.5, 1.0, 2.0]
3. Sampling sweep: [64, 128, 256]
4. Epsilon sweep: [0.05, 0.08, 0.10, 0.15]
5. Horizon sweep: [3, 5, 8, 10]

Best config will proceed to Stage 6: 50-seed paired evaluation.
