# Solution Candidates

## Tested Solutions

### S1: Sequential Batching (v9_fixed_seq)
- **Approach**: Fix shuffle=True bug, use contiguous trajectory segments
- **Result**: H=50 NMAE 0.385 (30% better), H=500 NMAE 0.476 (52% better)
- **Survival**: 0% at H=500 — trajectory explosion
- **Verdict**: Best NMAE but unacceptable survival

### S2: Sequential + Contractivity (v9_fixed_seq_contract)
- **Approach**: Add Jacobian eigenvalue penalty (lambda=0.05)
- **Result**: H=50 NMAE 0.647 (18% worse), H=500 NMAE 0.958 (4% better)
- **Survival**: 100% at all horizons
- **Verdict**: Survival achieved but NMAE regressed at short/mid horizons

### S3: Survival-Aware Loss (v9_tier1_survival)
- **Approach**: Penalize states exceeding physical limits during training rollout
- **Result**: Identical to S2 — survival loss was zero throughout training
- **Root cause**: Training max H=20, violations only occur at H=50+
- **Verdict**: NOT VIABLE with current curriculum

### S4: Soft Contractivity (soft_contract, lambda=0.01)
- **Approach**: 5x weaker contractivity penalty
- **Result**: H=50 NMAE 0.711 (30% worse), H=500 NMAE 0.987 (1% better)
- **Survival**: 100% at all horizons
- **Verdict**: Weaker contractivity doesn't help NMAE

### S5: Very Soft Contractivity (very_soft_contract, lambda=0.005)
- **Approach**: 10x weaker contractivity penalty
- **Result**: H=50 NMAE 0.727, H=500 NMAE 1.124 (worse than baseline!)
- **Survival**: 100%
- **Verdict**: Too weak to help, model still diverges at long horizons

### S6: Inference-Time Clipping (clip_seq)
- **Approach**: Clip predictions to 95% of physical limits after each step
- **Result**: H=50 NMAE 0.844 (119% worse!), H=500 NMAE 1.505 (216% worse!)
- **Survival**: 100%
- **Verdict**: Disastrous — clipping creates discontinuities causing error cascades

### S7: Safety-Blended Model (safety_blend_0.5) **BEST**
- **Approach**: Use sequential model normally, switch to contractive when near limits
- **Result**: H=50 NMAE 0.637 (16% worse), H=500 NMAE 0.946 (5.2% BETTER)
- **Survival**: 100%
- **Verdict**: Best compromise — beats baseline at H=500 with 100% survival

### S8: Fixed Blending (blend_0.3_seq)
- **Approach**: 30% sequential + 70% contractive predictions
- **Result**: H=50 NMAE 0.617, H=500 NMAE 0.859
- **Survival**: 40% at H=500
- **Verdict**: Good NMAE but insufficient survival

## Solution Ranking (configs with 100% survival)

| Rank | Config | Score | H=1 | H=50 | H=500 | Surv |
|------|--------|-------|-----|------|-------|------|
| 1 | v9_baseline_ref | 0.620 | 0.006 | 0.548 | 0.998 | 100% |
| 2 | safety_blend_0.5 | 0.640 | 0.035 | 0.637 | 0.946 | 100% |
| 3 | safety_blend_0.3 | 0.649 | 0.035 | 0.647 | 0.958 | 100% |
| 4 | v9_fixed_seq_contract | 0.649 | 0.035 | 0.647 | 0.958 | 100% |

Score = 0.2*H1 + 0.4*H50 + 0.4*H500 (lower is better)

## Key Insight

The accuracy-stability tradeoff is **partially breakable** via safety-blended inference:
- Use accurate model (V12 sequential) for normal operation
- Switch to safe model (V12 contractive) near physical limits
- Result: 5.2% better H=500 NMAE than baseline with 100% survival

However, H=50 NMAE (0.637) still exceeds baseline (0.548) by 16%.
This is because at H=50, some trajectories are already near limits, forcing the contractive model to activate.
