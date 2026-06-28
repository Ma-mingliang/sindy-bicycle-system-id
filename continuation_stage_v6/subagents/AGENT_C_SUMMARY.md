# Agent C: Horizon Crossover Analysis Summary

## Task
Determine the exact horizon where RDE models start outperforming pure GP in Mode B open-loop evaluation.

## Key Findings

### GP Divergence Horizon
**H = 200** (NMAE = 1.5748 > 1.0 threshold)

The GP model's NMAE grows exponentially after H=100:
- H=100: NMAE = 0.0169 (excellent)
- H=125: NMAE = 0.0480 (good)
- H=150: NMAE = 0.1467 (acceptable)
- H=200: NMAE = 1.5748 (diverged)

### Crossover Horizons (Where RDE Beats GP)

| Model | Crossover Horizon | GP NMAE at Crossover | Model NMAE at Crossover |
|-------|------------------|---------------------|------------------------|
| E1 (offline NN) | H = 500 | 27.71 | 26.46 |
| RDE-L (local) | H = 750 | 34.71 | 32.50 |
| RDE-T (trajectory) | Never | - | - |
| RDE-M (hybrid) | Never | - | - |

**Critical finding**: RDE-T and RDE-M NEVER outperform GP at any horizon. They diverge BEFORE GP does.

### RDE Divergence Horizons (Where RDE Diverges)

| Model | Divergence Horizon | GP Divergence |
|-------|-------------------|---------------|
| E1 | H = 200 | H = 200 |
| RDE-L | H = 150 | H = 200 |
| RDE-T | H = 100 | H = 200 |
| RDE-M | H = 50 | H = 200 |

**All RDE models diverge BEFORE or at the same horizon as GP.** The residual correction actually hurts long-horizon performance.

## NMAE vs Horizon Table

| Horizon | GP | E1 | RDE-L | RDE-T | RDE-M |
|---------|-----|-----|-------|-------|-------|
| 1 | 0.000056 | 0.000123 | 0.000130 | 0.007005 | 0.009872 |
| 2 | 0.000101 | 0.000290 | 0.001265 | 0.015344 | 0.020783 |
| 3 | 0.000156 | 0.000436 | 0.002378 | 0.024440 | 0.030738 |
| 5 | 0.000203 | 0.000551 | 0.003273 | 0.044250 | 0.042650 |
| 7 | 0.000221 | 0.000607 | 0.003331 | 0.054955 | 0.048165 |
| 10 | 0.000248 | 0.000683 | 0.003336 | 0.061083 | 0.061716 |
| 15 | 0.000309 | 0.000813 | 0.003692 | 0.073109 | 0.102115 |
| 20 | 0.000397 | 0.001000 | 0.004420 | 0.101167 | 0.150096 |
| 30 | 0.000778 | 0.001863 | 0.007590 | 0.187857 | 0.338532 |
| 40 | 0.001442 | 0.003746 | 0.013018 | 0.292508 | 0.720157 |
| 50 | 0.002334 | 0.006218 | 0.018846 | 0.407507 | 1.143512 |
| 75 | 0.005780 | 0.012754 | 0.044275 | 0.905826 | 4.184944 |
| 100 | 0.016931 | 0.042862 | 0.127358 | 1.954904 | 12.331835 |
| 125 | 0.047964 | 0.103343 | 0.426628 | 4.612580 | 23.083495 |
| 150 | 0.146662 | 0.377815 | 1.322337 | 8.064554 | 35.410923 |
| 200 | 1.574755 | 3.469658 | 5.095456 | 14.894667 | 73.449188 |
| 250 | 7.422086 | 11.100313 | 12.310929 | 20.251108 | 144.700791 |
| 300 | 13.402998 | 16.558142 | 18.092362 | 23.681529 | 290.849031 |
| 400 | 22.217653 | 22.691902 | 24.091117 | 30.334081 | 1146.038217 |
| 500 | 27.709682 | 26.463008 | 28.273628 | 36.128329 | 5040.006073 |
| 750 | 34.708376 | 31.231782 | 32.498271 | 40.261823 | 214121.380 |
| 1000 | 37.778217 | 36.386253 | 34.583904 | 41.041225 | 10626098.543 |

## Analysis

### Why RDE Models Underperform GP in Open-Loop

1. **State drift accumulation**: In Mode B, the model rolls out from its own predicted state. RDE residual corrections are trained on oracle (real) states, but during open-loop evaluation, the model state drifts away from the training distribution.

2. **Compounding residual error**: The NN residual correction adds noise that compounds exponentially over rollout steps. For RDE-M (2 DAgger rounds), this effect is most severe.

3. **RDE-T's paradox**: RDE-T is designed to handle state drift via trajectory-synced labels, but it performs worst among all models at short horizons (NMAE 100x worse than GP at H=1). This suggests the trajectory-sync training introduces significant variance.

4. **RDE-M's catastrophic divergence**: RDE-M (hybrid DAgger) diverges fastest (H=50) because its 2-round DAgger with local residual labels overfits to the training trajectory distribution.

### Why E1 and RDE-L Eventually Beat GP

At very long horizons (H=500+), all models have massive errors. GP's error plateaus while E1/RDE-L show slightly lower NMAE. This is likely because:
- GP's kernel structure limits its ability to capture the true dynamics at extreme state deviations
- E1/RDE-L's NN residual adds a small correction that happens to reduce error slightly in the "all models are wrong" regime

However, this crossover at H=500-750 is **not practically useful** because NMAE > 26 at these horizons for all models.

## Recommendation for MPC Planning Horizon

### Safe Operating Range: H <= 100

| Horizon Range | Recommendation | Rationale |
|---------------|----------------|-----------|
| H = 1-10 | GP is optimal | NMAE < 0.00025, all models work |
| H = 10-50 | GP preferred | NMAE < 0.0023, GP clearly best |
| H = 50-100 | GP acceptable | NMAE < 0.017, still accurate |
| H = 100-150 | Caution | NMAE growing rapidly (0.048-0.147) |
| H > 150 | **Do not use** | GP diverges (NMAE > 1.0) |

### Optimal MPC Horizon: H = 30-50

- **H=30**: NMAE = 0.000778 (excellent accuracy, good planning depth)
- **H=50**: NMAE = 0.002334 (still accurate, deeper planning)
- At H=50, GP is 3x better than E1, 8x better than RDE-L, 17x better than RDE-T

### Why Not RDE Models for MPC?

1. **Short horizons (H<100)**: GP is 2-100x better than any RDE model
2. **Long horizons (H>100)**: All models diverge, RDE models diverge first
3. **No crossover benefit**: The only crossover (E1 at H=500) is in the useless regime where NMAE > 26

### Practical MPC Configuration

```python
# Recommended MPC settings
mpc_horizon = 50          # Optimal balance of accuracy and planning depth
mpc_replan_freq = 10      # Replan every 10 steps
use_model = 'gp'          # GP is best for all practical horizons
```

## Technical Details

- **Training**: 5000 samples, GP max_samples=1000 (memory limit), 5-member NN ensembles, 50 epochs
- **Evaluation**: 3 Mode B segments, LQR controller with disturbances
- **Metric**: NMAE (Normalized Mean Absolute Error) across 4 state dimensions
- **Divergence threshold**: NMAE > 1.0
- **Runtime**: 357 seconds total (training ~300s, evaluation ~57s)

## Files Generated

- `continuation_stage_v6/subagents/AGENT_C_RESULTS.json` - Full numerical results
- `continuation_stage_v6/subagents/AGENT_C_SUMMARY.md` - This summary
- `continuation_stage_v6/subagents/v6_agent_c.py` - Analysis script
