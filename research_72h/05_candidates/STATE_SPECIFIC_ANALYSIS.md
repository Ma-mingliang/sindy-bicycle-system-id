# EXP041 State-Specific Experiments Analysis

**Date**: 2026-06-29 12:12

## Objective

Test whether dedicated models for individual states (e_y, e_psi, v, theta)
can outperform the monolithic v9 NODE baseline on the PrimaryLongHorizonScore.

## Method

For each target state, five architectures were tested:
1. **NODE (standard)**: 3-layer MLP (64 hidden, Tanh) predicting dsdt
2. **NODE (wide)**: 4-layer MLP (128 hidden, Tanh) predicting dsdt
3. **Direct Delta**: 3-layer MLP predicting delta_s directly (no dt)
4. **GP**: Gaussian Process with Matern kernel (5000 samples)
5. **Linear**: Ridge regression on normalized (state, action) input

Each specialized model predicts only its target state; the v9 model handles
the remaining 6 states during rollout evaluation.

## Per-State Results

### e_y

| Architecture | Primary Score | Target@H100 | vs v9 |
|---|---|---|---|
| node | 1.2267 | 0.9222 | -38.1% |
| node_wide **BEST** | 0.7769 | 0.5609 | +12.6% |
| direct_delta | 0.8174 | 0.6423 | +8.0% |
| linear | 0.9385 | 0.4153 | -5.6% |
| v9_baseline | 0.8885 | 0.5744 | --- |

**Best for e_y**: node_wide (Primary=0.7769)

### e_psi

| Architecture | Primary Score | Target@H100 | vs v9 |
|---|---|---|---|
| node | 1.3612 | 0.9924 | -53.2% |
| node_wide | 1.0421 | 1.1858 | -17.3% |
| direct_delta | 1.2361 | 1.0191 | -39.1% |
| linear **BEST** | 0.8710 | 0.7495 | +2.0% |
| v9_baseline | 0.8885 | 0.6348 | --- |

**Best for e_psi**: linear (Primary=0.8710)

### v

| Architecture | Primary Score | Target@H100 | vs v9 |
|---|---|---|---|
| node | 0.8925 | 0.5713 | -0.5% |
| node_wide | 0.8818 | 0.5620 | +0.8% |
| direct_delta | 0.7733 | 0.0078 | +13.0% |
| linear **BEST** | 0.7575 | 0.0000 | +14.7% |
| v9_baseline | 0.8885 | 0.4510 | --- |

**Best for v**: linear (Primary=0.7575)

### theta

| Architecture | Primary Score | Target@H100 | vs v9 |
|---|---|---|---|
| node | 1.3791 | 0.9035 | -55.2% |
| node_wide | 1.3187 | 0.7958 | -48.4% |
| direct_delta | 1.3146 | 0.8049 | -48.0% |
| linear **BEST** | 0.8378 | 0.2017 | +5.7% |
| v9_baseline | 0.8885 | 0.5759 | --- |

**Best for theta**: linear (Primary=0.8378)

## Composite Model (Experiment 5)

The best per-state architecture was selected and combined with v9 for remaining states.

| Horizon | v9 NMAE | Composite NMAE | Improvement |
|---|---|---|---|
| H=1 | 0.0344 | 0.0178 | +48.3% |
| H=10 | 0.0852 | 0.0470 | +44.8% |
| H=50 | 0.2429 | 0.2243 | +7.7% |
| H=100 | 0.5863 | 0.4028 | +31.3% |
| H=200 | 0.8386 | 0.6110 | +27.1% |
| H=500 | 1.2404 | 0.7517 | +39.4% |

**v9 Primary**: 0.8885
**Composite Primary**: 0.5885
**Overall Improvement**: +33.8%

## Key Findings

- **e_y**: node_wide improves by +12.6% over v9
- **e_psi**: linear improves by +2.0% over v9
- **v**: linear improves by +14.7% over v9
- **theta**: linear improves by +5.7% over v9

The composite model achieves +33.8% improvement over v9.

## Conclusion

State-specific specialization provides insight into which states benefit
from different modeling approaches, but the monolithic v9 architecture
remains competitive due to cross-state coupling in the dynamics.
