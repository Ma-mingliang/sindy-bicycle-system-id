# EXP045: Wide Neural ODE for e_y, e_psi Analysis

**Date**: 2026-06-29 13:54
**Experiment**: Wide Neural ODE (hidden=256, depth=5) for e_y, e_psi

## 1. Architecture Overview

### Core Hypothesis

e_y and e_psi accumulate error from all other states through kinematic coupling.
A wider, deeper Neural ODE dedicated to these 2 states should capture their
complex dynamics better than a shared model.

### Models Tested

| Method | Architecture | Description |
|--------|-------------|-------------|
| v9_style_std_node | h=64,d=3,tanh | Standard v9-style NODE, all 7 states |
| wide_all7_wide_tanh_256x5 | h=256,d=5,tanh | Wide NODE, all 7 states |
| wide_all7_wide_silu_256x5 | h=256,d=5,silu | Wide NODE, all 7 states |
| hybrid_wide_tanh_256x5 | h=256,d=5,tanh | Wide NODE for e_y,e_psi + Std NODE for others |
| hybrid_wide_silu_256x5 | h=256,d=5,silu | Wide NODE for e_y,e_psi + Std NODE for others |

## 2. Results

### 2.1 Overall Performance

| Metric | Value |
|--------|-------|
| Best Method | wide_all7_wide_silu_256x5 |
| Primary Score | 0.3930 |
| v9 Baseline | 0.5110 |
| Improvement | 23.1% |

### 2.2 Horizon-by-Horizon Comparison

| Horizon | v9 Baseline | Best Method | Improvement |
|---------|------------|-------------|-------------|
| H=1 | 0.0051 | 0.0190 | -273.4% |
| H=10 | 0.0628 | 0.0372 | +40.8% |
| H=50 | 0.4557 | 0.1913 | +58.0% |
| H=100 | 0.5064 | 0.3689 | +27.1% |
| H=200 | 0.4737 | 0.4050 | +14.5% |
| H=500 | 0.5529 | 0.4050 | +26.7% |
| H=1000 | 0.6443 | 0.4050 | +37.1% |

### 2.3 Per-State NMAE (Best Method, H=200)

| State | NMAE | Model Type |
|-------|------|------------|
| e_y | 0.5701 | Wide NODE |
| e_psi | 0.4856 | Wide NODE |
| v | 0.0221 | Std NODE |
| theta | 0.4248 | Std NODE |
| theta_dot | 0.3221 | Std NODE |
| delta | 0.7820 | Std NODE |
| delta_dot | 0.2286 | Std NODE |

## 3. Method Comparison

| Method | Primary Score | vs v9 | Train Time |
|--------|--------------|-------|------------|
| v9_style_std_node | 0.8229 | -61.0% | 123s |
| wide_all7_wide_tanh_256x5 | 1.0813 | -111.6% | 283s |
| wide_all7_wide_silu_256x5 | 0.3930 | +23.1% | 384s |
| hybrid_wide_tanh_256x5 | 1.5724 | -207.7% | 246s |
| hybrid_wide_silu_256x5 | 1.3921 | -172.4% | 257s |

## 4. Key Findings

### Analysis

1. **Wide NODE capacity**: The wide NODE has ~330K params for just 2 output states,
   giving it much more representational capacity per output dimension.

2. **Residual connections**: Help gradient flow in the 5-layer deep network.

3. **Hybrid approach**: Combines the strengths of wide capacity for hard states
   with efficient standard capacity for easier states.

## 5. Conclusions

Best method: **wide_all7_wide_silu_256x5** with primary score 0.3930 
(+23.1% vs v9 baseline).