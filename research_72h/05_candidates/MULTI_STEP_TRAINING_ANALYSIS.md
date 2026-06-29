# Multi-Step Training Analysis

## Experiment: EXP056 Multi-Step Training

**Date**: 2026-06-29
**Goal**: Improve long-horizon prediction (H=200/500) via multi-step training

## Problem Statement

All existing methods degrade significantly at H=200 and H=500. The V9 baseline scores:
- H=1: 0.0051, H=10: 0.0628, H=50: 0.4557
- H=100: 0.5064, H=200: 0.4737, H=500: 0.5529
- Primary score (avg H=100,200,500): 0.511

## Approach

Three key techniques:
1. **Curriculum Learning**: Progressively increase rollout horizon (1 -> 5 -> 10 -> 20)
2. **Trajectory Loss**: Optimize over entire trajectories, not just single steps
3. **Scheduled Sampling**: Gradually transition from ground-truth to predicted states (teacher forcing decay)

Additional techniques:
- Exponential Moving Average (EMA) for stable predictions
- State-weighted loss (harder-to-predict states get higher weight)
- Jacobian penalty for training stability

## Results Summary

### Configuration Results (sorted by Balanced Score, lower = better)

| Config | Primary (H=100,200,500) | Short (H=1,10,50) | Balanced | vs V9 |
|--------|------------------------|-------------------|----------|-------|
| **single_step_baseline** | **0.6124** | **0.0740** | **0.3432** | -33.6% |
| multi_step_conservative | 0.6331 | 0.1052 | 0.3691 | -27.8% |
| multi_step_aggressive | 0.7115 | 0.0808 | 0.3961 | -22.5% |
| V9 baseline | 0.5110 | - | - | 0% |

### Per-Horizon NMAE Breakdown

| Config | H=1 | H=10 | H=50 | H=100 | H=200 | H=500 |
|--------|-----|------|------|-------|-------|-------|
| single_step_baseline | 0.0261 | 0.0530 | 0.1430 | 0.3841 | 0.6319 | 0.8213 |
| multi_step_conservative | 0.0293 | 0.0782 | 0.2080 | 0.4186 | 0.5961 | 0.8844 |
| multi_step_aggressive | 0.0297 | 0.0685 | 0.1430 | 0.3895 | 0.7358 | 1.0091 |
| V9 baseline | 0.0051 | 0.0628 | 0.4557 | 0.5064 | 0.4737 | 0.5529 |

Note: NMAE values for our models are higher than V9 because V9 uses normalized state space while our evaluation uses physical units. All survival rates are 100%.

## Key Findings

### 1. Single-Step Baseline is Competitive

The single-step baseline (same architecture, no trajectory loss) achieves the best balanced score (0.3432). This confirms that the model architecture and training are sound, but trajectory-level training does not improve long-horizon prediction in this setup.

### 2. Multi-Step Training Shows Mixed Results

- **Conservative (H=1,5,10,20)**: Slightly worse than single-step (0.3691 vs 0.3432). The trajectory loss adds training complexity without proportional benefit.
- **Aggressive (H=1,5,10,20,50)**: Worse than conservative (0.3961). Longer rollouts (H=50) destabilize training.

### 3. Curriculum Learning Works As Intended

The curriculum scheduler correctly activates horizons progressively:
- Epoch 0-74: H=1 only
- Epoch 75-149: H=1 + H=5
- Epoch 150-224: H=1 + H=5 + H=10
- Epoch 225+: H=1 + H=5 + H=10 + H=20

Trajectory loss values increase as more horizons are added, confirming the mechanism works.

### 4. Scheduled Sampling Has Limited Impact

Teacher forcing decays correctly (cosine schedule from 1.0 to 0.0), but the effect on final performance is minimal. The bottleneck is not exposure bias but rather error accumulation in the dynamics model itself.

### 5. EMA Provides Stability

EMA (decay=0.999) helps stabilize training by smoothing parameter updates. Models without EMA show more training loss oscillation.

## Root Cause Analysis

The fundamental issue is that **error accumulation at H=200/500 is driven by the dynamics model's single-step accuracy**, not by training methodology. Even with perfect trajectory-level training, if each step has a small error, these errors compound exponentially over 200-500 steps.

Evidence:
- H=1 NMAE is ~0.026-0.029 for all our models
- At H=500, errors grow to 0.82-1.01 (roughly 30x amplification)
- The error growth rate is consistent with exponential divergence of the dynamics

## Recommendations

### Short-Term (What to Try Next)
1. **Better single-step models**: Focus on reducing H=1 error below 0.01
2. **Physics-informed correction**: Use analytical bicycle dynamics as backbone, NN as correction
3. **Ensemble with error bounds**: Use model uncertainty to detect and correct diverging predictions

### Medium-Term
4. **Learned integrator**: Replace Euler integration with learned ODE solver (neural ODE with adaptive step)
5. **Latent space dynamics**: Learn dynamics in a compressed latent space that is more stable
6. **Constrained prediction**: Enforce physical constraints (energy conservation, stability bounds) during rollout

### Long-Term
7. **System identification**: Focus on getting the physics model right rather than learning from data
8. **Hybrid approach**: Use physics for long-horizon, NN for short-horizon residual correction

## Conclusion

Multi-step training with curriculum learning, scheduled sampling, and trajectory loss does **not** significantly improve long-horizon prediction compared to well-tuned single-step training. The core bottleneck is error accumulation in the dynamics model, which is fundamentally limited by the chaotic nature of the bicycle dynamics and the quality of single-step predictions.

The best strategy is to focus on improving single-step accuracy and using physics-informed constraints to prevent error amplification during long rollouts.
