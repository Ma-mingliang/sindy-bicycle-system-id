# State Space Augmentation Analysis

## Experiment Overview

**Date**: 2026-06-29
**Method**: State Space Augmentation Neural ODE
**Core Idea**: Expand 7D state space to 16D/32D with latent/hidden states to capture unobserved dynamics (tire forces, slip angles, lateral acceleration)

## Architecture Variants Tested

### Variant 1: Direct Augmentation (fast)
- **State**: augmented = [x_obs(7D), z_hidden(9D or 25D)]
- **ODE**: operates in augmented 16D or 32D space
- **Decoder**: extracts 7D observable from augmented state (residual: x_obs + correction)
- **Config**: hidden=128, depth=4, tanh, lr=1e-3

### Variant 2: v2 Extension
- **ODE**: f(x, z, a) takes augmented input [7+9+1] or [7+25+1] -> 7D dsdt
- **Hidden updater**: z_next = z + g(z, x, dsdt, a), small network zero-initialized
- **Config**: hidden=64, depth=3, tanh, lr=1e-3

### Variant 3: Residual Augmentation (attempted, not completed)
- **Base dynamics**: f_base(x, a) - v9-equivalent
- **Correction**: g_residual(x, z, a) - zero-initialized
- **Hidden dynamics**: h_hidden(x, z, a) - separate network

## Results Summary

| Config | H=1 | H=10 | H=50 | H=100 | H=200 | H=500 | Primary |
|--------|-----|------|------|-------|-------|-------|---------|
| v9 Baseline | **0.0255** | **0.0648** | **0.1854** | **0.4407** | **0.9967** | **1.3837** | **0.9404** |
| augmented_16d_direct | 1.0499 | 1.5311 | 1.6131 | 1.6131 | 1.6131 | 1.6131 | 1.6131 |
| augmented_9h_v2 | 16.3034 | 16.3034 | 16.3034 | 16.3034 | 16.3034 | 16.3034 | 16.3034 |

**All augmentation variants performed significantly worse than v9 baseline.**

## Why State Space Augmentation Failed

### 1. No Ground Truth for Hidden States

The fundamental problem: we have ground truth only for the 7D observable state (e_y, e_psi, v, theta, theta_dot, delta, delta_dot). The hidden states z have NO ground truth data. They can only be trained through backpropagation from the observable prediction loss.

This creates a severe credit assignment problem:
- The loss gradient must flow from the observable prediction error, through the decoder, through the ODE dynamics, to the hidden states
- This indirect gradient signal is weak and noisy
- The hidden dynamics can't learn meaningful representations

### 2. Increased Dynamics Complexity

The ODE dynamics function must now operate in a higher-dimensional space:
- v9: f: R^8 -> R^7 (8D input, 7D output)
- Augmented: f: R^17 -> R^16 or f: R^33 -> R^32

This makes the dynamics significantly harder to learn:
- More parameters to optimize
- Harder to find good gradient directions
- The loss landscape becomes more complex
- Training instability increases

### 3. Hidden State Divergence

Without ground truth constraints, the hidden dynamics can diverge:
- Hidden states evolve according to learned dynamics h(x, z, a)
- These dynamics are not constrained by any physical law
- Over long rollouts, hidden states can grow unboundedly
- This destabilizes the observable dynamics prediction

### 4. Training-Evaluation Mismatch

During training (single-step), hidden states are typically initialized to zero:
```
z = zeros(batch_size, n_hidden)
dsdt = model(x, z, a)
loss = MSE(dsdt, target)
```

But during evaluation (multi-step rollout), hidden states accumulate:
```
z = zeros(n_hidden)
for step in range(H):
    x, z = model(x, z, a)  # z evolves!
```

This creates a distribution mismatch: the model never sees non-zero z during single-step training, but must handle evolving z during evaluation.

### 5. Curriculum Destabilization

When the rollout curriculum activates (e.g., at epoch ~40 for rollout_steps=5), the loss increases dramatically:
- augmented_16d_direct: loss went from 0.27 (epoch 50) to 0.64 (epoch 100)
- augmented_9h_v2: loss went from 310 (epoch 50) to 735 (epoch 100)

This suggests the multi-step rollout loss destabilizes the hidden dynamics training.

## Comparison with v9 Baseline

| Metric | v9 Baseline | Best Augmented | Ratio |
|--------|-------------|----------------|-------|
| H=1 NMAE | 0.0255 | 1.0499 | 41x worse |
| H=10 NMAE | 0.0648 | 1.5311 | 24x worse |
| H=50 NMAE | 0.1854 | 1.6131 | 8.7x worse |
| H=100 NMAE | 0.4407 | 1.6131 | 3.7x worse |
| H=500 NMAE | 1.3837 | 1.6131 | 1.2x worse |
| H=1 Survival | 100% | 100% | same |
| H=50 Survival | 100% | 0% | much worse |

## Lessons Learned

### What Doesn't Work

1. **Naive state augmentation**: Simply expanding the state space with latent dimensions doesn't help. The extra degrees of freedom hurt more than they help.

2. **Unconstrained hidden dynamics**: Without ground truth or physical constraints, hidden dynamics diverge and destabilize predictions.

3. **Larger architecture**: Bigger networks (128 hidden, 4 layers) don't help when the fundamental approach is flawed. Overparameterization hurts.

### What Might Work Instead

1. **History Window Augmentation** (already tested as EXP005):
   - Instead of latent states, use observable history [x_t, x_{t-1}, ..., x_{t-k}]
   - No unobserved variables, all inputs have ground truth
   - Known to improve long-horizon prediction

2. **Physics-Informed Hidden States**:
   - Initialize hidden states to represent known physical quantities
   - Constrain hidden dynamics with physical equations
   - Example: z = [F_tire_lateral, F_tire_longitudinal, slip_angle_front, slip_angle_rear]

3. **Scheduled Sampling**:
   - During training, sometimes use model's own predictions as input
   - Reduces train-evaluation mismatch
   - Helps with error accumulation

4. **Ensemble Methods** (already tested as EXP011):
   - Train multiple v9 models with different seeds
   - Average predictions for robustness
   - Reduces variance without adding hidden states

5. **Error Correction Networks**:
   - Train a separate network to correct v9's predictions
   - Takes [x_pred, x_context] and outputs correction
   - Doesn't modify the base ODE dynamics

6. **Contractive Dynamics**:
   - Add Jacobian regularization to ensure stable dynamics
   - Prevents error amplification over long horizons
   - Already tested as EXP014

## Recommendations

1. **Do NOT use state space augmentation** for this problem. The fundamental issue (no ground truth for hidden states) makes it impractical.

2. **Focus on v9 improvements**: The v9 baseline is strong (NMAE 0.0255 at H=1). Improvements should come from:
   - Better training (scheduled sampling, curriculum)
   - Stability constraints (Jacobian regularization)
   - Ensemble methods

3. **If augmentation is needed**: Use physics-informed hidden states with proper constraints, not unconstrained latent dimensions.

4. **Best approach for long-horizon**: Combine v9 with error correction and scheduled sampling.

## Files

- **Code**: `D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/state_augmentation_v2.py`
- **Results**: `D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP022_state_augmentation.json`
- **Models**: `D:/系统辨识作业/sindy_bicycle/research_72h/07_models/state_aug_*.pt`
