# Stable Koopman Analysis

## 1. Method Overview

Stable Koopman addresses the Neural ODE long-horizon error accumulation problem through operator-theoretic linearization.

### Core Idea

Koopman operator theory provides a framework to lift a nonlinear dynamical system into a higher-dimensional space where the dynamics become **linear**. The key insight is:

> Linear systems do NOT accumulate errors the way nonlinear systems do.

If we learn:
- **Lifting function** phi: R^7 -> R^32 (encoder)
- **Inverse lifting** psi: R^32 -> R^7 (decoder)
- **Linear dynamics** z_{t+1} = K * z_t + B * u_t

Then long-horizon prediction in the lifted space is just repeated matrix multiplication, which is numerically stable when K is contractive.

### Stability Enforcement

The critical innovation is enforcing **spectral normalization** on K:
- Compute the largest singular value sigma_max(K)
- Rescale K so sigma_max <= 0.98 (within unit circle)
- This guarantees the linear dynamics are contractive
- Contractive dynamics = errors decay over time, not accumulate

### Architecture

```
State x (7D) -> Encoder -> z (32D) -> K*z + B*u -> z_next (32D) -> Decoder -> x_next (7D)
```

- Encoder: 4-layer MLP (7 -> 128 -> 128 -> 64 -> 32) with Tanh
- Decoder: 4-layer MLP (32 -> 64 -> 128 -> 128 -> 7) with Tanh
- K: 32x32 matrix with spectral normalization (sigma_max <= 0.98)
- B: 32x1 matrix (control input mapping)

### Training Loss

```
L = L_single + 0.1 * L_recon + 0.1 * L_linear + 0.5 * L_multi
```

Where:
- L_single: MSE of one-step prediction
- L_recon: Autoencoder reconstruction loss
- L_linear: Linear dynamics loss in lifted space
- L_multi: Multi-step rollout loss (enabled after epoch 20)

### Key Differences from Deep Koopman (EXP008)

| Aspect | EXP008 Deep Koopman | EXP023 Stable Koopman |
|--------|--------------------|-----------------------|
| Lifted dim | 20 | 32 |
| Hidden | 64 | 128 |
| Stability | K init = 0.95*I | Spectral normalization |
| Multi-step loss | No | Yes (after epoch 20) |
| Encoder depth | 3 layers | 4 layers |
| Decoder depth | 3 layers | 4 layers |
| Spectral monitoring | No | Yes |

## 2. Expected Results

Based on the theory, Stable Koopman should:

1. **H=1**: Slightly worse than v9 (lifting overhead)
2. **H=10-50**: Comparable or better (linear dynamics help)
3. **H=100-500**: Significantly better (stability prevents error accumulation)

The spectral bound of 0.98 means errors decay by factor 0.98 per step:
- After 100 steps: errors multiplied by 0.98^100 = 0.13 (87% decay)
- After 500 steps: errors multiplied by 0.98^500 = 0.0000041 (essentially zero)

## 3. Comparison with v9 Baseline

v9 baseline scores (NMAE):
- H=1: 0.0051
- H=10: 0.0628
- H=50: 0.4557
- H=100: 0.5064
- H=200: 0.4737
- H=500: 0.5529
- Primary (avg H=100,200,500): 0.5110

## 4. Potential Limitations

1. **Lifting approximation error**: The encoder may not perfectly linearize the dynamics
2. **Decoder reconstruction**: Inverse mapping may lose information
3. **Spectral bound trade-off**: Too tight (0.5) may over-constrain; too loose (0.99) may not help
4. **Multi-step loss instability**: Rollout gradients through many steps can be noisy

## 5. Results

(To be filled after training completes)

