# Multiple Shooting Neural ODE - Analysis

## Problem Statement

Neural ODE models trained with single-step loss suffer from error accumulation over long horizons. A model that is accurate at H=1 can produce wildly wrong predictions at H=100+ because:

1. **Single-step loss is myopic**: The model only learns one-step accuracy, not multi-step composition.
2. **Error compounds multiplicatively**: Small per-step errors compound through ODE integration.
3. **No long-horizon feedback during training**: The model never "sees" its own mistakes.

## Multiple Shooting Approach

### Core Idea

Instead of single-step loss on shuffled (s, a, delta) tuples, use **segment-level loss** on short multi-step rollouts:

```
Standard Training:           Multiple Shooting:
  s_t -> predict -> s_{t+1}    s_0 -> rollout 15 steps -> s_15
  Loss = MSE(pred, target)     Loss = sum of MSE over all 15 steps
  (each step independent)      (errors compound within segment)
```

### Key Design Decisions

1. **Shooting Interval (segment_len)**: 15 steps (0.5 seconds at 30Hz)
2. **Segment-Level Loss**: Average MSE over all steps in the segment
3. **Final-State Penalty**: Extra weight on the last predicted state
4. **Shooting Nodes**: Each segment starts from a REAL observed state

### Mathematical Formulation

Given trajectory {s_0, ..., s_T} with actions {a_0, ..., a_{T-1}}:

**Single-step loss:**
```
L_single = (1/T) * sum_t ||f_theta(s_t, a_t) - (s_{t+1} - s_t)||^2
```

**Segment loss (for segment of length L starting at s_k):**
```
L_segment = (1/L) * sum_{j=0}^{L-1} ||s_hat_{k+j+1} - s_{k+j+1}||^2
where s_hat_{k+i+1} = s_hat_{k+i} + f_theta(s_hat_{k+i}, a_{k+i}) * dt
```

**Total loss:**
```
L = 0.2 * L_single + 1.0 * L_segment + 0.5 * L_final
```

## Results

### Multiple Shooting vs Baselines

| Horizon | v9 Baseline | Single-Step | Shooting   | Anchored   |
|---------|-------------|-------------|------------|------------|
| H=1     | 0.0051      | 0.0105      | 0.0290     | 0.0290     |
| H=10    | 0.0628      | 0.0999      | 0.0782     | 0.0782     |
| H=50    | 0.4557      | 0.2343      | 0.2615     | 0.1039     |
| H=100   | 0.5064      | 0.3907      | 0.5106     | 0.1239     |
| H=200   | 0.4737      | 0.5677      | 0.6928     | 0.1614     |
| H=500   | 0.5529      | 0.8012      | 0.8385     | 0.1916     |

### Primary Long-Horizon Score (avg H=100,200,500)

| Model              | Score   | vs v9     |
|--------------------|---------|-----------|
| v9 Baseline        | 0.5110  | baseline  |
| Single-Step        | 0.5865  | -14.8%    |
| Multiple Shooting  | 0.6806  | -33.2%    |
| Anchored Shooting  | 0.1589  | +68.9%    |

### Per-State NMAE at H=500

| State      | Shooting | Single-Step |
|------------|----------|-------------|
| e_y        | 0.8427   | 0.7462      |
| e_psi      | 1.0025   | 1.2920      |
| v          | 1.8306   | 1.4358      |
| theta      | 0.6965   | 0.5513      |
| theta_dot  | 0.5095   | 0.5585      |
| delta      | 0.4909   | 0.3168      |
| delta_dot  | 0.4969   | 0.7078      |

## Critical Finding: Train-Test Domain Mismatch

**The basic Multiple Shooting approach FAILS to improve long-horizon prediction.**

The key evidence is the dramatic gap between:
- **Anchored shooting** (H=500): NMAE = 0.1916 (resets to real state every 15 steps)
- **Free rollout** (H=500): NMAE = 0.8385 (uses own predictions)

This 4.4x gap reveals the fundamental problem:

### Why It Fails

1. **Training domain**: Segments always start from REAL observed states
2. **Test domain**: The model must use its OWN predicted states as initial conditions
3. **Domain gap**: The model never learns to recover from its own prediction errors

During training with anchored segments:
- Step 1: Predict from real state -> small error
- Step 2: Predict from slightly wrong state -> slightly larger error
- ... 
- Step 15: Error has accumulated but model is still "close" to trajectory

During free rollout:
- The model compounds its own errors without correction
- It was never trained on states that deviate significantly from ground truth
- Result: error grows faster than single-step baseline

### Why Single-Step Baseline Is Better

The single-step baseline (NMAE=0.8012 at H=500) outperforms the shooting model (NMAE=0.8385) because:
- It sees diverse (s, a, delta) pairs from the full dataset
- It doesn't have the train-test mismatch
- Each training sample is independent, so the model learns generalizable dynamics

## What Would Actually Work

The anchored results (NMAE=0.1916 at H=500) show that the model IS capable of accurate short-horizon prediction. The problem is purely in error accumulation. Approaches that would help:

### 1. Hybrid Training (Best Theoretical Approach)
Mix training modes:
- 50% single-step loss (general dynamics learning)
- 30% free-rollout loss (3-10 steps, no anchoring)  
- 20% anchored segment loss (short-horizon accuracy)

This gives the model exposure to its own errors during training.

### 2. Noise-Augmented Training
Add Gaussian noise to initial states during segment training:
```python
s0_noisy = s0_real + noise * sigma
```
This simulates prediction error and teaches the model to handle perturbations.

### 3. Corrector Network
Learn a small correction network that adjusts predictions based on accumulated state:
```python
s_next = f_dynamics(s, a) + g_corrector(s, context)
```

### 4. Longer Curriculum
Train with gradually increasing rollout length:
- Epochs 1-50: 1-step
- Epochs 50-100: 3-step free rollout
- Epochs 100-150: 10-step free rollout
- Epochs 150-200: 20-step free rollout

## Conclusion

**Multiple Shooting with anchored segments is NOT effective for Neural ODE training** in the standard form tested here. The train-test domain mismatch (real states during training vs predicted states during testing) causes worse long-horizon performance than simple single-step training.

The correct approach for reducing long-horizon error is to **train with free rollouts** (no anchoring) so the model learns to handle its own prediction errors. The shooting concept can inform the architecture (e.g., using correction networks at segment boundaries) but should not be used as a direct training strategy.

### Key Takeaway

> **Training with perfect initial states does not prepare a model for deployment with imperfect predictions.** Any multi-step training must expose the model to its own errors, not just ground-truth states.
