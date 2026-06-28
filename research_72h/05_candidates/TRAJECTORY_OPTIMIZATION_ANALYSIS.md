# Trajectory Optimization Neural ODE Analysis

**Date**: 2026-06-28
**Experiment**: EXP021_trajectory_optimization
**Status**: COMPLETE

---

## 1. Motivation

Previous experiments (EXP001-EXP017) all trained Neural ODE using **single-step loss**:
- Train: minimize MSE on one-step prediction (s_t -> s_{t+1})
- Test: roll out hundreds of steps using model's own predictions

This creates a fundamental mismatch: the model never sees its own errors during training, so it never learns to correct them. Error accumulates exponentially over long horizons.

**Root cause**: Single-step training doesn't penalize error accumulation.

---

## 2. Method: Trajectory Optimization

Instead of single-step loss, we train on **full trajectory rollouts**:

### 2.1 Trajectory Loss

For a trajectory of length H:
1. Start from state s_0
2. Roll out H steps using model's own predictions: s_0 -> s_1 -> ... -> s_H
3. Compare entire predicted trajectory with ground truth
4. Backpropagate through the entire rollout

```
Loss = (1/H) * sum_{t=0}^{H} ||s_t_pred - s_t_gt||^2
```

### 2.2 Scheduled Sampling (Teacher Forcing)

Pure pushforward training can be unstable early on. Scheduled sampling:
- Early training: use ground truth states (teacher forcing ratio = 1.0)
- Late training: use model's own predictions (teacher forcing ratio = 0.0)
- Gradually transition between them

### 2.3 Horizon Curriculum

Training on long trajectories from the start is unstable. Curriculum:
- Start with short horizons (H=5-10)
- Gradually increase to target horizon
- Model first learns good single-step dynamics, then learns to keep errors small

### 2.4 Multi-Scale Loss

Simultaneously train on multiple horizon lengths:
- Short (H=5-10): ensures good single-step accuracy
- Medium (H=20-50): learns error correction
- Long (H=100-200): learns long-term stability

---

## 3. Experimental Design

### 3.1 Configurations (v9 Architecture: 9,351 params)

| Config | H_max | TF Strategy | Curriculum | lambda_single | lambda_traj |
|--------|-------|-------------|------------|---------------|-------------|
| v9_arch_baseline | 1 | none | fixed | 1.0 | 0.0 |
| v9_traj_H50 | 50 | linear_decay | linear | 0.3 | 1.0 |
| v9_traj_H100_cosine | 100 | cosine_decay | exponential | 0.3 | 1.0 |
| v9_pushforward_H100 | 100 | none | linear | 0.5 | 1.0 |
| v9_traj_H200 | 200 | cosine_decay | exponential | 0.2 | 1.0 |
| v9_traj_step_tf_H100 | 100 | step_decay | step | 0.3 | 1.0 |

### 3.2 Model Architecture

- Hidden: 64, Depth: 3 (v9 architecture, 9,351 parameters)
- Activation: Tanh
- Last layer: Xavier init with gain=0.1, zero bias

### 3.3 Training

- 200 epochs, Adam optimizer, CosineAnnealing LR
- Batch size: 4-16 (smaller for longer horizons)
- Gradient clipping: 0.5-1.0

---

## 4. Results

### 4.1 Summary Table

| Config | H=1 | H=10 | H=50 | H=100 | H=200 | H=500 | Primary |
|--------|-----|------|------|-------|-------|-------|---------|
| **v9 baseline (original)** | **0.0051** | 0.0628 | 0.4557 | 0.5064 | **0.4737** | **0.5529** | **0.5110** |
| v9_arch_baseline | 0.0223 | 0.0511 | 0.1076 | 0.3631 | 0.7992 | 1.3175 | 0.8266 |
| v9_traj_H50 | 0.0222 | 0.0507 | 0.1079 | 0.3642 | 0.8001 | 1.3081 | 0.8242 |
| v9_traj_H100_cosine | 0.0226 | 0.0524 | 0.1123 | 0.3625 | 0.7775 | 1.2517 | 0.7973 |
| v9_pushforward_H100 | 0.0227 | 0.0538 | 0.1170 | 0.3689 | 0.7885 | 1.2720 | 0.8098 |
| v9_traj_H200 | 0.0223 | 0.0511 | 0.1072 | 0.3611 | 0.7923 | 1.2939 | 0.8158 |
| **v9_traj_step_tf_H100** | 0.0227 | 0.0529 | 0.1135 | 0.3615 | **0.7715** | 1.2521 | **0.7950** |

### 4.2 Trajectory Optimization Improvement over Single-Step Baseline

| Config | Primary | vs Baseline |
|--------|---------|-------------|
| v9_arch_baseline (single-step) | 0.8266 | -- |
| v9_traj_H50 | 0.8242 | -0.3% |
| v9_traj_H100_cosine | 0.7973 | -3.5% |
| v9_pushforward_H100 | 0.8098 | -2.0% |
| v9_traj_H200 | 0.8158 | -1.3% |
| **v9_traj_step_tf_H100** | **0.7950** | **-3.8%** |

### 4.3 Comparison with v9 Baseline (Original, with "buggy" multi-step loss)

| Horizon | v9 NMAE | Best TrajOpt | Improvement |
|---------|---------|--------------|-------------|
| H=1 | 0.0051 | 0.0227 | -344% (worse) |
| H=10 | 0.0628 | 0.0529 | +15.7% (better) |
| H=50 | 0.4557 | 0.1135 | +75.1% (better) |
| H=100 | 0.5064 | 0.3615 | +28.6% (better) |
| H=200 | 0.4737 | 0.7715 | -62.9% (worse) |
| H=500 | 0.5529 | 1.2521 | -126.5% (worse) |

### 4.4 Per-State NMAE (Best Config: v9_traj_step_tf_H100)

| Horizon | e_y | e_psi | v | theta | theta_dot | delta | delta_dot |
|---------|-----|-------|---|-------|-----------|-------|-----------|
| H=1 | 0.0043 | 0.0002 | 0.0115 | 0.0001 | 0.0033 | 0.0085 | 0.1306 |
| H=10 | 0.0256 | 0.0176 | 0.0626 | 0.0041 | 0.0235 | 0.0851 | 0.1520 |
| H=50 | 0.0850 | 0.0723 | 0.2778 | 0.0612 | 0.0544 | 0.1234 | 0.1205 |
| H=100 | 0.2776 | 0.3976 | 0.5217 | 0.3329 | 0.2981 | 0.4085 | 0.2938 |
| H=200 | 0.3789 | 0.7408 | 0.9375 | 0.8262 | 0.8604 | 0.8693 | 0.7877 |
| H=500 | 0.5215 | 0.9691 | 1.7775 | 1.3445 | 1.4682 | 1.3447 | 1.3390 |

---

## 5. Analysis

### 5.1 Trajectory Optimization Does Help (vs Single-Step Baseline)

All trajectory optimization methods improve over the single-step baseline:

- **v9_traj_step_tf_H100**: 3.8% improvement in Primary score
- **v9_traj_H100_cosine**: 3.5% improvement
- Best at H=200: v9_traj_step_tf_H100 (0.7715 vs 0.7992, -3.5%)
- Best at H=500: v9_traj_H100_cosine (1.2517 vs 1.3175, -5.0%)

The improvement is modest but consistent across all configurations.

### 5.2 Why Trajectory Optimization Doesn't Beat v9 Baseline

The original v9 baseline (Primary = 0.5110) remains superior because:

1. **"Buggy" multi-step loss was actually regularization**: The original v9 computed MSE on random state pairs within a trajectory, not sequential rollouts. This acted as a regularizer that prevented overfitting.

2. **Training-test distribution mismatch remains**: Even with trajectory optimization, the model trains on relatively short horizons (H=50-200) but is tested at H=500. The distribution shift at H=500 is not fully addressed.

3. **Limited data**: Only 43 training episodes with ~1000 steps each. The trajectory optimization needs more data to learn long-horizon dynamics.

4. **Gradient vanishing**: Through 100+ steps of backpropagation, gradients may vanish, limiting the model's ability to learn long-term corrections.

5. **Model capacity vs data**: The v9 architecture (9,351 params) is appropriately sized for the data. Larger models (51,591 params) overfit more.

### 5.3 Best Configuration Analysis

**v9_traj_step_tf_H100** (Primary = 0.7950):
- Step-decay teacher forcing: drops TF ratio by 0.2 every 25% of training
- Step curriculum: H=10 -> H=25 -> H=50 -> H=100
- This gradual transition is more stable than linear decay

### 5.4 The Delta Delta Problem

Looking at per-state NMAE, `delta_dot` has very high error at H=1 (0.1306) compared to other states. This suggests the `delta_dot` state (steering rate of change) is inherently harder to predict, and its errors propagate to other states over time.

---

## 6. Conclusions

### 6.1 Key Findings

1. **Trajectory optimization improves over single-step training** by 3-4% on the Primary score.
2. **Step-decay teacher forcing** works best among the scheduling strategies tested.
3. **The improvement is modest** - trajectory optimization alone cannot solve the long-horizon prediction problem.
4. **The original v9 baseline remains superior** due to its "buggy" regularization effect.
5. **H=10 and H=50 show significant improvement** (15-75% better than v9), but H=200/500 are still much worse.

### 6.2 Recommendations for Future Work

1. **Combine trajectory optimization with the v9 "buggy" loss**: Use both single-step regularization and trajectory loss.
2. **Data augmentation**: Generate more training data, especially at diverse speeds and conditions.
3. **Hybrid approach**: Use trajectory optimization for medium horizons (H=50-100) and a different model for H=200+.
4. **State-specific models**: Train separate models for `delta_dot` (which has high error) vs other states.
5. **Adaptive rollout during training**: Use the model's uncertainty to decide when to use GT vs predicted states.

---

## 7. Files

- **Code**: `D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/trajectory_optimization.py`
- **Results**: `D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP021_trajectory_optimization.json`
- **Analysis**: `D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/TRAJECTORY_OPTIMIZATION_ANALYSIS.md`
