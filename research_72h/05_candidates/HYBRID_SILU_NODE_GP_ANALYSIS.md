# Hybrid SiLU NODE + GP Experiment Analysis

## Experiment: EXP048_hybrid_silu_node_gp

**Date**: 2026-06-29
**Architecture**: Wide SiLU NODE (256x5, SiLU) for complex states + GP (Matern 2.5) for smooth states
**Training data**: 30,000 subsampled from 150k dataset

### State Grouping
- **SiLU NODE** (complex): e_y, e_psi, theta_dot, delta_dot
- **GP** (smooth): v, theta, delta

---

## 1. Raw Results

| Horizon | NMAE    | Survival |
|---------|---------|----------|
| 1       | 0.0246  | 100%     |
| 10      | 0.0525  | 100%     |
| 50      | 0.2082  | 100%     |
| 100     | 0.4861  | 100%     |
| 200     | 0.7703  | 100%     |
| 500     | 0.9969  | 80%      |
| 1000    | 1.1538  | 80%      |

**PrimaryLongHorizonScore** (avg H=100,200,500): **0.7511**

---

## 2. Baseline Comparison

### Primary Score Comparison

| Model                     | Primary Score | vs Hybrid  |
|---------------------------|---------------|------------|
| v9 baseline               | 0.5110        | -47.0%     |
| Wide SiLU NODE (256x5)    | 0.3930        | -91.1%     |
| Per-State Residual        | 0.3986        | -88.4%     |
| **Hybrid SiLU NODE + GP** | **0.7511**    | baseline   |

**Verdict: The hybrid model performs WORSE than all baselines.**

### Per-Horizon Comparison vs v9 Baseline

| Horizon | v9 NMAE | Hybrid NMAE | Improvement |
|---------|---------|-------------|-------------|
| 1       | 0.0051  | 0.0246      | **-381.8%** (much worse) |
| 10      | 0.0628  | 0.0525      | **+16.4%** (better) |
| 50      | 0.4557  | 0.2082      | **+54.3%** (much better) |
| 100     | 0.5064  | 0.4861      | **+4.0%** (slightly better) |
| 200     | 0.4737  | 0.7703      | **-62.6%** (worse) |
| 500     | 0.5529  | 0.9969      | **-80.3%** (much worse) |
| 1000    | 0.6443  | 1.1538      | **-79.1%** (much worse) |

### Key Findings by Horizon

- **H=50**: Largest improvement (+54.3% vs v9). GP for smooth states (v, theta, delta) provides excellent short-horizon accuracy.
- **H=1**: Largest deterioration (-381.8% vs v9). The SiLU NODE has very high delta_dot error (0.144) at H=1, dragging down performance.
- **H=500**: Significant deterioration (-80.3%). Model diverges at long horizons.

---

## 3. Per-State NMAE Analysis

### At H=100 (Primary horizon)

| State      | Hybrid  | Wide SiLU NODE | GP Per-State | v9      | Notes                    |
|------------|---------|----------------|--------------|---------|--------------------------|
| e_y        | 0.6671  | 0.5141         | 0.2090       | 0.5064  | Worse than all baselines |
| e_psi      | 0.6847  | 0.4298         | 0.2003       | -       | Worse than all baselines |
| v          | **0.0004** | 0.0203      | 0.4983       | -       | **GP near-perfect**      |
| theta      | 0.4001  | 0.3865         | 0.1658       | -       | Slightly worse than SiLU |
| theta_dot  | 0.4793  | 0.2957         | 0.1324       | -       | Worse than SiLU          |
| delta      | 0.5649  | 0.7160         | 0.2159       | -       | Better than SiLU, worse than GP |
| delta_dot  | 0.6065  | 0.2201         | 0.1659       | -       | **Much worse than SiLU** |

### Critical Observations

1. **v state**: GP achieves NMAE=0.0004, essentially perfect prediction. This confirms GP is ideal for smooth states.
2. **delta_dot state**: Hybrid NMAE=0.6065 vs Wide SiLU NODE NMAE=0.2201. The hybrid is 2.8x worse.
3. **e_y, e_psi**: The SiLU NODE for complex states performs worse in the hybrid than when predicting all 7 states jointly (Wide SiLU NODE).

---

## 4. Root Cause Analysis

### Problem 1: delta_dot prediction collapses
- Wide SiLU NODE (all 7 states): delta_dot NMAE = 0.2201 at H=100
- Hybrid (4 states): delta_dot NMAE = 0.6065 at H=100
- **Cause**: Predicting only 4 states (e_y, e_psi, theta_dot, delta_dot) loses the coupling information from v, theta, delta. The SiLU NODE was trained to predict delta_dot independently, but it actually depends on the other states' dynamics.

### Problem 2: e_y, e_psi predictions degrade
- Wide SiLU NODE: e_y=0.5141, e_psi=0.4298 at H=100
- Hybrid: e_y=0.6671, e_psi=0.6847 at H=100
- **Cause**: Same coupling issue. The cross-state dependencies are broken when splitting the prediction.

### Problem 3: Short-horizon delta_dot error propagates
- H=1: delta_dot NMAE = 0.1439 (highest of all states)
- This initial error propagates and amplifies through the trajectory.

### Problem 4: Training data subsampling
- Only 30k of 150k samples used for NN training
- The loss was still high (355.99 at epoch 200), suggesting underfitting
- Full 150k training would likely improve but the fundamental coupling problem remains.

---

## 5. Why the Hybrid Approach Failed

The fundamental assumption was wrong: **the states are not separable into independent prediction groups.**

The vehicle dynamics have strong cross-state couplings:
- delta_dot depends on v (steering rate is speed-dependent)
- e_y depends on theta and v (lateral error depends on heading and speed)
- theta_dot depends on delta (turning rate depends on steering angle)

When GP predicts v, theta, delta independently, and SiLU NODE predicts e_y, e_psi, theta_dot, delta_dot:
- The SiLU NODE never sees the actual next-step v, theta, delta during rollout
- It only sees the current state, but the GP predictions have different error characteristics than the true dynamics
- This creates a **distribution shift** at rollout time vs training time

The Wide SiLU NODE (all 7 states jointly) avoids this by maintaining the coupling in a single model.

---

## 6. 75% Improvement Target

**Target**: 75% improvement over v9 baseline (Primary < 0.1278)

**Result**: Primary = 0.7511, which is 47% WORSE than v9 (0.5110)

**Verdict**: NOT achieved. The hybrid model is significantly worse than the v9 baseline.

---

## 7. Recommendations

### Short-term (abandon hybrid approach)
1. **Keep Wide SiLU NODE (Primary=0.3930)** as the best model
2. Focus on improving it rather than splitting states

### Medium-term improvements for Wide SiLU NODE
1. **Increase training data**: Use full 150k samples with larger batch size
2. **Longer training**: 500+ epochs with early stopping
3. **Learning rate tuning**: Try cosine schedule with warmup
4. **Ensemble of seeds**: Average 3-5 models trained with different seeds

### Long-term research directions
1. **Physics-informed hybrid**: Instead of splitting states, use GP as a correction term: `f(x) = NODE(x) + GP_residual(x)`
2. **Attention-based coupling**: Use cross-attention between state groups to maintain coupling information
3. **Teacher forcing during rollout**: Feed back GP predictions to the NODE during training to handle distribution shift
4. **Residual GP on top of NODE**: Train NODE first, then fit GP on the residuals (hierarchical approach)

---

## 8. Summary

| Metric | Value | Status |
|--------|-------|--------|
| PrimaryLongHorizonScore | 0.7511 | FAIL (target: <0.1278) |
| vs v9 baseline | -47.0% | FAIL |
| vs Wide SiLU NODE | -91.1% | FAIL |
| vs Per-State Residual | -88.4% | FAIL |
| Best horizon improvement | H=50 (+54.3% vs v9) | Partial |
| Worst horizon degradation | H=1 (-381.8% vs v9) | FAIL |

**Conclusion**: The hybrid SiLU NODE + GP approach fundamentally fails because splitting state prediction breaks the dynamical coupling between states. The Wide SiLU NODE (all 7 states jointly) remains the best approach at Primary=0.3930.
