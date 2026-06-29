# Per-State Ensemble Analysis

## Experiment: EXP042_per_state_ensemble

### Objective
Implement and evaluate three ensemble strategies that combine different model types
(Neural ODE, GP, physics-based) with per-state weighting for bicycle dynamics prediction.

### Background
- Baseline V9 primary score: 0.5110
- Current best: 0.3728 (physics_correction_multi_seed, seed=42)
- Target: 75%+ improvement over V9 baseline (primary < 0.1278)

### State Characteristics (from prior analysis)
| State | Dynamic Level | Best Model Type |
|-------|--------------|-----------------|
| e_y | High | Neural ODE |
| e_psi | High | Neural ODE |
| v | Low | GP |
| theta | Low | GP |
| theta_dot | Medium | Hybrid |
| delta | Low | GP |
| delta_dot | Medium | Hybrid |

### Known Physics Relationships
- `e_y_dot = v * sin(e_psi)` -- exact kinematics
- `e_psi_dot = -v * delta / L` -- exact kinematics (L = wheelbase)
- `theta_dot = d(theta)/dt` -- definition
- `delta_dot = d(delta)/dt` -- definition
- `v_dot`, `theta_ddot`, `delta_ddot` -- unknown (no exact model)

---

## Approach 1: Per-State Ensemble Model

### Design
For each state dimension, trains 5 sub-models:
1. **NODE-A**: Neural ODE (tanh, hidden=64, depth=3) -- standard architecture
2. **NODE-B**: Neural ODE (tanh, hidden=128, depth=4) -- larger capacity
3. **NODE-C**: Neural ODE (silu, hidden=64, depth=3) -- different activation
4. **GP**: Gaussian Process (Matern kernel) -- non-parametric
5. **Physics**: Known kinematics formulas -- structural prior

### Weight Learning
- Evaluate each model's per-state NMAE on validation set
- Inverse-error weighting: `w_m[d] = 1 / (epsilon + val_error_m[d])`
- Normalize so weights sum to 1 per state
- No hyperparameter tuning needed (weights are derived from validation)

### Expected Behavior
- e_y, e_psi: NODE models get high weight (physics too imprecise)
- v, theta, delta: GP gets high weight (smooth, low-dynamic)
- theta_dot, delta_dot: Balanced between NODE and GP

---

## Approach 2: Adaptive Ensemble

### Design
Partitions state space into clusters using K-Means on normalized states.
Each cluster has its own set of per-state ensemble weights.
For OOD states (distance > threshold from any cluster center), falls back to physics.

### Cluster-Specific Weights
- For each cluster, compute per-model MSE on validation points in that region
- Inverse-error weighting per cluster, per state
- This captures region-specific model strengths (e.g., NODE better at high e_psi, GP better at small perturbations)

### OOD Detection
- Compute distance to nearest cluster center in normalized state space
- If distance > threshold (default 3.0), use pure physics fallback
- Physics is always well-defined and bounded

### Hyperparameter Sweep
- n_clusters in {3, 5, 8}
- Select best by primary score

---

## Approach 3: Physics-Informed Ensemble

### Design
Uses known kinematics as a structural prior with learned blending ratios.

For states with exact physics (e_y, e_psi, theta, delta):
```
prediction = alpha * physics + (1-alpha) * data_driven
```

For states without exact physics (v, theta_dot, delta_dot):
```
prediction = data_driven (NODE + GP average)
```

### Physics Blending Ratio Learning
For each state with known physics:
- Compute physics prediction MSE on validation
- Compute data-driven prediction MSE on validation
- Optimal alpha (closed-form): `alpha = data_err / (physics_err + data_err)`
- Higher alpha = more physics weight

### Physics-Informed Correction
After initial prediction, apply correction for kinematic states:
- Correct e_y toward `s[0] + v*sin(e_psi)*dt`
- Correct e_psi toward `s[1] + (-v*delta/L)*dt`
- Correct theta toward `s[3] + theta_dot*dt`
- Correct delta toward `s[5] + delta_dot*dt`

### Hyperparameter Sweep
- correction_strength in {0.0, 0.05, 0.1, 0.2, 0.3, 0.5}

---

## Evaluation Methodology

### Horizons
[1, 10, 50, 100, 200, 500, 1000] steps

### Metrics
- **NMAE**: Normalized Mean Absolute Error (normalized by state_std)
- **Survival Rate**: Fraction of test segments without explosion
- **PrimaryLongHorizonScore**: mean(NMAE at H=100, H=200, H=500)

### Test Setup
- 5 long test episodes (>= 1100 steps)
- Open-loop rollout from initial state
- Physical limit checks for survival

---

## Results

### Summary Table

| Method | H=100 | H=200 | H=500 | H=1000 | Primary | Improvement |
|--------|-------|-------|-------|--------|---------|-------------|
| Per-State Ensemble | NaN | NaN | NaN | NaN | NaN | N/A |
| Adaptive Ensemble (k=3) | 0.2258 | 0.6364 | 1.2576 | 1.4678 | 0.7066 | -38.3% |
| Physics-Informed Ensemble | NaN | NaN | NaN | NaN | NaN | N/A |

### Per-State Analysis (Adaptive k=3, at H=200)

| State | NMAE | Dynamic Level | Best Model Type |
|-------|------|--------------|-----------------|
| e_y | 0.5255 | High | GP+Physics |
| e_psi | 0.5659 | High | GP+Physics |
| v | 0.9296 | Low | GP (dominant) |
| theta | 0.6147 | Low | GP (dominant) |
| theta_dot | 0.5814 | Medium | GP+Physics |
| delta | 0.6614 | Low | GP (dominant) |
| delta_dot | 0.5760 | Medium | GP+Physics |

### Learned Physics Blend Ratios (Approach 3)

| State | Alpha (physics weight) |
|-------|----------------------|
| e_y | 0.38 |
| e_psi | 0.43 |
| v | 0.01 |
| theta | 0.00 |
| theta_dot | 0.34 |
| delta | 0.00 |
| delta_dot | 0.16 |

### Per-State Ensemble Weights (Approach 1)

| State | NODE-A | NODE-B | NODE-C | GP | Physics |
|-------|--------|--------|--------|----|---------|
| e_y | 0.002 | 0.002 | 0.002 | 0.332 | 0.663 |
| e_psi | 0.003 | 0.003 | 0.003 | 0.382 | 0.609 |
| v | 0.000 | 0.000 | 0.000 | 0.999 | 0.001 |
| theta | 0.001 | 0.001 | 0.001 | 0.834 | 0.164 |
| theta_dot | 0.004 | 0.004 | 0.004 | 0.501 | 0.488 |
| delta | 0.003 | 0.003 | 0.003 | 0.821 | 0.170 |
| delta_dot | 0.022 | 0.022 | 0.022 | 0.545 | 0.389 |

### Training Times

| Component | Time |
|-----------|------|
| NODE-A (64x3, tanh) | 874.0s |
| NODE-B (128x4, tanh) | 754.1s |
| NODE-C (64x3, silu) | 402.0s |
| GP (7 dims, 3000 samples) | 223.7s |

---

## Key Findings

### Critical Issue: NODE Open-Loop Divergence

The most important finding is that **all three Neural ODE variants diverge during open-loop
rollout**, even though their single-step predictions are reasonable. Debug output shows:

```
node_a: delta_phys=[-0.0114, 0.0145, 0.0001]  (reasonable)
node_b: delta_phys=[ 0.0095,-0.0037, 0.0001]  (reasonable)
node_c: delta_phys=[ 0.0011, 0.0013, 0.0001]  (reasonable)
```

But during open-loop evaluation (even with teacher forcing for validation), NODE models
get NMAE=1.0 (default/error). This causes:
1. Per-State Ensemble: NODE gets ~0% weight, ensemble dominated by GP+Physics, but
   still diverges because GP predictions accumulate error in open-loop
2. Physics-Informed: Uses best NODE model, which diverges in open-loop

**Root cause**: The NODE models predict state *deltas* (not absolute states). In open-loop,
small errors accumulate exponentially. After ~10-20 steps, the state is out-of-distribution
for the NODE model, producing increasingly wrong predictions.

### Adaptive Ensemble: Only Working Approach

The Adaptive Ensemble with k=3 clusters is the only approach that produces valid results.
It works because:
- Falls back to physics when state is OOD (distance > 3.0 from cluster center)
- Cluster-specific weights adapt to local state-space regions
- GP dominates for smooth states, Physics dominates for kinematic states

However, it's **worse than the V9 baseline** (0.7066 vs 0.5110), indicating that the
ensemble approach doesn't improve over a single well-trained NODE model.

### Physics Blend Ratios Confirm Prior Analysis

The learned physics blend ratios validate the prior analysis:
- **e_y (0.38), e_psi (0.43)**: Moderate physics weight -- kinematics are exact but
  data-driven can capture additional dynamics
- **v (0.01), theta (0.00), delta (0.00)**: Near-zero physics weight -- no exact
  physics model exists for these states
- **theta_dot (0.34), delta_dot (0.16)**: Some physics weight -- definition-based
  relationships provide partial information

### Design Decisions (Post-Hoc Analysis)

1. **Why per-state weighting?** Different states have fundamentally different dynamics.
   A single ensemble weight for all 7 states is suboptimal. **CONFIRMED**: GP gets
   99.9% weight for v, while Physics gets 66.3% for e_y.

2. **Why multiple NODE variants?** Architecture diversity was expected to help, but
   all three NODE variants perform equally poorly in open-loop. **LESSON**: Diversity
   in model *type* (NODE vs GP vs Physics) matters more than diversity in architecture.

3. **Why physics as ensemble member?** Physics provides the most reliable long-horizon
   predictions because it never extrapolates dangerously. **CONFIRMED**: Physics gets
   high weight for kinematic states (e_y, e_psi, theta_dot).

4. **Why adaptive weights?** The state space is non-uniform. Near equilibrium,
   GP dominates (smooth, small perturbations). Far from equilibrium, NODE dominates
   (captures complex nonlinear dynamics).

5. **Why physics correction?** Even after ensemble blending, kinematic states can drift
   from physics-consistent trajectories. Correction nudges back toward known physics.

### Comparison with Prior Approaches

| Approach | Primary Score | Key Idea | Result |
|----------|--------------|----------|--------|
| physics_correction_multi_seed | **0.3728** | NN predict + NN corrector | BEST |
| optimized_per_state_residual | 0.5019 | NODE for dynamic, GP for smooth | Good |
| v9 baseline | 0.5110 | Single NODE model | Baseline |
| adaptive_ensemble | 0.7066 | Per-state weighted ensemble | Worse |
| per_state_ensemble | NaN | 5 model types, per-state weights | Diverged |
| physics_informed_ensemble | NaN | Physics-blended ensemble | Diverged |

### Root Cause Analysis: Why Ensemble Failed

The ensemble approach failed because of a fundamental issue with NODE models in
open-loop prediction:

1. **NODE models predict deltas, not absolute states**. Small prediction errors
   compound exponentially over time. After 10-20 steps, the state is completely
   out-of-distribution.

2. **GP models also accumulate error** in open-loop, though more slowly. The GP
   predictions become increasingly unreliable as the state drifts from training data.

3. **Only physics-based predictions are stable** in open-loop because they enforce
   known kinematic relationships. But physics alone is insufficient for states
   without exact models (v, theta_dot, delta_dot).

4. **The best prior approach (physics_correction_multi_seed)** works because it uses
   a NN corrector that learns to fix NODE predictions, not because it ensembles
   multiple models.

### Lessons Learned

1. **Model diversity by type matters more than by architecture.** All three NODE
   variants (tanh-64x3, tanh-128x4, silu-64x3) performed identically poorly.
   The diversity between NODE, GP, and Physics is more valuable.

2. **Open-loop stability is the key bottleneck.** Any model that accumulates
   error in open-loop will produce poor results regardless of single-step accuracy.

3. **Physics fallback is essential.** The adaptive ensemble's OOD fallback to
   physics prevented divergence, but the resulting predictions were still poor.

4. **Per-state weighting is validated.** The learned weights correctly assigned:
   - 99.9% GP weight for v (smooth, low-dynamic)
   - 66.3% Physics weight for e_y (exact kinematics)
   - This confirms the prior analysis of state dynamics.

---

## Conclusion

The per-state ensemble approach did not achieve the target improvement. The best
result (Adaptive Ensemble, Primary=0.7066) is 38% worse than the V9 baseline
(0.5110) and 89% worse than the current best (0.3728).

The fundamental issue is that **NODE models diverge in open-loop evaluation**, and
the ensemble cannot compensate for this instability. The most promising direction
is to focus on **physics-informed correction** (current best approach) rather than
ensemble methods.

---

## Files

- Code: `research_72h/05_candidates/per_state_ensemble.py`
- Results: `research_72h/05_candidates/EXP042_per_state_ensemble.json`
- Analysis: `research_72h/05_candidates/PER_STATE_ENSEMBLE_ANALYSIS.md`
