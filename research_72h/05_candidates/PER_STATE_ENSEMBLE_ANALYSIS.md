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

**Note: Experiment is still running. Results will be populated upon completion.**

### Summary Table

| Method | H=100 | H=200 | H=500 | H=1000 | Primary | Improvement |
|--------|-------|-------|-------|--------|---------|-------------|
| Per-State Ensemble | -- | -- | -- | -- | -- | -- |
| Adaptive Ensemble | -- | -- | -- | -- | -- | -- |
| Physics-Informed Ensemble | -- | -- | -- | -- | -- | -- |

### Per-State Analysis (at H=200)
*To be populated*

---

## Key Insights

### Design Decisions

1. **Why per-state weighting?** Different states have fundamentally different dynamics.
   A single ensemble weight for all 7 states is suboptimal.

2. **Why multiple NODE variants?** Architecture diversity (different depths, activations)
   provides complementary predictions. Tanh is smooth, SiLU has different gradient flow.

3. **Why physics as ensemble member?** Known kinematics provide exact relationships for
   4 of 7 states. Physics never extrapolates dangerously -- it's always bounded.

4. **Why adaptive weights?** The state space is non-uniform. Near equilibrium,
   GP dominates (smooth, small perturbations). Far from equilibrium, NODE dominates
   (captures complex nonlinear dynamics).

5. **Why physics correction?** Even after ensemble blending, kinematic states can drift
   from physics-consistent trajectories. Correction nudges back toward known physics.

### Comparison with Prior Approaches

| Approach | Key Idea | Limitation |
|----------|----------|------------|
| physics_correction_multi_seed | NN predict + NN corrector | Single model, no per-state diversity |
| ensemble_switching | Multiple NODEs + switching | All models same type (NODE only) |
| optimized_per_state_residual | NODE + GP per state | Fixed assignment, no blending |
| **per_state_ensemble** | **5 model types, per-state weights** | **Enables per-state diversity** |

---

## Files

- Code: `research_72h/05_candidates/per_state_ensemble.py`
- Results: `research_72h/05_candidates/EXP042_per_state_ensemble.json`
- Analysis: `research_72h/05_candidates/PER_STATE_ENSEMBLE_ANALYSIS.md`
