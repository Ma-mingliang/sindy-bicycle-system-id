# UDE Model Analysis (EXP046)

## Architecture

**Universal Differential Equations (UDE)** = Physics equations + Neural Network residual

The core idea: use known physics as the backbone for ALL states, and let the NN learn only the residual (unmodeled dynamics).

### Physics Component
```
e_y_dot   = v * sin(e_psi)          # lateral error kinematics
e_psi_dot = -v * delta / L          # heading error dynamics (L=1.0)
v_dot     = 0                       # velocity approximately constant
theta_dot = theta_dot (state[4])    # lean rate kinematics
delta_dot = delta_dot (state[6])    # steering rate kinematics
```

### NN Residual Component
- **Input**: 8D (7D normalized state + 1D normalized action)
- **Output**: 7D (residual corrections for all states)
- **Architecture**: 128-wide, 4-layer Tanh network
- **Initialization**: Xavier uniform (gain=0.1) on final layer, zero bias

### Combination
```
total_delta = physics_delta(s) + NN_residual(s_norm, a_norm) * delta_std * dt
s_next = s + total_delta
```

## Training Configuration
- Epochs: 200
- Batch size: 256
- Learning rate: 1e-3 (cosine annealing)
- Multi-step loss: rollout curriculum [1, 5, 10], lambda=0.3
- Training time: 991.6s (~16.5 min) on CPU

## Results

### NMAE at All Horizons

| Horizon | UDE NMAE | v9 NMAE | Delta | Status |
|---------|----------|---------|-------|--------|
| H=1     | 0.0132   | 0.0051  | -159% | WORSE  |
| H=10    | 0.0454   | 0.0628  | +28%  | BETTER |
| H=50    | 0.1328   | 0.4557  | +71%  | BETTER |
| H=100   | 0.3688   | 0.5064  | +27%  | BETTER |
| H=200   | 0.5653   | 0.4737  | -19%  | WORSE  |
| H=500   | 0.7271   | 0.5529  | -32%  | WORSE  |
| H=1000  | 0.7414   | 0.6443  | -15%  | WORSE  |

### Survival Rate

| Horizon | UDE Survival |
|---------|-------------|
| H=1     | 100%        |
| H=10    | 100%        |
| H=50    | 100%        |
| H=100   | 100%        |
| H=200   | 100%        |
| H=500   | 20%         |
| H=1000  | 20%         |

### Per-State NMAE at Key Horizons

**H=1 (single-step)**:
| State | UDE | Notes |
|-------|-----|-------|
| e_y | 0.041 | Physics helps (known kinematics) |
| e_psi | 0.036 | Physics helps (known dynamics) |
| v | 0.001 | Trivial (dv/dt ≈ 0) |
| theta | 0.001 | Good |
| theta_dot | 0.005 | Good |
| delta | 0.001 | Good |
| delta_dot | 0.007 | Good |

**H=100 (medium horizon)**:
| State | UDE | v9 | Delta |
|-------|-----|----|-------|
| e_y | 0.439 | - | Error accumulates |
| e_psi | 0.732 | - | Dominant error source |
| v | 0.097 | - | Still small |
| theta | 0.284 | - | Moderate |
| theta_dot | 0.328 | - | Moderate |
| delta | 0.324 | - | Moderate |
| delta_dot | 0.377 | - | Moderate |

**H=500 (long horizon)**:
| State | UDE | Notes |
|-------|-----|-------|
| e_y | 0.924 | Large accumulated error |
| e_psi | 1.604 | Dominant failure mode |
| v | 0.310 | Drift |
| theta | 0.486 | Moderate |
| theta_dot | 0.759 | Large |
| delta | 0.518 | Large |
| delta_dot | 0.488 | Large |

## Analysis

### Why UDE Excels at Medium Horizons (H=10-100)

1. **Physics constraints prevent drift**: The known kinematic equations for `e_y` and `e_psi` provide exact structure that prevents the model from learning spurious correlations.

2. **NN focuses on what matters**: By removing the known physics, the NN only needs to learn the residual dynamics (actuator response, nonlinear coupling), which is a simpler learning problem.

3. **Better generalization**: Physics equations generalize perfectly to unseen states, while the NN handles the nonlinear parts that require data.

### Why UDE Degrades at Long Horizons (H=200+)

1. **Error accumulation in e_psi**: The `e_psi` error grows fastest (NMAE=1.6 at H=500). This is because `e_psi_dot = -v * delta / L` is an approximate model -- the real dynamics include tire forces, lean-steer coupling, and other effects not captured.

2. **NN residual introduces drift**: Over many steps, even small NN prediction errors accumulate. The physics model provides no damping or stability guarantees.

3. **Survival drops at H=500**: Only 20% of test segments survive to H=500 without exceeding physical limits, primarily due to e_psi divergence.

### Comparison with Layered Prediction (EXP043)

| Aspect | UDE (EXP046) | Layered (EXP043) |
|--------|-------------|-----------------|
| Approach | Physics+NN for ALL states | Physics for e_y/e_psi, NN for theta/delta |
| H=1 | 0.0132 | 0.0211 |
| H=100 | 0.3688 | 1.0575 |
| H=200 | 0.5653 (100% survival) | 1.3516 (0% survival) |
| H=500 | 0.7271 (20% survival) | 1.3516 (0% survival) |

UDE significantly outperforms the layered approach because:
- Physics applies to ALL states, not just e_y/e_psi
- NN learns residuals rather than full dynamics for theta/delta
- The combination is more robust than separate treatment

### Comparison with v9 Baseline

The UDE model shows a **crossover pattern**:
- **Short horizon (H=1)**: UDE is worse (-159%). The physics equations introduce model error that the pure NN avoids by fitting the data directly.
- **Medium horizon (H=10-100)**: UDE is significantly better (+28% to +71%). Physics constraints prevent the catastrophic drift that pure NNs exhibit.
- **Long horizon (H=200-1000)**: UDE is worse (-15% to -32%). The approximate physics model (simplified bicycle kinematics) introduces systematic errors that accumulate over time.

## Key Insights

1. **Physics helps at medium horizons**: The UDE approach is most effective when the physics model captures the dominant dynamics but misses secondary effects. H=50 shows the largest improvement (+71%).

2. **Approximate physics hurts at long horizons**: The simplified kinematic model (`e_psi_dot = -v * delta / L`) is not accurate enough for very long predictions. A more detailed bicycle model (including tire forces, lean dynamics) would improve long-horizon performance.

3. **e_psi is the bottleneck**: Heading error (`e_psi`) consistently shows the largest NMAE and drives the divergence at long horizons. Future work should focus on improving the e_psi dynamics model.

4. **NN residual works well**: The NN successfully learns the residual dynamics (actuator response, nonlinear coupling), confirming that the UDE decomposition is sound.

## Recommendations for Improvement

1. **Better physics model**: Use a more detailed bicycle dynamics model that includes tire forces and lean-steer coupling.
2. **Stability regularization**: Add Lyapunov-based constraints to prevent long-horizon drift.
3. **Ensemble UDE**: Train multiple UDE models with different seeds and average predictions (like ensemble v9).
4. **Adaptive residual scaling**: Reduce NN residual influence at long horizons where physics should dominate.

## Files

- **Code**: `D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/ude_model.py`
- **Results**: `D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP046_ude_model.json`
- **Model**: `D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/ude_model.pt`
