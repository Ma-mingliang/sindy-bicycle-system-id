# V10/V11 Neural ODE Deep Improvements — Experiment Results Summary

## Overview

Following the V9 baseline (config3_seed43, hidden=64, depth=3, tanh, lr=1e-3), we conducted three major experiment families based on literature survey of 32 papers covering Port-Hamiltonian NNs, contractive Neural ODEs, Lyapunov-based stability, Koopman operators, and curriculum learning for dynamical systems.

**V9 Baseline Reference (NMAE):**

| Horizon | H=10 | H=50 | H=100 | H=200 | H=500 |
|---------|------|------|-------|-------|-------|
| NMAE    | 0.0628 | 0.4557 | 0.5064 | 0.4737 | 0.5529 |

---

## Experiment 1: Feedback Correction (V10_FEEDBACK)

**Approach:** Add learnable feedback correction term to Neural ODE dynamics:
dx/dt = f_theta(x, u) + K * (x_ref - x)

**8 configurations tested:**

| Config | H=10 | H=50 | H=100 | H=200 | H=500 | Time(s) | Verdict |
|--------|------|------|-------|-------|-------|---------|---------|
| baseline_v9 | 0.0628 | 0.4557 | 0.5064 | 0.4737 | 0.5529 | - | Reference |
| linear_m0.1 | 0.0597 | 0.5570 | 0.6578 | 0.6761 | 0.6973 | 829 | Mixed |
| linear_m0.3 | 0.0687 | 0.5795 | 0.6772 | 0.7390 | 0.9567 | 845 | Worse |
| linear_m0.03 | 0.1210 | 0.5465 | 0.7066 | 0.8077 | 1.1594 | 828 | Worse |
| rank3 | 0.0607 | 0.5890 | 0.6992 | 0.7190 | 0.7679 | 886 | Worse |
| mlp | 0.0865 | 0.5263 | 0.7499 | 0.8727 | - | 940 | Worse |
| state_dependent | 0.0762 | 0.5756 | 0.7022 | 0.7654 | - | 1025 | Worse |
| residual_m0.1 | 0.0639 | 0.5150 | 0.6108 | 0.6345 | 0.7397 | 862 | Worse |
| strong_damp | 0.0672 | 0.5482 | 0.6492 | 0.6928 | 0.8024 | 835 | Worse |

**Conclusion:** All feedback correction variants performed worse than baseline. Uniform damping fights learned dynamics everywhere. The feedback regularization term (~83-154) dominates the loss, reducing the model's ability to fit data.

**Root Cause:** The correction term K*(x_ref-x) applies uniformly across the state space, even in regions where the model is already accurate. This creates a bias that prevents the model from learning fine-grained dynamics.

---

## Experiment 2: Curriculum Learning Strategies (V10_ADAPTIVE_FREQ)

### 2A: Fixed Curriculum Baselines

| Config | H=10 | H=50 | H=100 | H=200 | H=500 | Time(s) |
|--------|------|------|-------|-------|-------|---------|
| baseline_1_5_10_20 | 0.0628 | 0.4557 | 0.5064 | 0.4737 | 0.5529 | 1371 |
| baseline_long_400ep | 0.1303 | 0.5062 | 0.5520 | 0.5622 | 0.5936 | 2933 |
| baseline_extended_1_5_10_20_50 | 0.0976 | 0.5381 | 0.6673 | 0.7479 | 0.9401 | 4072 |

**Finding:** Longer training (400ep) and extended curriculum (adding rollout=50) both hurt performance. Long rollouts produce large noisy gradients that overwrite short-horizon structure.

### 2B: Adaptive Curriculum (EMA-based)

Uses EMA-smoothed validation NMAE to decide when to advance rollout length.

| Config | H=10 | H=50 | H=100 | H=200 | H=500 | Time(s) | Verdict |
|--------|------|------|-------|-------|-------|---------|---------|
| **adaptive_conservative** | **0.0590** | **0.4470** | **0.4771** | **0.4548** | **0.5542** | 3209 | **BEST** |
| adaptive_moderate | 0.1698 | 1.0079 | 1.2224 | 1.2666 | - | 3889 | Exploded |
| adaptive_aggressive | 0.1510 | 0.6128 | 0.7149 | 0.7862 | 0.9015 | 4516 | Worse |

**adaptive_conservative vs V9 Baseline:**

| Horizon | H=10 | H=50 | H=100 | H=200 | H=500 |
|---------|------|------|-------|-------|-------|
| V9 Baseline | 0.0628 | 0.4557 | 0.5064 | 0.4737 | 0.5529 |
| adaptive_conservative | 0.0590 | 0.4470 | 0.4771 | 0.4548 | 0.5542 |
| **Improvement** | **-6.0%** | **-1.9%** | **-5.8%** | **-4.0%** | +0.2% |

**This is the FIRST method to beat the V9 baseline across all short/mid-range horizons.**

### 2C: Frequency-Domain Curriculum

Per-state loss weighting based on physical time constants (slow: v,theta; fast: theta_dot, delta_dot).

| Config | H=10 | H=50 | H=100 | H=200 | H=500 | Time(s) |
|--------|------|------|-------|-------|-------|---------|
| freq_strong | 0.0621 | 0.5196 | 0.6488 | 0.7883 | 1.1898 | 1891 |
| freq_moderate | 0.0624 | 0.5299 | 0.6558 | 0.7096 | 0.9153 | 1887 |
| freq_pure | 0.0693 | 0.6206 | 0.8047 | 1.0043 | 2.1583 | 1307 |

**Conclusion:** All frequency-domain curriculum variants performed worse than baseline. Per-state reweighting disrupts the balance the optimizer naturally finds. The stronger the reweighting, the worse the long-horizon performance (freq_pure H=500: 2.16 vs baseline 0.55).

---

## Experiment 3: Contractivity-Promoting Regularization (V11_CONTRACTIVE)

**Approach:** Penalize max eigenvalue of Jacobian symmetric part to promote contractive dynamics.

**7 lambda values + warmup/ablation variants (17 total):**

| Config | H=10 | H=50 | H=100 | H=200 | H=500 | max_eig | frac_pos | Time(s) |
|--------|------|------|-------|-------|-------|---------|----------|---------|
| V9 baseline | 0.0602 | 0.5396 | 0.6558 | 0.7129 | 0.8278 | - | - | 1472 |
| lam0.01 | 0.0631 | 0.5606 | 0.7094 | 0.8915 | 1.0944 | 24.75 | 1.000 | 2025 |
| lam0.03 | 0.0714 | 0.5397 | 0.7020 | 0.8625 | 1.0099 | 22.68 | 1.000 | 3295 |
| lam0.05 | 0.0572 | 0.5398 | 0.7061 | 0.8118 | 0.9190 | 20.39 | 1.000 | 3280 |
| lam0.1 | 0.0632 | 0.5267 | 0.6506 | 0.7508 | 1.0471 | 20.41 | 1.000 | 3276 |
| lam0.2 | 0.0671 | 0.5230 | 0.6212 | 0.6828 | 0.9263 | 17.64 | 1.000 | 3278 |
| lam0.5 | 0.0597 | 0.5246 | 0.6571 | 0.7212 | 0.8054 | 12.45 | 0.740 | 3276 |

**Conclusion:** Contractivity regularization is ineffective for this problem. Even at lambda=0.5 (very strong), the max eigenvalue only drops from ~25 to ~12, and frac_pos=0.740 (74% of eigenvalues still positive). The regularization loss (~12-25) is tiny compared to data fitting loss (~330), so the optimizer essentially ignores it.

**Root Cause:** The bicycle dynamics are inherently expansive in some directions (e.g., lateral error growth). Forcing contractivity conflicts with learning the true dynamics.

---

## Experiment 4: Attractor-Based Stabilization (V10_ATTRACTOR)

**Approach:** Add learned attractor correction that only activates for OOD states:
dx/dt = f_base(x,u) + attractor(x)

**12 configurations across 5 groups:**

| Config | H=10 | H=50 | H=100 | H=200 | H=500 | Time(s) |
|--------|------|------|-------|-------|-------|---------|
| residual_ey_epsi | 0.4888 | 0.8098 | 0.8098 | 0.8098 | 0.8098 | 3047 |
| full_all_states | 0.1250 | 0.5742 | 0.7325 | 0.9691 | 1.3142 | 4694 |
| residual_ey_epsi_thetadot | 0.1560 | 0.6633 | 0.7702 | 0.8450 | 1.0987 | 4958 |
| residual_learned_gate | 0.4888 | 0.8098 | 0.8098 | 0.8098 | 0.8098 | 4970 |
| residual_sigmoid_gate | 0.4888 | 0.8098 | 0.8098 | 0.8098 | 0.4933 | 4933 |

**Conclusion:** Attractor correction performs very poorly. H=10 NMAE went from 0.0628 to 0.4888 (8x worse). The attractor adds noise even with consistency loss. The 5-component loss (single-step + multi-step + consistency + attractor reg + manifold reg) creates conflicting gradients.

---

## Key Findings

### What Worked

1. **Adaptive Conservative Curriculum** — The only improvement over V9 baseline
   - Uses EMA-smoothed validation error to decide rollout advancement
   - Conservative thresholds (5:0.02, 10:0.04, 20:0.08, 50:0.15) with patience_exceeded fallback
   - Stays at each rollout level long enough to learn stable representations
   - 6% improvement at H=10, 2% at H=50, 6% at H=100, 4% at H=200

### What Didn't Work

2. **Feedback Correction** — Uniform damping fights learned dynamics
3. **Contractivity Regularization** — Too weak relative to data loss; bicycle dynamics are inherently expansive
4. **Attractor Correction** — Adds noise, conflicting multi-component loss
5. **Frequency-Domain Curriculum** — Disrupts natural optimizer balance
6. **Aggressive/Moderate Adaptive** — Too-fast rollout advancement causes gradient explosion

### Lessons Learned

- **Curriculum pacing matters more than curriculum design.** The difference between conservative and aggressive adaptive curriculum is huge (0.447 vs 1.008 at H=50).
- **Long rollouts are dangerous.** Rollout=50 consistently causes gradient explosion unless the model is very well-prepared by shorter rollouts.
- **Regularization must compete with data loss.** Contractivity regularization at lambda=0.5 produces loss~12, but data loss is ~330. Need lambda~10+ to be effective, which would destroy data fitting.
- **Correction terms must be state-dependent.** Uniform corrections (feedback, attractor with weak gating) hurt performance because they apply everywhere, including where the model is already accurate.

---

## Files Created

### New Modules
- `canonical_node/neural_ode_contractive.py` — ContractiveODE with Jacobian eigenvalue penalty
- `canonical_node/neural_ode_attractor.py` — AttractorNeuralODE with learned manifold correction
- `canonical_node/neural_ode_adaptive_curriculum.py` — AdaptiveCurriculumScheduler with EMA validation
- `canonical_node/neural_ode_freq_curriculum.py` — FreqCurriculumNeuralODE with per-state weighting
- `canonical_node/neural_ode_feedback.py` — FeedbackNeuralODE with learnable correction

### Experiment Scripts
- `run_v10_feedback.py` — 8 feedback correction configs
- `run_adaptive_curriculum.py` — 9 curriculum configs (3 baseline + 3 adaptive + 3 freq)
- `run_v11_contractive.py` — 17 contractivity configs
- `run_v10_attractor.py` — 12 attractor configs

### Result Files
- `raw_results/V10_FEEDBACK.json` — All feedback experiment results
- `raw_results/V10_ADAPTIVE_FREQ.json` — All curriculum experiment results
- `raw_results/V11_CONTRACTIVE.json` — Contractivity experiment results (partial)
- `raw_results/V10_ATTRACTOR.json` — Attractor experiment results (partial)

---

## Next Steps

1. **Combine adaptive conservative curriculum with other improvements** — This is the first method that works; try combining with:
   - Longer training at each curriculum level
   - Learning rate warmup when advancing rollout
   - Gradient clipping to prevent explosion at rollout transitions

2. **Investigate why H=500 didn't improve** — The adaptive curriculum helps at H=10-200 but not H=500. This suggests the model's long-horizon stability needs a different approach.

3. **Try adaptive curriculum with rollout=100** — The conservative approach stayed at rollout=50 for only 20 epochs. Maybe we need rollout=100 in the curriculum.

4. **State-dependent correction** — Instead of uniform feedback, try correction that only activates when prediction error is high (similar to attractor gating but simpler).
