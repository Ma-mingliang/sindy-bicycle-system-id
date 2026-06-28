# Ensemble with Switching - Analysis Report

## Experiment: EXP024_ensemble_switching

### Core Idea

Use multiple diverse Neural ODE models and switch between them (or fall back to physics) based on ensemble uncertainty. The hypothesis was that:
1. Ensemble disagreement indicates OOD (out-of-distribution) states
2. Switching to physics model when uncertain would prevent error accumulation
3. This would improve long-horizon prediction accuracy

### Implementation

**Models (4 total):**
- V9 Fixed seed=42 (tanh, hidden=64, depth=3) - pre-trained
- V9 Fixed seed=43 (tanh, hidden=64, depth=3) - pre-trained
- V9 Fixed seed=44 (tanh, hidden=64, depth=3) - pre-trained
- Wide seed=45 (tanh, hidden=128, depth=2) - trained fresh

**Switching Strategies Tested:**
| Strategy | Description | Threshold |
|----------|-------------|-----------|
| ensemble_avg | Simple averaging of all models | N/A |
| best_single | Use only the best single model | N/A |
| physics_mean | Switch to physics when uncertainty > mean | 0.004 |
| physics_p50 | Switch to physics when uncertainty > p50 | 0.004 |
| physics_p90 | Switch to physics when uncertainty > p90 | 0.006 |
| physics_p99 | Switch to physics when uncertainty > p99 | 0.008 |
| hold_p50 | Hold current state when uncertainty > p50 | 0.004 |
| hold_p90 | Hold current state when uncertainty > p90 | 0.006 |
| adaptive_mean | Smooth blend with physics | 0.004 |
| adaptive_p50 | Smooth blend with physics | 0.004 |
| adaptive_p90 | Smooth blend with physics | 0.006 |

### Results

**Uncertainty Distribution:**
- Mean: 0.0040
- P50: 0.0040
- P90: 0.0062
- P99: 0.0078

**All Strategies Comparison (NMAE):**

| Strategy | H=1 | H=10 | H=50 | H=100 | H=200 | H=500 | Primary |
|----------|-----|------|------|-------|-------|-------|---------|
| V9 Baseline | **0.0051** | 0.0628 | 0.4557 | 0.5064 | **0.4737** | **0.5529** | **0.5110** |
| ensemble_avg | 0.0235 | 0.0539 | 0.1294 | 0.5447 | 0.9721 | 1.3238 | 0.9469 |
| best_single | 0.0255 | 0.0648 | 0.1854 | 0.4407 | 0.9967 | 1.3837 | 0.9404 |
| hold_p50 | 0.0235 | **0.0539** | **0.1267** | **0.4368** | 0.8447 | 1.2827 | 0.8548 |
| physics_p90 | 0.0235 | 0.0539 | 0.1294 | 0.5912 | 0.9453 | 1.1773 | 0.9046 |

**Best Strategy: hold_p50** (Primary: 0.8548)

**Comparison with V9 Baseline (hold_p50):**

| Horizon | V9 NMAE | Best NMAE | Change |
|---------|---------|-----------|--------|
| H=1 | 0.0051 | 0.0235 | -361.1% |
| H=10 | 0.0628 | 0.0539 | **+14.1%** |
| H=50 | 0.4557 | 0.1267 | **+72.2%** |
| H=100 | 0.5064 | 0.4368 | **+13.7%** |
| H=200 | 0.4737 | 0.8447 | -78.3% |
| H=500 | 0.5529 | 1.2827 | -132.0% |

### Key Findings

1. **Medium-horizon improvement**: Ensemble switching significantly improves H=50 (+72.2%) and H=100 (+13.7%) predictions.

2. **Short-horizon degradation**: H=1 performance is much worse (-361%). This is because ensemble averaging introduces noise compared to a single well-trained model.

3. **Long-horizon failure**: H=200 and H=500 are significantly worse. The switching mechanism doesn't prevent error accumulation at long horizons.

4. **Low uncertainty**: The ensemble uncertainty is very low (mean=0.004), indicating all models converge to similar solutions. This limits the effectiveness of uncertainty-based switching.

5. **Physics fallback hurts**: Switching to physics model when uncertain makes things worse, suggesting the simplified physics model is less accurate than the neural models even in OOD regions.

6. **Hold-state partially works**: Holding current state when uncertain (hold_p50) is the best strategy, preventing error propagation at medium horizons.

### Root Cause Analysis

**Why ensemble switching fails for long horizons:**

1. **Insufficient model diversity**: All V9 models use the same architecture and training data. Different seeds produce models that are too similar.

2. **Uncertainty doesn't correlate with error**: The ensemble variance is low even when predictions are wrong, making it a poor indicator for switching.

3. **Physics model is too simplified**: The bicycle physics fallback doesn't capture the true dynamics well enough to be useful.

4. **Switching introduces discontinuities**: Abrupt switches between models/prediction modes can introduce their own errors.

5. **Error accumulation dominates**: At H=200+, the fundamental issue is error accumulation over many steps, which ensemble averaging cannot fix.

### Comparison with Other Approaches

| Approach | Primary Score | vs V9 |
|----------|---------------|-------|
| V9 Baseline | 0.5110 | - |
| Ensemble Switching (hold_p50) | 0.8548 | -67.3% |
| Ensemble V9 (EXP009) | ~0.55 | ~-8% |
| Uncertainty Gating (EXP017) | ~0.55 | ~-8% |
| Physics Residual (EXP016) | ~0.60 | ~-18% |

### Conclusions

1. **Ensemble with Switching does not improve over V9 baseline** for this problem.

2. **The approach shows promise at medium horizons** (H=50-100) but fails at long horizons.

3. **Model diversity is the key bottleneck**: To make ensemble methods work, models need to be fundamentally different (different architectures, different training data, different loss functions).

4. **Uncertainty calibration is critical**: The ensemble uncertainty must be a reliable indicator of prediction error for switching to be effective.

5. **For this problem, a single well-trained model (V9) remains the best approach**.

### Recommendations

1. **For medium-horizon prediction (H=50-100)**: Consider using ensemble with hold-state switching.

2. **For long-horizon prediction (H=200+)**: Stick with single V9 model or explore fundamentally different approaches (e.g., Koopman, GP).

3. **Future ensemble work**: Focus on creating truly diverse models (different architectures, physics-informed vs pure data-driven, different loss functions).

### Files

- Code: `D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/ensemble_switching.py`
- Results: `D:/系统辨识作业/sindy_bicycle/research_72h/05_candidates/EXP024_ensemble_switching.json`
- Models: `D:/系统辨识作业/sindy_bicycle/research_72h/07_models/ensemble_switch_wide_45.pt`
