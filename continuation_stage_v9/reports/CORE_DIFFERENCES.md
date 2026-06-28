# Core Differences Analysis

## V9 vs V12 Normalization

| Aspect | V9 | V12 |
|--------|-----|-----|
| Training targets | `deltas / (delta_std * dt)` | `deltas / state_std` |
| Model output | `ds/dt * dt` (rate * dt) | `delta / state_std` (normalized delta) |
| Integration | `s_cur + dsdt * dt` (physical space) | `s_norm + model_out` (normalized space) |
| Denormalization | Per-step: `s + dsdt * delta_std * dt` | End: `s_next_norm * state_std` |

**Key insight**: Both normalizations are mathematically equivalent for single-step. The difference is:
- V9 multi-step was broken (shuffled data + wrong integration)
- V12 multi-step is correct (sequential segments + normalized integration)

## Root Causes of V9 Baseline's "Good" Results

V9 baseline achieves H=500 NMAE=0.998 with 100% survival. Sub-agent analysis reveals this is likely **trivial prediction**:

1. **Shuffle=True** breaks temporal ordering → multi-step loss is pure noise
2. **lambda_multi=0.3** is low → model is optimized for single-step only
3. **Result**: Model learns to predict "near-zero change" → conservative constant-like predictions
4. **Consequence**: Low NMAE at H=1 (0.006) because ground truth changes are small; high NMAE at H=500 (0.998) because errors accumulate slowly but never explode

**Evidence**: Per-state NMAE at H=500 shows `delta_dot: 1.706` (very high) but `theta: 0.315` (moderate). The model preserves some states well but fails on others — consistent with "mostly constant" prediction.

## The Accuracy-Stability Tradeoff

| Config | H=1 | H=50 | H=500 | Surv@500 | Character |
|--------|-----|------|-------|----------|-----------|
| v9_baseline | 0.006 | 0.548 | 0.998 | 100% | Conservative, slow drift |
| v9_fixed_seq | 0.056 | 0.385 | 0.476 | 0% | Accurate, diverges |
| v9_fixed_seq_contract | 0.035 | 0.647 | 0.958 | 100% | Contractive, stable |
| soft_contract (0.01) | 0.046 | 0.711 | 0.987 | 100% | Weakly contractive |
| clip_seq (0.95) | 0.056 | 0.844 | 1.505 | 100% | Clipped, error accumulation |

**Critical observation**: Contractivity (lambda=0.05) improves survival but WORSENS NMAE at all horizons compared to v9_fixed_seq. Even lambda=0.01 doesn't help.

**Inference-time clipping** restores survival but dramatically worsens NMAE (H=500: 1.505 vs 0.476). Clipping creates discontinuities that the model wasn't trained on, causing error cascades.

## Sub-Agent Conclusions

### Agent A (Code Explorer)
- Confirmed V9 multi-step normalization bug (missing delta_std factor)
- Confirmed evaluation uses model.predict() with physical-limit survival checks
- V12's predict() correctly denormalizes: `s_next_norm * state_std`

### Agent B (Root Cause Analyst)
- PRIMARY: Normalization mismatch (H3) — V9's delta_std*dt factor creates different scale
- SECONDARY: Metric misalignment (H2) — training vs evaluation mismatch
- TERTIARY: Error accumulation (H4) — fundamental challenge for long rollouts

### Agent C (Solution Reviewer)
- Recommended survival-aware loss during training (but proved ineffective with H=20 curriculum)
- Contractivity is the only training-time mechanism that works

### Agent D (Counter-Argument Expert)
- "V9's 0.998 NMAE with 100% survival may be evidence of trivial constant prediction"
- "Contractivity's NMAE worsening suggests the model is learning overly conservative dynamics"
- "The accuracy-stability tradeoff may be fundamental, not fixable with simple tricks"

## Key Unresolved Questions

1. Can we achieve V12-sequential NMAE with V9-baseline survival?
2. Is the accuracy-stability tradeoff fundamental or can it be broken?
3. Would longer training (400+ epochs) with extended curriculum help?
4. Is there a sweet spot in the contractivity lambda space?
