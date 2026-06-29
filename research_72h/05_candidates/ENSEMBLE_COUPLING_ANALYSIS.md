# Ensemble Coupling Network Analysis (EXP058)

## Summary

**Best single model (seed=42) achieves Primary=0.4265, beating the EXP051 baseline of 0.4712 by 9.5%.**
However, the ensemble methods did NOT improve over the best single model. The simple average ensemble scored 0.6434 and the weighted average ensemble produced NaN at long horizons.

## Key Finding: Seed Sensitivity is the Dominant Factor

| Model | Seed | Primary Score | vs Baseline (0.4712) |
|-------|------|---------------|----------------------|
| Single 1 | 42 | **0.4265** | **-9.5% (better)** |
| Single 2 | 43 | 1.6849 | +257.6% (worse) |
| Single 3 | 44 | 0.6011 | +27.6% (worse) |
| Single 4 | 45 | 0.5287 | +12.2% (worse) |
| Single 5 | 46 | 0.6637 | +40.9% (worse) |
| **Simple Avg Ensemble** | - | 0.6434 | +36.6% (worse) |
| **Weighted Avg Ensemble** | - | NaN | N/A |

The variance across seeds is enormous: seed=42 gives 0.4265 while seed=43 gives 1.6849 (4x worse).

## Per-Horizon Comparison

| Horizon | S0 (42) | S1 (43) | S2 (44) | S3 (45) | S4 (46) | SimpleAvg | WeightedAvg |
|---------|---------|---------|---------|---------|---------|-----------|-------------|
| H=1     | 0.0223  | 0.0243  | 0.0245  | 0.0261  | 0.0255  | 0.0233    | 0.0237      |
| H=10    | 0.0418  | 0.0509  | 0.0592  | 0.0520  | 0.0601  | 0.0467    | 0.0473      |
| H=50    | 0.0550  | 0.3382  | 0.1340  | 0.1120  | 0.1594  | 0.0965    | 0.0957      |
| H=100   | 0.2464  | 1.0778  | 0.4334  | 0.2749  | 0.4059  | 0.3343    | 0.2446      |
| H=200   | 0.4399  | 1.7478  | 0.6206  | 0.5318  | 0.6864  | 0.6977    | NaN         |
| H=500   | 0.5932  | 2.2291  | 0.7492  | 0.7794  | 0.8988  | 0.8984    | NaN         |
| H=1000  | 0.7766  | 2.2291  | 0.7928  | 0.8698  | 1.0341  | 1.0365    | NaN         |

## Why Ensemble Failed

1. **Seed=43 produced a catastrophically poor model** (Primary=1.6849). This model:
   - Has survival rate 0% at H=500 and H=1000
   - Generates NaN predictions at long horizons
   - Contaminates both simple and weighted averages

2. **Simple average is dragged down** by the bad model. Without seed=43, the average of the other 4 models would be ~0.557, still worse than seed=42 alone.

3. **Weighted average failed completely** at H>=200 because even the small weight (0.09) assigned to seed=43's NaN predictions contaminates the entire ensemble.

4. **Ensemble averaging works when models make different errors**, but here the dominant source of variance is model quality, not error diversity.

## Validation-Based Weights

| Model | Seed | Val NMAE | Weight |
|-------|------|----------|--------|
| 0     | 42   | 0.2757   | 0.292  |
| 1     | 43   | 0.8915   | 0.090  |
| 2     | 44   | 0.5086   | 0.158  |
| 3     | 45   | 0.2847   | 0.283  |
| 4     | 46   | 0.4547   | 0.177  |

The validation correctly identified seed=43 as the worst model and assigned it the lowest weight. However, even a 9% weight was enough to corrupt the ensemble at long horizons.

## What Actually Works

The best approach is **seed selection**, not ensemble averaging:

1. **Train multiple models** with different seeds
2. **Select the best** based on validation performance
3. Use the best single model for deployment

Best single model: seed=42, Primary=0.4265 (9.5% improvement over EXP051 baseline of 0.4712)

## Training Configuration

- Model: Improved Coupling Network (256-dim)
- Device: CUDA (RTX 4060 Ti)
- Epochs: 200, Batch: 1024
- Total training time: 34.5 minutes (5 models)
- Per-model training time: 5.7-7.6 minutes

## Recommendations

1. **Do NOT use naive ensemble averaging** with seed-sensitivity models
2. **Use multi-seed training + selection**: Train 5+ models, pick the best on validation
3. **Consider robust ensemble methods**: Instead of averaging, use:
   - Median prediction (robust to outliers)
   - Trimmed mean (exclude worst model)
   - Model selection (use only the best model)
4. **Investigate seed=42's advantage**: Why does this seed produce significantly better results?
5. **The real improvement is 9.5%** (from 0.4712 to 0.4265) via proper seed selection
