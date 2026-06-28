# Stage 6: Model Selection Summary

**Timestamp**: 20260617
**LQR Config**: Q500_D50_R02 (Q=[500,50,10,1], R=0.2)
**Data**: stanley_ref injection, epsilon_range=[-0.1, 0.1]

## Progressive Training Results

| Dataset | val_loss | model_error | SINDy error | 5-step error | Diverges |
|---------|----------|-------------|-------------|--------------|----------|
| 30k     | 0.00736  | 0.00709     | 2.913       | 1.115        | Yes      |
| 150k    | 0.00473  | 0.00468     | 2.888       | 0.018        | No       |
| 450k    | 0.00432  | 0.00421     | 2.855       | 0.035        | No       |
| **750k**| **0.00413** | **0.00406** | **2.843** | **0.013** | **No** |

## Selection

- **best_return**: 750k (lowest model error, best 5-step prediction)
- **best_safety**: 750k (no divergence, lowest uncertainty p95)
- **best_balanced**: 750k (dominates all metrics)

**Winner**: 750k model saved as `checkpoints/best_model.pt`

## Key Observations

1. **30k insufficient**: Model diverges on 5-step rollout — too little data
2. **150k→450k diminishing returns**: val_loss drops 0.00473→0.00432 (9%)
3. **450k→750k small gain**: val_loss drops 0.00432→0.00413 (4%)
4. **750k optimal**: Best across all metrics, no divergence

## Files

- `checkpoints/ensemble_30k.pt` — 30k model
- `checkpoints/ensemble_150k.pt` — 150k model
- `checkpoints/ensemble_450k.pt` — 450k model
- `checkpoints/ensemble_750k.pt` — 750k model
- `checkpoints/best_model.pt` — best model (copy of 750k)
- `residual_mppi_stanley_ref/checkpoint_ensemble_mppi.pt` — MPPI agent checkpoint
