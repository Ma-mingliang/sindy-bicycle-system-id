# Patch Log

## Code Modifications

### 1. `canonical_node/neural_ode_v12.py`

**Modification A: Survival-aware loss parameters** (lines 103-106)
```python
# Added to V12Config:
use_survival_loss: bool = False
lambda_survival: float = 0.1
survival_limit_scale: float = 0.95
```
- Purpose: Enable survival-aware loss during training
- Status: Implemented but INEFFECTIVE (loss always zero with H=20 curriculum)

**Modification B: Normalized physical limits computation** (lines 856-866)
```python
if cfg.use_survival_loss:
    from .config_v9 import PHYSICAL_LIMITS
    limit_keys = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
    limits_phys = np.array([PHYSICAL_LIMITS[k] for k in limit_keys])
    self._limits_norm = torch.FloatTensor(
        limits_phys / self._state_std * cfg.survival_limit_scale
    )
```
- Purpose: Precompute normalized limits for survival penalty
- Status: Correct implementation, but penalty never triggers

**Modification C: Multi-step rollout loss signature** (lines 388-439)
```python
def _multi_step_rollout_loss(model, states, actions, dt, rollout_steps,
                             method='euler', n_substeps=2,
                             limits_norm=None) -> Tuple[torch.Tensor, torch.Tensor]:
    # Added survival penalty inside rollout loop:
    if limits_norm is not None:
        violation = torch.relu(torch.abs(s_cur) - limits_norm.unsqueeze(0))
        loss_surv = loss_surv + violation.pow(2).mean()
```
- Purpose: Penalize predicted states exceeding physical limits
- Status: Correct implementation, but violations don't occur at H=20

**Modification D: Training loop integration** (survival loss tracking)
- Added `loss_survival` to epoch logging
- Added `lambda_sur` to epoch print output
- Status: Working correctly

### 2. `run_v9_fixed_rerun.py` (NEW FILE)
- Purpose: Compare V9 baseline vs fixed versions
- Configs: v9_baseline_ref, v9_fixed_seq, v9_fixed_seq_contract, v9_tier1_survival
- Output: `raw_results/V9_FIXED_RERUN.json`

### 3. `run_repair_experiments.py` (NEW FILE)
- Purpose: Test soft contractivity and inference-time clipping
- Configs: soft_contract, clip_seq, soft_contract_clip, very_soft_contract, very_soft_contract_clip
- Output: `raw_results/REPAIR_EXPERIMENTS.json`

### 4. `run_hybrid_experiments.py` (NEW FILE)
- Purpose: Test blended model approaches
- Configs: blend_0.5, blend_0.7_seq, blend_0.3_seq, blend_0.9_seq, safety_blend_0.7/0.5/0.3
- Output: `raw_results/HYBRID_EXPERIMENTS.json`

## Checkpoints Generated

| Checkpoint | Config | Status |
|------------|--------|--------|
| V9_FIXED_v9_fixed_seq.pt | Sequential batching only | Best NMAE, 0% survival |
| V9_FIXED_v9_fixed_seq_contract.pt | Sequential + contractivity | 100% survival, moderate NMAE |
| V9_FIXED_v9_tier1_survival.pt | Sequential + contractivity + survival loss | Identical to above |
| REPAIR_soft_contract.pt | lambda=0.01 contractivity | 100% survival, worse NMAE |
| REPAIR_soft_contract_clip.pt | lambda=0.01 + clipping | 100% survival, worse NMAE |
| REPAIR_very_soft_contract.pt | lambda=0.005 contractivity | 100% survival, worse NMAE |
| REPAIR_very_soft_contract_clip.pt | lambda=0.005 + clipping | 100% survival, worse NMAE |
