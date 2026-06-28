# Divergence Handling Audit

> Generated: 2026-06-25
> Agent: Agent B (Evaluation Protocol Auditor)

---

## 1. Divergence Detection Method

Two divergence indicators are used:
1. **endpoint_error = NaN**: Model state contains NaN at final step (hard divergence)
2. **first_exceed_3sigma**: Step where error exceeds 3x training std (soft divergence)

### Survival Steps Bug

**CONFIRMED BUG**: `run_mode_b_fixed` sets `survival_steps = n_valid` (the requested
horizon) regardless of actual divergence. The NaN padding after the break statement
correctly fills remaining steps, but `survival_steps` is not updated.

```python
# BUG in eval_modes_v2.py, run_mode_b_fixed:
n_valid = min(n_steps, len(actions))  # set BEFORE loop
for i in range(n_valid):
    ...
    if np.any(np.isnan(s_model)) or np.any(np.abs(s_model) > 100):
        remaining = n_valid - i - 1
        states_model.extend([np.full(4, np.nan)] * remaining)
        break
# survival_steps = n_valid  <-- WRONG: should be i+1 after break
```

**Impact**: Reported survival_steps always equals requested horizon, even when model
diverged earlier. The NaN padding and metric computation (NaN filtering) are correct,
but survival_steps metadata is misleading.

---

## 2. Per-Model Divergence Analysis (Mode B, 5 seeds)

### linearized_model

| Seed | Horizon | MAE | Endpoint NaN? | First Exceed 3sigma | Divergence Step |
|------|---------|-----|---------------|--------------------|-----------------|
| 42 | 10 | 0.1193 | no | none | none |
| 42 | 50 | 0.0631 | no | none | none |
| 42 | 100 | 0.2367 | no | 79, 96 | none |
| 42 | 500 | 6.9595 | YES | 101, 79, 127, 96 | 79 |
| 42 | 1000 | 6.9595 | YES | 101, 79, 127, 96 | 79 |
| 43 | 10 | 0.0416 | no | none | none |
| 43 | 50 | 0.1234 | no | none | none |
| 43 | 100 | 0.4033 | no | 56, 83 | none |
| 43 | 500 | 5.3173 | YES | 111, 56, 126, 83 | 56 |
| 43 | 1000 | 5.3173 | YES | 111, 56, 126, 83 | 56 |
| 44 | 10 | 0.2104 | no | none | none |
| 44 | 50 | 0.1278 | no | none | none |
| 44 | 100 | 0.4244 | no | 97, 78, 92 | none |
| 44 | 500 | 5.6954 | YES | 97, 78, 114, 92 | 78 |
| 44 | 1000 | 5.6954 | YES | 97, 78, 114, 92 | 78 |
| 45 | 10 | 0.0651 | no | none | none |
| 45 | 50 | 0.1970 | no | none | none |
| 45 | 100 | 1.0020 | no | 80, 55, 100, 75 | none |
| 45 | 500 | 7.1008 | YES | 80, 55, 100, 75 | 55 |
| 45 | 1000 | 7.1008 | YES | 80, 55, 100, 75 | 55 |
| 46 | 10 | 0.0296 | no | none | none |
| 46 | 50 | 0.0882 | no | none | none |
| 46 | 100 | 0.2548 | no | 62 | none |
| 46 | 500 | 5.4339 | YES | 111, 62, 127, 107 | 62 |
| 46 | 1000 | 5.4339 | YES | 111, 62, 127, 107 | 62 |

### sindy_4d

| Seed | Horizon | MAE | Endpoint NaN? | First Exceed 3sigma | Divergence Step |
|------|---------|-----|---------------|--------------------|-----------------|
| 42 | 10 | 0.0165 | no | none | none |
| 42 | 50 | 0.0313 | no | none | none |
| 42 | 100 | 0.2198 | no | 97, 94 | none |
| 42 | 500 | 5.1078 | YES | 101, 97, 119, 94 | 94 |
| 42 | 1000 | 5.1078 | YES | 101, 97, 119, 94 | 94 |
| 43 | 10 | 0.0171 | no | none | none |
| 43 | 50 | 0.0473 | no | none | none |
| 43 | 100 | 0.3485 | no | 99, 76, 97, 95 | none |
| 43 | 500 | 5.4929 | YES | 99, 76, 97, 95 | 76 |
| 43 | 1000 | 5.4929 | YES | 99, 76, 97, 95 | 76 |
| 44 | 10 | 0.0327 | no | none | none |
| 44 | 50 | 0.0661 | no | none | none |
| 44 | 100 | 0.4813 | no | 91, 95, 97, 87 | none |
| 44 | 500 | 10.6453 | YES | 91, 95, 97, 87 | 87 |
| 44 | 1000 | 10.6453 | YES | 91, 95, 97, 87 | 87 |
| 45 | 10 | 0.0165 | no | none | none |
| 45 | 50 | 0.0756 | no | none | none |
| 45 | 100 | 0.4815 | no | 83, 79, 76 | none |
| 45 | 500 | 5.7843 | YES | 83, 79, 101, 76 | 76 |
| 45 | 1000 | 5.7843 | YES | 83, 79, 101, 76 | 76 |
| 46 | 10 | 0.0126 | no | none | none |
| 46 | 50 | 0.0373 | no | none | none |
| 46 | 100 | 0.2835 | no | 99, 95 | none |
| 46 | 500 | 6.4302 | YES | 99, 103, 103, 95 | 95 |
| 46 | 1000 | 6.4302 | YES | 99, 103, 103, 95 | 95 |

### gp_standard

| Seed | Horizon | MAE | Endpoint NaN? | First Exceed 3sigma | Divergence Step |
|------|---------|-----|---------------|--------------------|-----------------|
| 42 | 10 | 0.0000 | no | none | none |
| 42 | 50 | 0.0001 | no | none | none |
| 42 | 100 | 0.0009 | no | none | none |
| 42 | 500 | 11.7933 | no | 209, 212, 226, 205 | none |
| 42 | 1000 | 17.7159 | no | 209, 212, 226, 205 | none |
| 43 | 10 | 0.0000 | no | none | none |
| 43 | 50 | 0.0004 | no | none | none |
| 43 | 100 | 0.0029 | no | none | none |
| 43 | 500 | 12.1034 | no | 203, 179, 196, 175 | none |
| 43 | 1000 | 16.9886 | no | 203, 179, 196, 175 | none |
| 44 | 10 | 0.0001 | no | none | none |
| 44 | 50 | 0.0004 | no | none | none |
| 44 | 100 | 0.0027 | no | none | none |
| 44 | 500 | 12.4912 | no | 205, 179, 198, 177 | none |
| 44 | 1000 | 17.5905 | no | 205, 179, 198, 177 | none |
| 45 | 10 | 0.0000 | no | none | none |
| 45 | 50 | 0.0002 | no | none | none |
| 45 | 100 | 0.0012 | no | none | none |
| 45 | 500 | 11.6311 | no | 197, 199, 217, 192 | none |
| 45 | 1000 | 16.7384 | no | 197, 199, 217, 192 | none |
| 46 | 10 | 0.0000 | no | none | none |
| 46 | 50 | 0.0004 | no | none | none |
| 46 | 100 | 0.0029 | no | none | none |
| 46 | 500 | 12.7265 | no | 205, 180, 198, 176 | none |
| 46 | 1000 | 17.8907 | no | 205, 180, 198, 176 | none |

### gp_b4_sparse

| Seed | Horizon | MAE | Endpoint NaN? | First Exceed 3sigma | Divergence Step |
|------|---------|-----|---------------|--------------------|-----------------|
| 42 | 10 | 0.0014 | no | none | none |
| 42 | 50 | 0.0068 | no | none | none |
| 42 | 100 | 0.0492 | no | none | none |
| 42 | 500 | 7.9664 | YES | 156, 122 | 122 |
| 42 | 1000 | 7.9664 | YES | 156, 122 | 122 |
| 43 | 10 | 0.0003 | no | none | none |
| 43 | 50 | 0.0048 | no | none | none |
| 43 | 100 | 0.0360 | no | none | none |
| 43 | 500 | 7.6465 | YES | 145, 273, 230, 160 | 145 |
| 43 | 1000 | 7.6465 | YES | 145, 273, 230, 160 | 145 |
| 44 | 10 | 0.8981 | no | 4 | none |
| 44 | 50 | 3.8608 | no | 4 | none |
| 44 | 100 | 7.4407 | no | 53, 4 | none |
| 44 | 500 | 13.2414 | YES | 53, 4 | 4 |
| 44 | 1000 | 13.2414 | YES | 53, 4 | 4 |
| 45 | 10 | 0.0033 | no | none | none |
| 45 | 50 | 0.0081 | no | none | none |
| 45 | 100 | 0.0440 | no | none | none |
| 45 | 500 | 6.5996 | YES | 122, 124 | 122 |
| 45 | 1000 | 6.5996 | YES | 122, 124 | 122 |
| 46 | 10 | 0.0005 | no | none | none |
| 46 | 50 | 0.0057 | no | none | none |
| 46 | 100 | 0.0397 | no | none | none |
| 46 | 500 | 7.6329 | YES | 136, 140, 225, 132 | 132 |
| 46 | 1000 | 7.6329 | YES | 136, 140, 225, 132 | 132 |

### gp_ensemble_5_0

| Seed | Horizon | MAE | Endpoint NaN? | First Exceed 3sigma | Divergence Step |
|------|---------|-----|---------------|--------------------|-----------------|
| 42 | 10 | 0.0000 | YES | none | none |
| 42 | 50 | 0.0000 | YES | none | none |
| 42 | 100 | 0.0000 | YES | none | none |
| 42 | 500 | 0.0000 | YES | none | none |
| 42 | 1000 | 0.0000 | YES | none | none |
| 43 | 10 | 0.0000 | YES | none | none |
| 43 | 50 | 0.0000 | YES | none | none |
| 43 | 100 | 0.0000 | YES | none | none |
| 43 | 500 | 0.0000 | YES | none | none |
| 43 | 1000 | 0.0000 | YES | none | none |
| 44 | 10 | 0.0000 | YES | none | none |
| 44 | 50 | 0.0000 | YES | none | none |
| 44 | 100 | 0.0000 | YES | none | none |
| 44 | 500 | 0.0000 | YES | none | none |
| 44 | 1000 | 0.0000 | YES | none | none |
| 45 | 10 | 0.0000 | YES | none | none |
| 45 | 50 | 0.0000 | YES | none | none |
| 45 | 100 | 0.0000 | YES | none | none |
| 45 | 500 | 0.0000 | YES | none | none |
| 45 | 1000 | 0.0000 | YES | none | none |
| 46 | 10 | 0.0000 | YES | none | none |
| 46 | 50 | 0.0000 | YES | none | none |
| 46 | 100 | 0.0000 | YES | none | none |
| 46 | 500 | 0.0000 | YES | none | none |
| 46 | 1000 | 0.0000 | YES | none | none |

### gp_ensemble_5_3

| Seed | Horizon | MAE | Endpoint NaN? | First Exceed 3sigma | Divergence Step |
|------|---------|-----|---------------|--------------------|-----------------|
| 42 | 10 | 0.0135 | no | none | none |
| 42 | 50 | 0.2242 | no | none | none |
| 42 | 100 | 0.3136 | no | none | none |
| 42 | 500 | 0.3707 | no | none | none |
| 42 | 1000 | 0.3858 | no | none | none |
| 43 | 10 | 0.0209 | no | none | none |
| 43 | 50 | 0.1431 | no | none | none |
| 43 | 100 | 0.2085 | no | none | none |
| 43 | 500 | 0.2608 | no | none | none |
| 43 | 1000 | 0.2676 | no | none | none |
| 44 | 10 | 0.2552 | no | none | none |
| 44 | 50 | 0.3675 | no | none | none |
| 44 | 100 | 0.3566 | no | none | none |
| 44 | 500 | 0.3808 | no | none | none |
| 44 | 1000 | 0.3830 | no | none | none |
| 45 | 10 | 0.0797 | no | none | none |
| 45 | 50 | 0.3210 | no | none | none |
| 45 | 100 | 0.2707 | no | none | none |
| 45 | 500 | 0.2287 | no | none | none |
| 45 | 1000 | 0.2232 | no | none | none |
| 46 | 10 | 0.0388 | no | none | none |
| 46 | 50 | 0.2362 | no | none | none |
| 46 | 100 | 0.2801 | no | none | none |
| 46 | 500 | 0.3017 | no | none | none |
| 46 | 1000 | 0.3042 | no | none | none |

### gp_ensemble_7_5

| Seed | Horizon | MAE | Endpoint NaN? | First Exceed 3sigma | Divergence Step |
|------|---------|-----|---------------|--------------------|-----------------|
| 42 | 10 | 0.0655 | no | none | none |
| 42 | 50 | 0.1232 | no | none | none |
| 42 | 100 | 0.2451 | no | none | none |
| 42 | 500 | 0.3500 | no | none | none |
| 42 | 1000 | 0.3641 | no | none | none |
| 43 | 10 | 0.0695 | no | none | none |
| 43 | 50 | 0.1288 | no | none | none |
| 43 | 100 | 0.0903 | no | none | none |
| 43 | 500 | 0.0479 | no | none | none |
| 43 | 1000 | 0.0420 | no | none | none |
| 44 | 10 | 0.1983 | no | none | none |
| 44 | 50 | 0.4077 | no | none | none |
| 44 | 100 | 0.4045 | no | none | none |
| 44 | 500 | 0.3753 | no | none | none |
| 44 | 1000 | 0.3759 | no | none | none |
| 45 | 10 | 0.0880 | no | none | none |
| 45 | 50 | 0.2613 | no | none | none |
| 45 | 100 | 0.2319 | no | none | none |
| 45 | 500 | 0.2163 | no | none | none |
| 45 | 1000 | 0.2145 | no | none | none |
| 46 | 10 | 0.0350 | no | none | none |
| 46 | 50 | 0.1038 | no | none | none |
| 46 | 100 | 0.1401 | no | none | none |
| 46 | 500 | 0.2700 | no | none | none |
| 46 | 1000 | 0.2897 | no | none | none |

---

## 3. Divergence Summary

| Model | Diverges? | Approx Divergence Range | Survival Steps Correct? |
|-------|-----------|------------------------|------------------------|
| linearized_model | YES | steps 55-79 | NO (always reports requested horizon) |
| sindy_4d | YES | steps 76-95 | NO (always reports requested horizon) |
| gp_standard | NO | N/A | NO (always reports requested horizon) |
| gp_b4_sparse | YES | steps 4-145 | NO (always reports requested horizon) |
| gp_ensemble_5_0 | YES | unknown (NaN endpoint) | NO (always reports requested horizon) |
| gp_ensemble_5_3 | NO | N/A | NO (always reports requested horizon) |
| gp_ensemble_7_5 | NO | N/A | NO (always reports requested horizon) |

---

## 4. Padding Method

**Method**: NaN fill after divergence detection.

```python
# eval_modes_v2.py, run_mode_b_fixed, lines 78-81:
if np.any(np.isnan(s_model)) or np.any(np.abs(s_model) > 100):
    remaining = n_valid - i - 1
    states_model.extend([np.full(4, np.nan)] * remaining)
    break
```

**Consequence**: `compute_all_metrics` filters NaN steps via `valid = ~np.any(np.isnan(...), axis=1)`.
MAE is computed only on valid (pre-divergence) steps. This means:
- 500-step and 1000-step MAE are identical if divergence occurs before step 500
- The MAE value represents average error over surviving steps only
- No 'hallucinated' metrics from padded NaN values

**Verdict**: NaN padding is correct. The bug is in survival_steps reporting, not in metric computation.

---

## 5. Whether Padded

| Model | Padded? | Padding Method |
|-------|---------|----------------|
| linearized_model | YES | NaN fill (np.full(4, np.nan)) |
| sindy_4d | YES | NaN fill (np.full(4, np.nan)) |
| gp_standard | NO | N/A |
| gp_b4_sparse | YES | NaN fill (np.full(4, np.nan)) |
| gp_ensemble_5_0 | YES | NaN fill (np.full(4, np.nan)) |
| gp_ensemble_5_3 | NO | N/A |
| gp_ensemble_7_5 | NO | N/A |