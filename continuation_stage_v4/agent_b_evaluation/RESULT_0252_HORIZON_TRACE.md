# Result 0.252 Horizon Trace

> Generated: 2026-06-25
> Agent: Agent B (Evaluation Protocol Auditor)

---

## 1. The Contradiction

Two contradictory claims exist in the project reports:

**Claim A** (FINAL_HANDOVER_V3.md, line 78):
> "0.252: Mode B full-trajectory MAE at step 100 (seeded)"

**Claim B** (FINAL_HANDOVER_V3.md, line 134, table):
> gp_ensemble_7_5 row, 500-step column: **0.252**

**Claim C** (HISTORICAL_0064_0066_CODE_TRACE.md, line 13):
> "0.252 (evaluation v2 Mode B 100 steps)"

---

## 2. Actual Data from full_evaluation_v2.json

gp_ensemble_7_5, Mode B, 5 seeds (42-46):

| Horizon | MAE per seed | Mean MAE |
|---------|-------------|----------|
| 10 | 0.0913, 0.0660, 0.1447, 0.0820, 0.0636 | **0.0913** |
| 50 | 0.2049, 0.0660, 0.4045, 0.1687, 0.1807 | **0.2049** |
| 100 | 0.2224, 0.0903, 0.4045, 0.2319, 0.1401 | **0.2224** |
| 500 | 0.3500, 0.0479, 0.3753, 0.2163, 0.2700 | **0.2519** |
| 1000 | 0.3641, 0.0420, 0.3759, 0.2145, 0.2897 | **0.2573** |

---

## 3. Trace Resolution

**0.252 corresponds to:**
- **Value**: Mean MAE across 5 seeds
- **Model**: gp_ensemble_7_5
- **Mode**: B (fixed-action open-loop)
- **Horizon**: 500 steps (NOT 100 steps)
- **Metric**: Full-trajectory MAE (all 4 states)
- **Source file**: `continuation_stage/results/full_evaluation_v2.json`
- **Seed range**: 42-46

**The correct value for 100 steps is 0.2224, NOT 0.252.**

---

## 4. Root Cause of Contradiction

The FINAL_HANDOVER_V3.md table (line 134) is correct:
```
| gp_ensemble_7_5 | 0.091 | 0.205 | 0.222 | **0.252** | **0.257** | 未发散 |
                  | 10步  | 50步  | 100步 |   500步   |  1000步   |        |
```

The text description (line 78) is incorrect:
> "0.252: Mode B full-trajectory MAE at step 100 (seeded)"

Should be:
> "0.252: Mode B full-trajectory MAE at step 500 (seeded)"

Similarly, HISTORICAL_0064_0066_CODE_TRACE.md (line 13) says "100 steps" but should say "500 steps".

---

## 5. Corrected Values

| Source | Claimed | Actual | Horizon | Correct? |
|--------|---------|--------|---------|----------|
| FINAL_HANDOVER_V3.md table | 0.252 | 0.2519 | 500 | YES |
| FINAL_HANDOVER_V3.md text | 0.252 at step 100 | 0.2224 at step 100 | 100 | NO - wrong horizon |
| HISTORICAL_0064_0066_CODE_TRACE.md | 0.252 at step 100 | 0.2224 at step 100 | 100 | NO - wrong horizon |
| HISTORICAL_RESULT_FACTS.json | 0.252 at 100 steps | 0.2519 at 500 steps | 500 | NO - wrong horizon |

---

## 6. Verdict

**0.252 is the Mode B 500-step MAE for gp_ensemble_7_5, averaged over 5 seeds (42-46).**

The "100 steps" label in two documents is a transcription error. The table data is correct.

---
