# Sub-Agent Cross-Review Summary

## Agents Deployed

| Agent | Role | Key Finding |
|-------|------|-------------|
| A: Code Explorer | Trace V9 vs V12 code paths | V9 multi-step missing delta_std factor |
| B: Root Cause Analyst | Identify primary/secondary causes | Normalization mismatch is PRIMARY |
| C: Solution Reviewer | Evaluate proposed fixes | Survival-aware loss needs H>20 curriculum |
| D: Counter-Argument Expert | Challenge conclusions | V9's "good" results may be trivial prediction |

## Cross-Validation of Findings

### Finding 1: V9 Multi-Step Bug
- **Agent A**: Confirmed `s_cur = s_cur + dsdt * self._dt` missing delta_std
- **Agent B**: Confirmed, but noted V9's predict() compensates: `s + dsdt * delta_std * dt`
- **Verdict**: Bug exists in training but not inference. Training multi-step loss is wrong, but single-step loss and predict() are correct.

### Finding 2: Shuffle Breaking Temporal Ordering
- **Agent A**: Confirmed `shuffle=True` on TensorDataset
- **Agent B**: Confirmed as CRITICAL — multi-step loss assumes sb[i+1] is successor of sb[i]
- **Verdict**: Most impactful bug. Fixing it dramatically improves NMAE but causes survival collapse.

### Finding 3: Contractivity Tradeoff
- **Agent C**: Recommended contractivity as primary solution
- **Agent D**: Challenged — "contractivity forces conservative dynamics, explaining NMAE regression"
- **Experiment confirmed**: lambda=0.05 worsens H=50 NMAE by 18%. Even lambda=0.01 doesn't help.

### Finding 4: V9 Baseline's Nature
- **Agent D**: "V9's 0.998 NMAE with 100% survival may be trivial constant prediction"
- **Evidence**: Per-state NMAE at H=500 shows delta_dot=1.706 but theta=0.315 — inconsistent errors suggest partial constant prediction
- **Verdict**: Plausible but not conclusively proven. V9 may have learned a conservative but non-trivial dynamics model.

## Disagreements

1. **Agent C vs D on contractivity**: C sees it as solution, D sees it as source of NMAE regression. Experiment supports D.
2. **Agent B vs C on normalization importance**: B says normalization is primary cause, C focuses on training procedure. Both are correct — normalization affects training, not inference.

## Synthesis

The sub-agents converge on one key insight: **the shuffle bug fix is necessary but not sufficient**. It corrects the training signal but exposes a deeper challenge — accurate multi-step prediction naturally leads to trajectory divergence at long horizons without explicit stability constraints.
