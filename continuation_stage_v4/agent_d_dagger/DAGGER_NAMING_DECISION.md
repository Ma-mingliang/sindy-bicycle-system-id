# DAgger Naming Decision

**Date:** 2026-06-25
**Decision Authority:** DAgger Semantics & Residual Structure Sub-Agent

---

## 1. Current Code Naming vs Reality

### What the code claims

The class is named `GPEEnsemble` with method `_collect_dagger_data()`.
The DAgger Semantics Audit (DAGGER_SEMANTICS_AUDIT.md) concluded:
> "The current DAgger implementation computes local dynamics residuals... Verdict: Valid DAgger implementation. No renaming needed."

### What the code actually does

**The audit is WRONG.** Detailed code trace (CURRENT_DAGGER_EXACT_FORMULA.md) proves:

**Round 0** (lines 288-292):
```
r_k = (F_real(s_expert, u) - GP(s_expert, u)) / std
```
This IS Version L (local dynamics residual at expert states). **CONFIRMED.**

**Round 1+** (lines 352-379):
```
r_k = (s_real[k+1] - GP(GP(s_model[k], u), u)) / std
```
This is a TWO-STEP GP trajectory comparison. **CONFIRMED.** It is:
- NOT Version L (real and GP at different states)
- NOT standard Version T (GP called twice, not once)
- A corrupted variant that mixes trajectory comparison with multi-step GP prediction

### The inconsistency

Round 0 trains the NN on "GP error at expert states" (Version L).
Round 1+ trains the NN on "real next state vs two-step GP prediction" (corrupted trajectory comparison).

These are fundamentally different training objectives.

---

## 2. Evaluation of the Four Naming Options

### Option A: "Standard DAgger"

**Verdict: REJECTED**

Standard DAgger (Ross et al., 2011) collects (expert_state, expert_action) pairs during model rollouts and trains the model to mimic the expert's action. The current implementation does NOT do this:
- It trains an NN residual, not a full policy
- The residual is not the expert action, but a dynamics correction
- Round 0 and Round 1+ use different residual definitions

The DAgger spirit (training on model-visited states with expert labels) is partially present, but the specific formulation is non-standard.

### Option B: "Trajectory-Synchronization Residual Aggregation"

**Verdict: REJECTED for the original code**

This name would be accurate for Version T (pure `s_real[k+1] - GP(s_model[k], u)`), but the original code uses a two-step GP variant that does not fit this description cleanly.

The original code is closer to a trajectory comparison, but the two-step GP makes it a "multi-step trajectory synchronization" which is non-standard.

### Option C: "Local Dynamics Residual Aggregation"

**Verdict: REJECTED for the original code**

This name fits Round 0 perfectly, but Round 1+ is NOT a local dynamics residual. The real dynamics are evaluated at s_real[k] while the GP is evaluated at GP(s_base[k], tau) -- different states.

### Option D: "Residual Dynamics Ensemble with Inconsistent Training Objectives"

**Verdict: ACCEPTED (descriptive)**

This is the most accurate description. The implementation:
1. Uses a GP baseline for coarse dynamics
2. Trains an NN ensemble to predict residuals
3. Uses DAgger-style data collection on model trajectories
4. Has INCONSISTENT residual definitions across rounds

---

## 3. Recommended Naming for the Two Clean Versions

### Version L: "Local Dynamics Residual DAgger"

**Justification:**
- Both real dynamics and GP evaluated at the same state (model state)
- Residual is the pure GP prediction error
- Clean, interpretable, well-defined
- Standard DAgger spirit: trains on model-visited states with local error labels

**When to use:**
- Single-step prediction accuracy is the priority
- Feedback controller will compensate for errors (Mode D)
- System stays within training distribution

### Version T: "Trajectory-Synchronization Residual DAgger"

**Justification:**
- Real dynamics at s_real[k], GP at s_model[k]
- Residual measures trajectory gap (how far real next state is from GP prediction)
- Standard trajectory-level DAgger residual
- Trains on model-visited states with trajectory-level labels

**When to use:**
- Multi-step open-loop prediction (Mode B)
- Varying initial conditions
- Long-horizon planning
- Steering dynamics (delta, delta_dot) are important

---

## 4. Final Naming Recommendation

| Version | Recommended Name | Abbreviation |
|---------|-----------------|--------------|
| Current original | "Inconsistent Residual Ensemble" | IRE |
| Version L | "Local Dynamics Residual DAgger" | L-Dagger |
| Version T | "Trajectory-Synchronization Residual DAgger" | T-Dagger |

### For the project

If only one name is needed:
- **Use "Residual Dynamics Ensemble" (RDE)** as a neutral umbrella term
- Specify the residual type when comparing: "RDE-L" or "RDE-T"
- The term "DAgger" should be qualified: "DAgger-style data collection" rather than "DAgger"

### Why "DAgger" should be qualified

Standard DAgger (Ross et al., 2011) specifically:
1. Collects (s_model, a_expert) pairs
2. Trains a policy to output the expert action
3. Uses the trained policy for the next rollout

The current implementation:
1. Collects (s_model, residual_label) pairs
2. Trains an NN to predict a residual correction
3. Uses GP + scaled_NN for the next rollout

This is a "DAgger-style" data collection strategy, but not standard DAgger. The name "DAgger" is acceptable in the broader sense (adaptive dataset aggregation) but should not imply exact equivalence to the original algorithm.

---

## 5. Impact Assessment

### Does the naming affect scientific validity?

**No.** Both Version L and Version T are legitimate residual learning approaches. The key question is which residual formulation works better for the intended application, not whether it exactly matches the original DAgger paper.

### Does the original code's naming cause confusion?

**Yes.** The audit's conclusion ("Valid DAgger implementation. No renaming needed.") is incorrect and could mislead readers into thinking the residual is a clean local dynamics error. The two-step GP evaluation in Round 1+ makes the residual harder to interpret.

### Should the original code be renamed?

**Yes.** Recommended rename: `GPEEnsemble` -> `ResidualGPEnsemble` with a docstring noting the inconsistent residual definitions across rounds.

---

## Decision Summary

| Question | Answer |
|----------|--------|
| Is the original code "Standard DAgger"? | **NO** |
| Is the original code "Version L"? | **Only in Round 0** |
| Is the original code "Version T"? | **No** (corrupted two-step variant) |
| Should the original be renamed? | **Yes**: "Inconsistent Residual Ensemble" or "Residual GP Ensemble" |
| Which clean version is better? | **Depends on use case** (see evaluation results) |
| Is "DAgger" an appropriate term? | **Qualified yes**: "DAgger-style" or "DAgger-inspired" |
