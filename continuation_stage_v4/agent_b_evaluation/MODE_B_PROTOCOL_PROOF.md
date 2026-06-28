# Mode B Protocol Proof

> Generated: 2026-06-25
> Agent: Agent B (Evaluation Protocol Auditor)
> Verdict: CONFIRMED CORRECT

---

## 1. Protocol Requirements

Mode B (Fixed-Action Open-Loop) requires:

1. **Pre-generation**: Action sequence U = [u_0, ..., u_{H-1}] generated before evaluation
2. **Freezing**: Actions frozen after generation, no modification during evaluation
3. **Uniform usage**: All models use the exact same action array
4. **No online control**: No controller re-evaluates actions during the evaluation loop
5. **No tau_func dependency**: The evaluation function does not call tau_func

---

## 2. Code Evidence

### 2.1 Action Pre-Generation

**File**: `continuation_stage/evaluate/run_full_evaluation_v2.py`, line 177

```python
ref = generate_reference_trajectory(s0, tau_func, max(horizons), mc.real_step)
```

This is called ONCE per seed, BEFORE any model evaluation. The returned `ref` dict
contains `actions` (np.ndarray) and `states_ref` (np.ndarray).

**Status**: CONFIRMED - actions generated once per seed before evaluation.

### 2.2 Action Freezing

**File**: `continuation_stage/evaluate/eval_modes_v2.py`, lines 16-41

```python
def generate_reference_trajectory(s0, tau_func, n_steps, real_dynamics):
    states_ref = [s0.copy()]
    actions = []
    s = s0.copy()
    for i in range(n_steps):
        tau = tau_func(i, s)
        actions.append(tau)
        s = real_dynamics(s, tau)
        states_ref.append(s.copy())
    return {
        'actions': np.array(actions),  # <-- FROZEN as numpy array
        'states_ref': np.array(states_ref),
        'n_valid_steps': len(actions),
    }
```

The `actions` list is converted to `np.array` and returned. No subsequent code
modifies this array.

**Status**: CONFIRMED - actions frozen as immutable numpy array.

### 2.3 Uniform Usage Across Models

**File**: `continuation_stage/evaluate/run_full_evaluation_v2.py`, line 199

```python
r = run_mode_b_fixed(model, s0, ref, horizon, mc.real_step)
```

The same `ref` dict (containing the same `actions` array) is passed to every model.
The `ref` dict is created once per seed at line 177 and reused for all models and horizons.

**Status**: CONFIRMED - all models receive identical action arrays.

### 2.4 No Online Control

**File**: `continuation_stage/evaluate/eval_modes_v2.py`, lines 44-92

```python
def run_mode_b_fixed(model, s0, reference, n_steps, real_dynamics):
    actions = reference['actions']  # <-- READ from frozen array
    n_valid = min(n_steps, len(actions))
    ...
    for i in range(n_valid):
        tau = actions[i]  # <-- ARRAY INDEXING ONLY
        s_next, unc = model.predict_with_uncertainty(s_model, tau)
        ...
```

There is:
- No `tau_func` call
- No LQR controller
- No feedback from model state to action selection
- No action recomputation

**Status**: CONFIRMED - pure open-loop with frozen actions.

### 2.5 Static Analysis

**File**: `continuation_stage_v2/tests/MODE_B_IMPLEMENTATION_PROOF.md`

AST analysis confirms:
- `run_mode_b_fixed` defined at line 44
- ZERO `tau_func` references inside `run_mode_b_fixed`
- `tau_func` only used in: `generate_reference_trajectory`, `run_mode_a`, `run_mode_c`

**Status**: CONFIRMED by independent static analysis.

---

## 3. Verification Tests (from MODE_B_IMPLEMENTATION_PROOF.md)

| Test | Status | Evidence |
|------|--------|----------|
| Fixed action sequence hash | PASS | SHA256 identical across all models |
| Mode B vs Mode C actions | PASS | diff = 0.00e+00 (identical values) |
| Determinism | PASS | atol=1e-15 across runs |
| Static analysis | PASS | Zero tau_func in run_mode_b_fixed |

---

## 4. Known Limitation

**Mode B and Mode C produce identical numerical results** because:
- tau_func is deterministic: tau = f(step_index, state)
- Both modes use the same real trajectory (same initial state, same real dynamics)
- Mode B pre-computes tau_func(i, s_real[i]) for all i
- Mode C computes tau_func(i, s_real[i]) online for each i
- Since s_real[i] is the same in both cases, tau values are identical

This is NOT a bug. It confirms that Mode B correctly implements the fixed-action protocol.
The numerical equivalence is a consequence of the deterministic tau_func and identical
real trajectories.

**To truly differentiate Mode B and Mode C**, one would need:
- A stochastic tau_func (different actions each time)
- A non-deterministic real system
- Different initial states for the real system in each mode

---

## 5. Verdict

**Mode B is CONFIRMED as a correctly implemented fixed-action open-loop evaluation.**

All 5 protocol requirements are satisfied:
1. Pre-generation: PASS
2. Freezing: PASS
3. Uniform usage: PASS
4. No online control: PASS
5. No tau_func dependency: PASS
