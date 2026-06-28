# Mode B Call Graph and Timing Protocol

> Generated: 2026-06-25
> Agent: Agent B (Evaluation Protocol Auditor)
> Status: CONFIRMED (code analysis) + INFERRED (timing order)

---

## 1. Mode B Call Sequence (Fixed-Action Open-Loop)

```
run_full_evaluation_v2.py
  |
  |-- [Phase 1: Reference Generation] (ONCE per seed)
  |     |
  |     +-> generate_reference_trajectory(s0, tau_func, max_horizon, real_step)
  |           |
  |           |-- FOR i in range(n_steps):
  |           |     tau = tau_func(i, s_real)        <-- computes action from REAL state
  |           |     actions.append(tau)               <-- action frozen into array
  |           |     s_real = real_dynamics(s_real, tau) <-- real system advances
  |           |
  |           +-> RETURN { actions: np.array([...]), states_ref: np.array([...]) }
  |                    ^
  |                    | actions array is FROZEN at this point
  |
  |-- [Phase 2: Model Evaluation] (per model, per horizon)
        |
        +-> run_mode_b_fixed(model, s0, ref, horizon, real_step)
              |
              |-- actions = ref['actions']             <-- READ from frozen array
              |-- FOR i in range(n_valid):
              |     tau = actions[i]                   <-- READ only, no computation
              |     s_next, unc = model.predict(s_model, tau)  <-- model predicts
              |     s_model = s_next                   <-- model state updates
              |     [if diverged: pad NaN, break]
              |
              +-> RETURN { states_model, states_real, survival_steps }
```

### Key Properties

| Property | Status | Evidence |
|----------|--------|----------|
| Action array frozen before evaluation | CONFIRMED | `ref['actions']` read at line 53 of eval_modes_v2.py |
| No tau_func call in run_mode_b_fixed | CONFIRMED | Static analysis: zero tau_func references |
| No controller in run_mode_b_fixed | CONFIRMED | No LQR, no feedback logic |
| Actions read-only during evaluation | CONFIRMED | `tau = actions[i]` is array indexing only |
| Same actions for all models | CONFIRMED | Same `ref` dict passed to all models |

---

## 2. Mode C Call Sequence (Hybrid - tau from real state)

```
run_full_evaluation_v2.py
  |
  +-> run_mode_c(model, s0, tau_func, horizon, real_step)
        |
        |-- FOR i in range(n_steps):
        |     tau = tau_func(i, s_real)              <-- COMPUTES action from real state EACH STEP
        |     actions.append(tau)                     <-- records action
        |     s_real = real_step(s_real, tau)         <-- real system advances
        |     s_next, unc = model.predict(s_model, tau) <-- model predicts
        |     s_model = s_next                        <-- model state updates
        |
        +-> RETURN { states_model, states_real, actions }
```

### Key Difference from Mode B

| Aspect | Mode B | Mode C |
|--------|--------|--------|
| Action source | Pre-frozen array from reference | Computed online from real state |
| tau_func called during eval? | NO | YES, every step |
| Action computation timing | Before evaluation | During evaluation |
| Action depends on real state during eval? | NO (frozen) | YES (s_real changes each step) |

---

## 3. Timing Diagram

```
TIME -->

Mode B:
  t0: generate_reference_trajectory() --> actions FROZEN
  t1: run_mode_b_fixed() reads actions[0], actions[1], ..., actions[H-1]
      (no tau_func calls, no action recomputation)

Mode C:
  t0: (no pre-generation)
  t1: run_mode_c() step 0: tau = tau_func(0, s_real) --> COMPUTE
  t2: run_mode_c() step 1: tau = tau_func(1, s_real) --> COMPUTE
  ...
  tH: run_mode_c() step H-1: tau = tau_func(H-1, s_real) --> COMPUTE
```

---

## 4. Why Mode B and Mode C Produce Identical Actions

**CONFIRMED**: When tau_func is a deterministic function of (step_index, state) and the real
trajectory is identical, both modes produce the same actions.

**Mode B**: `tau = actions[i]` where `actions[i] = tau_func(i, s_ref[i])` (pre-computed)
**Mode C**: `tau = tau_func(i, s_real[i])` where `s_real[i] = s_ref[i]` (same trajectory)

Since `s_ref[i] == s_real[i]` (same initial state, same real dynamics, same actions),
the actions are mathematically identical.

**However**, the IMPLEMENTATION is different:
- Mode B: array read (O(1) lookup)
- Mode C: function call + real state tracking (O(1) computation)

This is a protocol difference, not a numerical difference.

---

## 5. Compliance Verdict

| Requirement | Mode B | Mode C |
|-------------|--------|--------|
| Fixed action sequence | PASS | N/A (actions recomputed) |
| No online controller | PASS | N/A (tau_func is not a controller) |
| No tau_func in eval loop | PASS | FAIL (by design - this is Mode C) |
| Same initial state | PASS | PASS |
| Same real dynamics | PASS | PASS |

**Mode B is a correctly implemented fixed-action open-loop evaluation.**
