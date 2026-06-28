# Mode B Implementation Proof

**Date:** 2026-06-25
**Project:** D:\系统辨识作业\sindy_bicycle
**Mode B Implementation:** continuation_stage/evaluate/eval_modes_v2.py
**Evaluation Script:** continuation_stage/evaluate/run_full_evaluation_v2.py

---

## Executive Summary

Mode B implementation is **CORRECT**. All 4 verification tests pass.

1. **Fixed Action Sequence Hash Test**: PASS
2. **Mode B vs Mode C Action Comparison**: PASS
3. **Determinism Test**: PASS
4. **Static Analysis**: PASS

---

## Test 1: Fixed Action Sequence Hash Test

**Status:** PASS

For each seed (42-46), compute SHA256 of the action sequence from generate_reference_trajectory().
Run run_mode_b_fixed() with each model and verify hash matches.

| Seed | SHA256 (prefix) | n_actions | Action Range |
|------|----------------|-----------|--------------|
| 42 | d3d0acb001b9e0cf... | 1000 | [-9.4142, 2.8301] |
| 43 | b8abce76e62004d1... | 1000 | [-4.4614, 2.0685] |
| 44 | 06dac816690b8533... | 1000 | [-4.1146, 17.2980] |
| 45 | 4638e4b1054f704c... | 1000 | [-2.8372, 2.5781] |
| 46 | c3c3e99b5c3f3335... | 1000 | [-1.7149, 3.2165] |

Cross-model consistency (seed 42, horizon=100):

| Model | Action Hash (prefix) |
|-------|---------------------|
| real_dynamics | d58e50ea24d7e0ba... |
| linearized_model | d58e50ea24d7e0ba... |
| sindy_4d | d58e50ea24d7e0ba... |

All models produce **identical** action sequences. The action sequence is frozen at reference generation time.

---

## Test 2: Mode B vs Mode C Action Comparison

**Status:** PASS

Seed 42, horizon 50, linearized_model. All 50 steps identical (diff = 0.00e+00).

Both modes compute actions via tau_func(i, s_real) where s_real follows the same real trajectory.
Mode B uses a pre-frozen sequence; Mode C recomputes from the same real state. Actions are identical.

Mode B state MAE: 0.063096, Mode C state MAE: 0.063096

---

## Test 3: Determinism Test

**Status:** PASS

| Model | States Match | Actions Match | Survival Match |
|-------|-------------|--------------|----------------|
| sindy_4d | True (atol=1e-15) | True (atol=1e-15) | True (100) |
| real_dynamics | True | N/A | N/A |
| linearized_model | True | N/A | N/A |

---

## Test 4: Static Analysis

**Status:** PASS

AST analysis of eval_modes_v2.py confirms:
- run_mode_b_fixed defined at line 44
- NO tau_func references inside run_mode_b_fixed
- tau_func only used in: generate_reference_trajectory (line 29), run_mode_a (line 104), run_mode_c (line 137)
- run_mode_b_fixed reads actions from reference dict only (line 53)

---

## Compliance with Mode B Requirements

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Pre-generated fixed action sequence | PASS | line 53: actions = reference["actions"] |
| Actions frozen before evaluation | PASS | eval script line 177: generate_reference_trajectory() |
| No recomputation from real state | PASS | No tau_func in run_mode_b_fixed |
| No recomputation from model state | PASS | No controller logic |
| No controller updates | PASS | No tau_func calls |
| No modification of action sequence | PASS | Actions read-only from reference dict |
| Deterministic | PASS | Test 3 confirms identical results |

---

## Source Code References

| File | Line(s) | Function |
|------|---------|----------|
| eval_modes_v2.py | 16-41 | generate_reference_trajectory |
| eval_modes_v2.py | 44-92 | run_mode_b_fixed |
| run_full_evaluation_v2.py | 177 | Reference generated once per seed |
| run_full_evaluation_v2.py | 199 | run_mode_b_fixed(model, s0, ref, horizon, mc.real_step) |

---

**Verdict: Mode B implementation is CORRECT and COMPLIANT.**