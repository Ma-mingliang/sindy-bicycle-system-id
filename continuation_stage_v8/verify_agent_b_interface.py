"""Agent B: Verify canonical_7d package interface.

Tests all imports, data loading, model training, prediction, metrics,
and residual model uncertainty interfaces.
"""
import sys
import json
import time
import traceback
import numpy as np
from pathlib import Path

sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')

results = {}
PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


def record(test_name, status, detail="", elapsed_s=0.0):
    results[test_name] = {
        "status": status,
        "detail": detail,
        "elapsed_s": round(elapsed_s, 3),
    }
    tag = f"[{status}]"
    print(f"  {tag} {test_name}: {detail}")


# ============================================================
# TEST 1: All imports from canonical_7d
# ============================================================
print("\n=== TEST 1: All imports ===")
t0 = time.time()
try:
    from canonical_7d import (
        STATE_NAMES_7D, STATE_DIM, ACTION_DIM, IDX_7D_FROM_8D,
        DataConfig, TrainingConfig, RDEConfig, EvalConfig,
        CalibConfig, PlanConfig, StatisticalConfig,
        load_7d_data, get_test_segments,
        BaseModel, GPModel, SINDyModel,
        E1OfflineNN, RDELocal, RDETrajectory, RDEHybrid,
        EvalResult, run_mode_a, run_mode_b, run_mode_c,
        compute_metrics, compute_cost,
        train_all_models, evaluate_model_on_segments, aggregate_results,
    )
    # Verify constants
    assert STATE_DIM == 7, f"STATE_DIM={STATE_DIM}, expected 7"
    assert ACTION_DIM == 1, f"ACTION_DIM={ACTION_DIM}, expected 1"
    assert len(STATE_NAMES_7D) == 7, f"STATE_NAMES_7D has {len(STATE_NAMES_7D)} entries"
    assert IDX_7D_FROM_8D == [0, 1, 2, 3, 4, 6, 7], f"IDX wrong: {IDX_7D_FROM_8D}"
    record("test1_all_imports", PASS,
           f"STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}, "
           f"names={STATE_NAMES_7D}, idx={IDX_7D_FROM_8D}",
           time.time() - t0)
except Exception as e:
    record("test1_all_imports", FAIL, f"{e}\n{traceback.format_exc()}", time.time() - t0)

# ============================================================
# TEST 2: load_7d_data() returns correct shapes
# ============================================================
print("\n=== TEST 2: load_7d_data ===")
t0 = time.time()
try:
    data = load_7d_data()
    required_keys = [
        'train_states', 'train_actions', 'train_deltas',
        'test_states', 'test_actions', 'test_deltas',
        'state_std', 'action_std', 'delta_std',
        'n_episodes', 'episode_starts', 'episode_ends',
        'train_mask', 'test_mask',
        'all_states', 'all_actions', 'all_deltas', 'all_done',
    ]
    missing = [k for k in required_keys if k not in data]
    assert not missing, f"Missing keys: {missing}"

    # Shape checks
    N_train = data['train_states'].shape[0]
    N_test = data['test_states'].shape[0]
    N_all = data['all_states'].shape[0]

    assert data['train_states'].shape[1] == 7, f"train_states cols={data['train_states'].shape[1]}"
    assert data['test_states'].shape[1] == 7, f"test_states cols={data['test_states'].shape[1]}"
    assert data['train_deltas'].shape[1] == 7, f"train_deltas cols={data['train_deltas'].shape[1]}"
    assert data['test_deltas'].shape[1] == 7, f"test_deltas cols={data['test_deltas'].shape[1]}"
    assert data['train_actions'].ndim == 1, f"train_actions ndim={data['train_actions'].ndim}"
    assert data['test_actions'].ndim == 1, f"test_actions ndim={data['test_actions'].ndim}"
    assert data['state_std'].shape == (7,), f"state_std shape={data['state_std'].shape}"
    assert data['delta_std'].shape == (7,), f"delta_std shape={data['delta_std'].shape}"
    assert isinstance(data['action_std'], float), f"action_std type={type(data['action_std'])}"
    assert N_train + N_test <= N_all, f"train+test={N_train+N_test} > all={N_all}"
    assert data['n_episodes'] > 0, f"n_episodes={data['n_episodes']}"
    assert len(data['episode_starts']) == len(data['episode_ends']), "episodes mismatch"

    detail = (f"train=({N_train},7), test=({N_test},7), all=({N_all},7), "
              f"episodes={data['n_episodes']}, state_std_shape={data['state_std'].shape}")
    record("test2_load_7d_data", PASS, detail, time.time() - t0)
except Exception as e:
    record("test2_load_7d_data", FAIL, f"{e}\n{traceback.format_exc()}", time.time() - t0)

# ============================================================
# TEST 3: get_test_segments() returns correct shapes
# ============================================================
print("\n=== TEST 3: get_test_segments ===")
t0 = time.time()
try:
    segments = get_test_segments(data, n_segments=3, segment_length=50, seed=42)
    assert len(segments) == 3, f"Expected 3 segments, got {len(segments)}"
    for i, seg in enumerate(segments):
        assert 'states' in seg, f"Segment {i} missing 'states'"
        assert 'actions' in seg, f"Segment {i} missing 'actions'"
        assert 'start_idx' in seg, f"Segment {i} missing 'start_idx'"
        assert seg['states'].shape == (51, 7), f"Seg {i} states shape={seg['states'].shape}"
        assert seg['actions'].shape == (50,), f"Seg {i} actions shape={seg['actions'].shape}"
        assert isinstance(seg['start_idx'], (int, np.integer)), f"start_idx type={type(seg['start_idx'])}"

    detail = (f"3 segments, states=(51,7), actions=(50,), "
              f"start_idxs={[int(s['start_idx']) for s in segments]}")
    record("test3_get_test_segments", PASS, detail, time.time() - t0)
except Exception as e:
    record("test3_get_test_segments", FAIL, f"{e}\n{traceback.format_exc()}", time.time() - t0)

# ============================================================
# TEST 4: GPModel train + predict on small subset (100 samples)
# ============================================================
print("\n=== TEST 4: GPModel train+predict (100 samples) ===")
t0 = time.time()
try:
    gp = GPModel(max_samples=100, n_restarts=1)
    # Use a small subset
    gp.train(
        data['train_states'][:200],
        data['train_actions'][:200],
        data['train_deltas'][:200],
        data['state_std'], data['action_std'], data['delta_std']
    )
    assert gp._gps is not None, "GP list not created"
    assert len(gp._gps) == 7, f"Expected 7 GPs, got {len(gp._gps)}"

    s_test = data['test_states'][0]
    a_test = float(data['test_actions'][0])
    pred = gp.predict(s_test, a_test)
    assert pred.shape == (7,), f"GP predict shape={pred.shape}"
    assert not np.any(np.isnan(pred)), "GP predict has NaN"
    assert not np.any(np.isinf(pred)), "GP predict has Inf"

    # Also test predict_with_uncertainty (inherited from BaseModel)
    pred2, unc = gp.predict_with_uncertainty(s_test, a_test)
    assert pred2.shape == (7,), "GP predict_with_uncertainty shape wrong"
    assert unc is None, "GP predict_with_uncertainty should return None for std"

    detail = (f"gp.n_gps={len(gp._gps)}, predict_shape={pred.shape}, "
              f"predict_val={np.array2string(pred, precision=4)}")
    record("test4_gp_train_predict", PASS, detail, time.time() - t0)
except Exception as e:
    record("test4_gp_train_predict", FAIL, f"{e}\n{traceback.format_exc()}", time.time() - t0)

# ============================================================
# TEST 5: SINDyModel train + predict
# ============================================================
print("\n=== TEST 5: SINDyModel train+predict ===")
t0 = time.time()
try:
    sindy = SINDyModel()
    sindy.train(
        data['train_states'][:500],
        data['train_actions'][:500],
        data['train_deltas'][:500],
        data['state_std'], data['action_std'], data['delta_std']
    )
    assert sindy._coefficients is not None, "SINDy coefficients not set"
    assert sindy._coefficients.ndim == 2, f"SINDy coefficients ndim={sindy._coefficients.ndim}"

    s_test = data['test_states'][0]
    a_test = float(data['test_actions'][0])
    pred = sindy.predict(s_test, a_test)
    assert pred.shape == (7,), f"SINDy predict shape={pred.shape}"
    assert not np.any(np.isnan(pred)), "SINDy predict has NaN"
    assert not np.any(np.isinf(pred)), "SINDy predict has Inf"

    detail = (f"coeffs_shape={sindy._coefficients.shape}, "
              f"predict_shape={pred.shape}, "
              f"predict_val={np.array2string(pred, precision=4)}")
    record("test5_sindy_train_predict", PASS, detail, time.time() - t0)
except Exception as e:
    record("test5_sindy_train_predict", FAIL, f"{e}\n{traceback.format_exc()}", time.time() - t0)

# ============================================================
# TEST 6: compute_metrics() with synthetic data
# ============================================================
print("\n=== TEST 6: compute_metrics ===")
t0 = time.time()
try:
    np.random.seed(99)
    n_steps = 20
    states_model = np.random.randn(n_steps + 1, 7)
    states_real = np.random.randn(n_steps + 1, 7) + 0.1
    state_std = np.ones(7) * 0.5

    m = compute_metrics(states_model, states_real, state_std, n_steps)

    # Check structure
    assert 'overall' in m, "Missing 'overall' key"
    assert 'stability' in m, "Missing 'stability' key"
    for name in STATE_NAMES_7D:
        assert name in m, f"Missing per-state key '{name}'"

    # Check overall metrics
    for key in ['mae', 'rmse', 'nmae', 'nrmse', 'max_error', 'p95_error', 'endpoint_error']:
        assert key in m['overall'], f"Missing overall metric '{key}'"
        assert isinstance(m['overall'][key], float), f"overall.{key} type={type(m['overall'][key])}"

    # Check per-state metrics
    for name in STATE_NAMES_7D:
        for key in ['mae', 'rmse', 'nmae', 'max_error', 'p95_error', 'endpoint_error']:
            assert key in m[name], f"Missing {name}.{key}"

    assert m['stability']['survival_steps'] >= 0

    detail = (f"overall_nmae={m['overall']['nmae']:.4f}, "
              f"overall_rmse={m['overall']['rmse']:.4f}, "
              f"survival={m['stability']['survival_steps']}")
    record("test6_compute_metrics", PASS, detail, time.time() - t0)
except Exception as e:
    record("test6_compute_metrics", FAIL, f"{e}\n{traceback.format_exc()}", time.time() - t0)

# ============================================================
# TEST 7: compute_cost() with synthetic data
# ============================================================
print("\n=== TEST 7: compute_cost ===")
t0 = time.time()
try:
    np.random.seed(100)
    n = 10
    states = np.random.randn(n + 1, 7)
    actions = np.random.randn(n)
    weights = {
        'e_y': 100.0, 'e_psi': 100.0, 'v': 1.0,
        'theta': 10.0, 'theta_dot': 1.0,
        'delta': 10.0, 'delta_dot': 1.0, 'u': 0.1
    }

    J = compute_cost(states, actions, weights)
    assert isinstance(J, float), f"compute_cost return type={type(J)}"
    assert not np.isnan(J), "compute_cost returned NaN"
    assert not np.isinf(J), "compute_cost returned Inf"
    assert J >= 0.0, f"compute_cost returned negative: {J}"

    # Verify with manual calculation
    J_manual = 0.0
    for k in range(n):
        s = states[k + 1]
        u = actions[k]
        J_manual += 100 * s[0]**2 + 100 * s[1]**2 + 1 * s[2]**2
        J_manual += 10 * s[3]**2 + 1 * s[4]**2 + 10 * s[5]**2 + 1 * s[6]**2
        J_manual += 0.1 * u**2
    assert abs(J - J_manual) < 1e-10, f"J={J} != J_manual={J_manual}"

    # Edge case: single state
    J_edge = compute_cost(states[:1], actions[:0], weights)
    assert J_edge == 0.0, f"Edge case: J={J_edge}"

    detail = f"J={J:.4f}, J_manual={J_manual:.4f}, edge_case_J={J_edge}"
    record("test7_compute_cost", PASS, detail, time.time() - t0)
except Exception as e:
    record("test7_compute_cost", FAIL, f"{e}\n{traceback.format_exc()}", time.time() - t0)

# ============================================================
# TEST 8: All model names are unique
# ============================================================
print("\n=== TEST 8: Model names unique ===")
t0 = time.time()
try:
    model_instances = [
        GPModel(max_samples=100, n_restarts=1),
        SINDyModel(),
        E1OfflineNN(n_models=1, n_epochs=2),
        RDELocal(n_models=1, n_epochs=2),
        RDETrajectory(n_models=1, n_epochs=2, dagger_rounds=0),
        RDEHybrid(n_models=1, n_epochs=2, dagger_rounds=0),
    ]
    names = [m.name() for m in model_instances]
    assert len(names) == len(set(names)), f"Duplicate names found: {names}"

    expected_names = {'gp', 'sindy', 'e1_offline_nn', 'rde_local', 'rde_trajectory', 'rde_hybrid'}
    assert set(names) == expected_names, f"Name mismatch: got {set(names)}, expected {expected_names}"

    detail = f"names={names}, all_unique=True"
    record("test8_model_names_unique", PASS, detail, time.time() - t0)
except Exception as e:
    record("test8_model_names_unique", FAIL, f"{e}\n{traceback.format_exc()}", time.time() - t0)

# ============================================================
# TEST 9: predict() returns 7D arrays for all models
# ============================================================
print("\n=== TEST 9: predict() returns 7D ===")
t0 = time.time()
try:
    # Train E1, RDE-L briefly for predict test
    small_states = data['train_states'][:200]
    small_actions = data['train_actions'][:200]
    small_deltas = data['train_deltas'][:200]

    test_models = []

    # GP
    gp = GPModel(max_samples=100, n_restarts=1)
    gp.train(small_states, small_actions, small_deltas,
             data['state_std'], data['action_std'], data['delta_std'])
    test_models.append(('gp', gp))

    # SINDy
    sindy = SINDyModel()
    sindy.train(small_states, small_actions, small_deltas,
                data['state_std'], data['action_std'], data['delta_std'])
    test_models.append(('sindy', sindy))

    # E1
    e1 = E1OfflineNN(n_models=2, n_epochs=5, residual_scale=0.3)
    e1.train(small_states, small_actions, small_deltas,
             data['state_std'], data['action_std'], data['delta_std'])
    test_models.append(('e1', e1))

    # RDE-L
    rde_l = RDELocal(n_models=2, n_epochs=5, residual_scale=0.3)
    rde_l.train(small_states, small_actions, small_deltas,
                data['state_std'], data['action_std'], data['delta_std'])
    test_models.append(('rde_l', rde_l))

    s_test = data['test_states'][0]
    a_test = float(data['test_actions'][0])

    details = []
    for name, model in test_models:
        pred = model.predict(s_test, a_test)
        assert pred.shape == (7,), f"{name} predict shape={pred.shape}"
        assert not np.any(np.isnan(pred)), f"{name} predict has NaN"
        assert not np.any(np.isinf(pred)), f"{name} predict has Inf"
        details.append(f"{name}:shape=(7)")

    # RDE-T and RDE-M need DAgger rounds with segments, test with dagger_rounds=0
    rde_t = RDETrajectory(n_models=2, n_epochs=5, residual_scale=0.3, dagger_rounds=0)
    rde_t.train(small_states, small_actions, small_deltas,
                data['state_std'], data['action_std'], data['delta_std'])
    pred_t = rde_t.predict(s_test, a_test)
    assert pred_t.shape == (7,), f"rde_t predict shape={pred_t.shape}"
    details.append("rde_t:shape=(7)")

    rde_m = RDEHybrid(n_models=2, n_epochs=5, residual_scale=0.3, dagger_rounds=0)
    rde_m.train(small_states, small_actions, small_deltas,
                data['state_std'], data['action_std'], data['delta_std'])
    pred_m = rde_m.predict(s_test, a_test)
    assert pred_m.shape == (7,), f"rde_m predict shape={pred_m.shape}"
    details.append("rde_m:shape=(7)")

    detail = f"all 6 models return (7,): {', '.join(details)}"
    record("test9_predict_7d_all_models", PASS, detail, time.time() - t0)
except Exception as e:
    record("test9_predict_7d_all_models", FAIL, f"{e}\n{traceback.format_exc()}", time.time() - t0)

# ============================================================
# TEST 10: E1/RDE-L/RDE-T/RDE-M all have predict_with_uncertainty()
# ============================================================
print("\n=== TEST 10: predict_with_uncertainty ===")
t0 = time.time()
try:
    s_test = data['test_states'][0]
    a_test = float(data['test_actions'][0])

    uncertainty_models = [
        ('e1_offline_nn', e1),
        ('rde_local', rde_l),
        ('rde_trajectory', rde_t),
        ('rde_hybrid', rde_m),
    ]

    details = []
    for name, model in uncertainty_models:
        assert hasattr(model, 'predict_with_uncertainty'), \
            f"{name} missing predict_with_uncertainty"

        s_next, unc = model.predict_with_uncertainty(s_test, a_test)
        assert s_next.shape == (7,), f"{name} s_next shape={s_next.shape}"
        assert unc.shape == (7,), f"{name} uncertainty shape={unc.shape}"
        assert not np.any(np.isnan(s_next)), f"{name} s_next has NaN"
        assert not np.any(np.isnan(unc)), f"{name} uncertainty has NaN"
        assert not np.any(np.isinf(s_next)), f"{name} s_next has Inf"
        # Uncertainty should be non-negative
        assert np.all(unc >= 0), f"{name} has negative uncertainty: {unc}"
        details.append(f"{name}:unc_mean={np.mean(unc):.4f}")

    # Also verify GP returns None uncertainty (inherited default)
    gp_pred, gp_unc = gp.predict_with_uncertainty(s_test, a_test)
    assert gp_pred.shape == (7,), "GP predict_with_uncertainty shape wrong"
    assert gp_unc is None, "GP should return None uncertainty"
    details.append("gp:unc=None(default)")

    # Also verify SINDy returns None uncertainty
    sindy_pred, sindy_unc = sindy.predict_with_uncertainty(s_test, a_test)
    assert sindy_pred.shape == (7,)
    assert sindy_unc is None, "SINDy should return None uncertainty"
    details.append("sindy:unc=None(default)")

    detail = f"all 6 models have predict_with_uncertainty: {', '.join(details)}"
    record("test10_predict_with_uncertainty", PASS, detail, time.time() - t0)
except Exception as e:
    record("test10_predict_with_uncertainty", FAIL, f"{e}\n{traceback.format_exc()}", time.time() - t0)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("AGENT B INTERFACE VERIFICATION SUMMARY")
print("=" * 60)

n_pass = sum(1 for v in results.values() if v['status'] == PASS)
n_fail = sum(1 for v in results.values() if v['status'] == FAIL)
n_total = len(results)

for name, info in results.items():
    print(f"  [{info['status']}] {name} ({info['elapsed_s']:.1f}s): {info['detail'][:120]}")

print(f"\nTotal: {n_total}, PASS: {n_pass}, FAIL: {n_fail}")

if n_fail > 0:
    print("\nFAILED TESTS:")
    for name, info in results.items():
        if info['status'] == FAIL:
            print(f"  - {name}: {info['detail'][:200]}")

# ============================================================
# SAVE JSON
# ============================================================
output_path = Path("D:/系统辨识作业/sindy_bicycle/continuation_stage_v8/raw_results/AGENT_B_INTERFACE_VERIFY.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

output = {
    "agent": "B",
    "task": "canonical_7d package interface verification",
    "total_tests": n_total,
    "passed": n_pass,
    "failed": n_fail,
    "results": results,
}

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: {output_path}")
print("DONE")
