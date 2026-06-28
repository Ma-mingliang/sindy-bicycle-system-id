"""V6 Subagent C: Uncertainty Calibration, Reliability, and OOD Audit.

C1: Unified terminology
C2: Calibration data leakage check
C3: Proper calibration methods
C4: Reliability metrics (ECE, NLL, etc.)
C5: Rollout uncertainty
C6: Hard OOD detection
"""
import sys, os, json, time, warnings
import numpy as np
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

sys.path.insert(0, str(Path(__file__).parent.parent))
from canonical_4d.config import DynamicsConfig, TrainingConfig, CalibConfig
from canonical_4d.dynamics import BicycleDynamics
from canonical_4d.models import GPModel
from canonical_4d.residual_models import E1OfflineNN
from canonical_4d.evaluation_modes import run_mode_b
from canonical_4d.runner import generate_training_data
from canonical_4d.metrics import compute_metrics

OUTPUT = Path(__file__).parent.parent / "raw_results"
OUTPUT.mkdir(exist_ok=True)


def compute_ece(errors, uncertainties, n_bins=10):
    """ECE with proper shape handling."""
    abs_err = np.abs(errors)
    std = np.mean(uncertainties, axis=1) if uncertainties.ndim > 1 else uncertainties
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(abs_err)
    bin_data = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        z_lo = stats.norm.ppf(0.5 + lo/2) if lo > 0 else 0
        z_hi = stats.norm.ppf(0.5 + hi/2) if hi < 1 else 10
        mask = (std >= z_lo * 0.1) & (std < z_hi * 0.1)
        if np.sum(mask) < 2:
            continue
        z = stats.norm.ppf(0.5 + (hi+lo)/2/2)
        std_col = std[mask][:, np.newaxis] if abs_err[mask].ndim > 1 else std[mask]
        empirical = float(np.mean(np.abs(errors[mask]) <= z * std_col))
        nominal = (hi + lo) / 2
        gap = abs(empirical - nominal)
        ece += gap * np.sum(mask) / total
        bin_data.append({'nominal': nominal, 'empirical': empirical, 'gap': gap, 'count': int(np.sum(mask))})

    return {'ece': float(ece), 'bins': bin_data}


def compute_coverage(errors, uncertainties, levels=[0.68, 0.90, 0.95, 0.99]):
    """Per-state and average coverage."""
    results = {}
    for cl in levels:
        z = stats.norm.ppf(0.5 + cl/2)
        covered = np.abs(errors) <= z * uncertainties
        coverage = np.mean(covered, axis=0)
        results[f'coverage_{int(cl*100)}'] = coverage.tolist()
        results[f'avg_coverage_{int(cl*100)}'] = float(np.mean(coverage))
    return results


def main():
    print("="*80)
    print("V6 SUBAGENT C: Uncertainty Calibration & OOD Audit")
    print("="*80)
    t0 = time.time()

    dyn = BicycleDynamics()
    cal_cfg = CalibConfig()

    # Generate data with explicit splits
    print("Generating data with train/cal/test splits...")
    n_total = 8000
    states_all, actions_all, deltas_all, state_std, action_std, delta_std = generate_training_data(n_total, 42, dyn)

    # Split: train 60%, cal 20%, test 20% (random single-step, no episode leakage)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_total)
    n_train = int(0.6 * n_total)
    n_cal = int(0.2 * n_total)
    train_idx = perm[:n_train]
    cal_idx = perm[n_train:n_train+n_cal]
    test_idx = perm[n_train+n_cal:]

    states_train, actions_train, deltas_train = states_all[train_idx], actions_all[train_idx], deltas_all[train_idx]
    states_cal, actions_cal, deltas_cal = states_all[cal_idx], actions_all[cal_idx], deltas_all[cal_idx]
    states_test, actions_test, deltas_test = states_all[test_idx], actions_all[test_idx], deltas_all[test_idx]

    print(f"  Train: {len(train_idx)}, Cal: {len(cal_idx)}, Test: {len(test_idx)}")

    # Check for leakage
    train_set = set(map(tuple, states_train))
    cal_set = set(map(tuple, states_cal))
    test_set = set(map(tuple, states_test))
    leak_cal = len(train_set & cal_set)
    leak_test = len(train_set & test_set)
    print(f"  Leakage: train&cal={leak_cal}, train&test={leak_test}")

    # Train model
    print("\nTraining GP model...")
    gp = GPModel(max_samples=2000, n_restarts=2)
    gp.train(states_train, actions_train, deltas_train, state_std, action_std, delta_std)

    # Compute predictions on cal and test
    print("Computing predictions...")
    def get_preds(states, actions):
        preds = np.empty_like(states)
        for i in range(len(states)):
            preds[i] = gp.predict(states[i], actions[i]) - states[i]
        return preds

    pred_cal = get_preds(states_cal, actions_cal)
    pred_test = get_preds(states_test, actions_test)

    # Errors
    errors_cal = pred_cal - deltas_cal
    errors_test = pred_test - deltas_test

    # Uncertainty: use GP posterior std
    def get_stds(states, actions):
        stds = np.empty((len(states), 4))
        for i in range(len(states)):
            s_norm = states[i] / state_std
            a_norm = actions[i] / action_std
            x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
            for j, gp_j in enumerate(gp._gps):
                _, std = gp_j.predict(x, return_std=True)
                stds[i, j] = std[0] * delta_std[j]
        return stds

    print("Computing uncertainties (may take a moment)...")
    stds_cal = get_stds(states_cal, actions_cal)
    stds_test = get_stds(states_test, actions_test)

    # C1-C2: Terminology and leakage report
    print("\n--- C1-C2: Data Splits & Leakage ---")
    leakage_report = {
        'train_size': len(train_idx),
        'cal_size': len(cal_idx),
        'test_size': len(test_idx),
        'leakage_train_cal': leak_cal,
        'leakage_train_test': leak_test,
        'random_split': True,
        'episode_based': False,
        'note': 'Single-step random splits. No episode structure in training data. Rollout calibration is separate.'
    }
    print(f"  Leakage check: train&cal={leak_cal}, train&test={leak_test}")

    # C3: Calibration
    print("\n--- C3: Calibration ---")
    cal_coverage = compute_coverage(errors_cal, stds_cal, cal_cfg.confidence_levels)
    print(f"  Cal set raw 95% coverage: {cal_coverage['avg_coverage_95']:.4f}")

    # Find optimal scalar scale
    best_scale = 1.0
    best_gap = float('inf')
    for scale in cal_cfg.scale_search_range:
        scaled_stds = stds_cal * scale
        cov = compute_coverage(errors_cal, scaled_stds, [0.95])
        gap = abs(cov['avg_coverage_95'] - cal_cfg.target_coverage)
        if gap < best_gap:
            best_gap = gap
            best_scale = scale

    print(f"  Optimal scalar scale: {best_scale:.3f} (gap={best_gap:.4f})")

    # Test set with calibrated scale
    test_coverage = compute_coverage(errors_test, stds_test * best_scale, cal_cfg.confidence_levels)
    print(f"  Test 95% coverage (calibrated): {test_coverage['avg_coverage_95']:.4f}")

    # Per-state calibration
    per_state = {}
    for i, name in enumerate(['phi', 'delta', 'phi_dot', 'delta_dot']):
        state_errors = errors_test[:, i:i+1]
        state_stds = stds_test[:, i:i+1] * best_scale
        sc = compute_coverage(state_errors, state_stds, [0.95])
        per_state[name] = {
            'scale': best_scale,
            'test_coverage_95': sc['avg_coverage_95'],
        }
        print(f"  {name}: scale={best_scale:.3f}, coverage_95={sc['avg_coverage_95']:.4f}")

    # C4: Reliability metrics
    print("\n--- C4: Reliability Metrics ---")
    ece_raw = compute_ece(errors_cal, stds_cal, cal_cfg.n_bins_ece)
    ece_cal = compute_ece(errors_test, stds_test * best_scale, cal_cfg.n_bins_ece)
    print(f"  Raw ECE (cal set): {ece_raw['ece']:.4f}")
    print(f"  Calibrated ECE (test): {ece_cal['ece']:.4f}")

    # NLL
    nll = 0.0
    for i in range(len(errors_test)):
        for j in range(4):
            s = stds_test[i, j] * best_scale
            if s > 1e-10:
                nll += 0.5 * np.log(2 * np.pi * s**2) + 0.5 * (errors_test[i, j] / s)**2
    nll /= len(errors_test)
    print(f"  NLL (test): {nll:.4f}")

    # C5: Rollout uncertainty
    print("\n--- C5: Rollout Uncertainty ---")
    # Train ensemble for rollout uncertainty
    from canonical_4d.residual_models import E1OfflineNN
    e1 = E1OfflineNN(n_models=5, n_epochs=50, residual_scale=0.3)
    e1.train(states_train, actions_train, deltas_train, state_std, action_std, delta_std)

    rollout_results = {}
    for horizon in [10, 50, 100, 200]:
        results = []
        for seg in range(5):
            tau_func = dyn.generate_lqr_tau(seg + 200, horizon + 10)
            s0 = np.array([np.random.uniform(-0.15, 0.15), 0.0, 0.0, 0.0])
            result = run_mode_b(e1, s0, tau_func, horizon, dyn)
            n = min(horizon + 1, len(result.states_model))
            if result.uncertainties is not None and len(result.uncertainties) >= n - 1:
                unc = result.uncertainties[:n-1]
                err = np.abs(result.states_model[1:n] - result.states_real[1:n])
                # Average uncertainty vs average error
                mean_unc = float(np.mean(unc))
                mean_err = float(np.mean(err))
                results.append({'mean_uncertainty': mean_unc, 'mean_error': mean_err})
        if results:
            avg_unc = np.mean([r['mean_uncertainty'] for r in results])
            avg_err = np.mean([r['mean_error'] for r in results])
            rollout_results[str(horizon)] = {
                'mean_uncertainty': float(avg_unc),
                'mean_error': float(avg_err),
                'ratio': float(avg_unc / avg_err) if avg_err > 1e-10 else float('inf'),
            }
            print(f"  {horizon}-step: unc={avg_unc:.6f}, err={avg_err:.6f}, ratio={avg_unc/avg_err:.2f}")

    # C6: Hard OOD
    print("\n--- C6: Hard OOD Detection ---")
    ood_results = {}

    # OOD Type 1: State out of range
    ood1_states = rng.uniform([-0.8, -0.5, -3, -1.5], [0.8, 0.5, 3, 1.5], (500, 4))
    ood1_actions = rng.uniform(-50, 50, 500)
    ood1_preds = get_preds(ood1_states, ood1_actions)
    ood1_errors = ood1_preds - np.zeros_like(ood1_preds)  # pseudo-errors
    ood1_stds = get_stds(ood1_states, ood1_actions)

    # OOD Type 2: Action extreme
    ood2_states = rng.uniform([-0.3, -0.2, -1, -0.5], [0.3, 0.2, 1, 0.5], (500, 4))
    ood2_actions = rng.uniform(-100, 100, 500)
    ood2_preds = get_preds(ood2_states, ood2_actions)
    ood2_errors = ood2_preds - np.zeros_like(ood2_preds)
    ood2_stds = get_stds(ood2_states, ood2_actions)

    # OOD Type 3: Near critical region
    ood3_states = np.column_stack([
        rng.uniform(0.4, 0.6, 500),
        rng.uniform(-0.3, 0.3, 500),
        rng.uniform(-2, 2, 500),
        rng.uniform(-1, 1, 500),
    ])
    ood3_actions = rng.uniform(-50, 50, 500)
    ood3_preds = get_preds(ood3_states, ood3_actions)
    ood3_errors = ood3_preds - np.zeros_like(ood3_preds)
    ood3_stds = get_stds(ood3_states, ood3_actions)

    # ID: test set
    id_stds = stds_test
    id_mean_std = float(np.mean(id_stds))

    ood_results = {
        'id_mean_uncertainty': id_mean_std,
        'ood_state_range': float(np.mean(ood1_stds)),
        'ood_action_extreme': float(np.mean(ood2_stds)),
        'ood_critical_region': float(np.mean(ood3_stds)),
    }
    print(f"  ID mean unc: {id_mean_std:.6f}")
    print(f"  OOD state range: {ood_results['ood_state_range']:.6f} (ratio={ood_results['ood_state_range']/id_mean_std:.1f}x)")
    print(f"  OOD action extreme: {ood_results['ood_action_extreme']:.6f} (ratio={ood_results['ood_action_extreme']/id_mean_std:.1f}x)")
    print(f"  OOD critical: {ood_results['ood_critical_region']:.6f} (ratio={ood_results['ood_critical_region']/id_mean_std:.1f}x)")

    # Save results
    output = {
        'subagent': 'C',
        'leakage_report': leakage_report,
        'calibration': {
            'optimal_scalar_scale': best_scale,
            'cal_set_coverage': cal_coverage,
            'test_set_coverage': test_coverage,
            'per_state': per_state,
        },
        'reliability': {
            'ece_raw': ece_raw,
            'ece_calibrated': ece_cal,
            'nll': float(nll),
        },
        'rollout_uncertainty': rollout_results,
        'ood': ood_results,
        'mode_b_equals_c': 'In current deterministic setup, Mode B and C use same tau_func from real state. Numerically equivalent but protocol semantics differ.',
    }

    with open(OUTPUT / "SUBAGENT_C_RESULTS.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nSubagent C done in {elapsed:.1f}s")
    return output


if __name__ == "__main__":
    main()
