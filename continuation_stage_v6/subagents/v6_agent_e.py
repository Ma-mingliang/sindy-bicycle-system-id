"""V6 Subagent E: Uncertainty Calibration and OOD Detection.

Re-verify that RDE-M's ensemble uncertainty is properly calibrated and
useful for out-of-distribution detection.

Methodology:
  1. Generate 5000 samples, split into train=2500, cal=1250, test=1250
  2. Train RDE-M on training set
  3. Evaluate raw uncertainty on test set (correlation, coverage)
  4. Calibrate on calibration set (scalar + per-state scaling)
  5. Re-evaluate calibrated uncertainty on test set
  6. OOD detection test with AUROC/AUPRC
"""
import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

sys.path.insert(0, str(Path(__file__).parent.parent))
from canonical_4d.config import DynamicsConfig, TrainingConfig
from canonical_4d.dynamics import BicycleDynamics
from canonical_4d.residual_models import RDEHybrid
from canonical_4d.models import GPModel

# Patch GPModel to use fewer samples (avoid OOM on 2000x2000 kernel matrix)
_ORIG_GP_INIT = GPModel.__init__
def _patched_gp_init(self, max_samples=1000, n_restarts=2):
    _ORIG_GP_INIT(self, max_samples=min(max_samples, 1000), n_restarts=n_restarts)
GPModel.__init__ = _patched_gp_init

STATE_NAMES = ['phi', 'delta', 'phi_dot', 'delta_dot']
CONFIDENCE_LEVELS = [0.68, 0.90, 0.95, 0.99]
# z-scores for two-sided normal confidence intervals
Z_SCORES = {
    0.68: 1.0,
    0.90: 1.6449,
    0.95: 1.9600,
    0.99: 2.5758,
}


def generate_data(n_samples, seed, dynamics):
    """Generate (s, tau, s_next) triplets with known distribution."""
    rng = np.random.RandomState(seed)
    phis = rng.uniform(-0.5, 0.5, n_samples)
    deltas = rng.uniform(-0.3, 0.3, n_samples)
    phi_dots = rng.uniform(-2.0, 2.0, n_samples)
    delta_dots = rng.uniform(-1.0, 1.0, n_samples)
    taus = rng.uniform(-50, 50, n_samples)

    states = np.column_stack([phis, deltas, phi_dots, delta_dots])
    s_nexts = np.empty_like(states)
    for i in range(n_samples):
        s_nexts[i] = dynamics.step(states[i], taus[i])

    return states, taus, s_nexts


def compute_raw_coverage(errors, uncertainties, confidence_levels):
    """Compute coverage: fraction of |error| < z_alpha * uncertainty.

    Args:
        errors: (N, 4) absolute errors
        uncertainties: (N, 4) predicted uncertainties
        confidence_levels: list of target coverage levels

    Returns:
        dict: coverage[alpha][state_name] = fraction covered
    """
    n_states = errors.shape[1]
    result = {}
    for alpha in confidence_levels:
        z = Z_SCORES[alpha]
        state_coverage = {}
        for i in range(n_states):
            covered = np.sum(np.abs(errors[:, i]) < z * uncertainties[:, i])
            state_coverage[STATE_NAMES[i]] = float(covered / len(errors))
        # Overall coverage (all 4 states simultaneously)
        all_covered = np.all(np.abs(errors) < z * uncertainties, axis=1)
        state_coverage['all'] = float(np.mean(all_covered))
        result[str(alpha)] = state_coverage
    return result


def compute_correlation(errors, uncertainties):
    """Pearson correlation between |error| and uncertainty per state."""
    n_states = errors.shape[1]
    result = {}
    for i in range(n_states):
        r, p = stats.pearsonr(np.abs(errors[:, i]), uncertainties[:, i])
        result[STATE_NAMES[i]] = {'pearson_r': float(r), 'p_value': float(p)}
    return result


def find_scalar_scale(cal_errors, cal_uncertainties, target_alpha=0.95):
    """Find scalar s that minimizes |coverage_95(s) - target|.

    Tests candidate scales and picks the one closest to target coverage.
    """
    z = Z_SCORES[target_alpha]
    best_s = 1.0
    best_gap = float('inf')

    # Search over scales from 0.1 to 5.0
    candidates = np.linspace(0.1, 5.0, 200)
    for s in candidates:
        covered = np.all(np.abs(cal_errors) < z * s * cal_uncertainties, axis=1)
        actual_coverage = np.mean(covered)
        gap = abs(actual_coverage - target_alpha)
        if gap < best_gap:
            best_gap = gap
            best_s = s

    return float(best_s), float(best_gap)


def find_per_state_scale(cal_errors, cal_uncertainties, target_alpha=0.95):
    """Find per-state scale s_i that minimizes |coverage_i(s_i) - target|."""
    z = Z_SCORES[target_alpha]
    n_states = cal_errors.shape[1]
    scales = np.ones(n_states)
    gaps = np.zeros(n_states)

    for i in range(n_states):
        best_s = 1.0
        best_gap = float('inf')
        candidates = np.linspace(0.1, 5.0, 200)
        for s in candidates:
            covered = np.abs(cal_errors[:, i]) < z * s * cal_uncertainties[:, i]
            actual = np.mean(covered)
            gap = abs(actual - target_alpha)
            if gap < best_gap:
                best_gap = gap
                best_s = s
        scales[i] = best_s
        gaps[i] = best_gap

    return scales.tolist(), gaps.tolist()


def compute_ece(errors, uncertainties, n_bins=10):
    """Expected Calibration Error.

    For each sample, compute the predicted probability that error < z_0.95 * sigma.
    Bin by predicted probability, compare with actual fraction.
    """
    z_95 = Z_SCORES[0.95]
    n_states = errors.shape[1]

    # Per-state ECE
    ece_per_state = {}
    for i in range(n_states):
        # Predicted confidence: probability that |e_i| < z_95 * sigma_i
        # Assuming Gaussian: P = erf(z_95 / sqrt(2)) ≈ 0.95 (constant for Gaussian)
        # Better approach: use the actual |error| / sigma ratio
        ratios = np.abs(errors[:, i]) / (uncertainties[:, i] + 1e-10)
        # Map to [0, 1] via CDF of half-normal
        predicted_conf = 1.0 - np.exp(-0.5 * (ratios / z_95) ** 2)
        predicted_conf = np.clip(predicted_conf, 0, 1)

        # Actual: is error within z_95 * sigma?
        actual_correct = (np.abs(errors[:, i]) < z_95 * uncertainties[:, i]).astype(float)

        # Bin by predicted confidence
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = len(predicted_conf)
        for b in range(n_bins):
            mask = (predicted_conf >= bin_edges[b]) & (predicted_conf < bin_edges[b + 1])
            if b == n_bins - 1:  # include right edge for last bin
                mask = (predicted_conf >= bin_edges[b]) & (predicted_conf <= bin_edges[b + 1])
            n_bin = np.sum(mask)
            if n_bin > 0:
                avg_pred = np.mean(predicted_conf[mask])
                avg_actual = np.mean(actual_correct[mask])
                ece += (n_bin / total_samples) * abs(avg_pred - avg_actual)

        ece_per_state[STATE_NAMES[i]] = float(ece)

    # Overall ECE (averaged across states)
    ece_per_state['overall'] = float(np.mean(list(ece_per_state.values())))
    return ece_per_state


def compute_ood_detection(test_errors, test_uncertainties, ood_errors, ood_uncertainties):
    """OOD detection metrics: AUROC, AUPRC, OOD/ID uncertainty ratio.

    ID samples have lower uncertainty, OOD samples have higher uncertainty.
    We use the mean uncertainty across states as the detection score.
    """
    # Mean uncertainty across states
    id_scores = np.mean(test_uncertainties, axis=1)
    ood_scores = np.mean(ood_uncertainties, axis=1)

    # Combine for AUROC/AUPRC
    all_scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])

    # Compute AUROC using trapezoidal rule
    sorted_idx = np.argsort(all_scores)[::-1]  # descending
    sorted_labels = labels[sorted_idx]
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0
    for lab in sorted_labels:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # AUROC = area under ROC curve
    auroc = 0.0
    for i in range(1, len(tpr_list)):
        auroc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2.0

    # Compute AUPRC
    precision_list = [1.0]
    recall_list = [0.0]
    tp = 0
    fp = 0
    for lab in sorted_labels:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        precision_list.append(tp / (tp + fp))
        recall_list.append(tp / n_pos)

    auprc = 0.0
    for i in range(1, len(recall_list)):
        auprc += (recall_list[i] - recall_list[i - 1]) * precision_list[i]

    # OOD/ID uncertainty ratio
    ratio = float(np.mean(ood_scores) / (np.mean(id_scores) + 1e-10))

    return {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'ood_mean_uncertainty': float(np.mean(ood_scores)),
        'id_mean_uncertainty': float(np.mean(id_scores)),
        'ood_id_ratio': ratio,
    }


def generate_ood_data(n_samples, seed, dynamics):
    """Generate OOD data: states with |phi| > 0.8 (outside training range of [-0.5, 0.5])."""
    rng = np.random.RandomState(seed)
    # OOD: extreme lean angles
    phis = rng.choice([-1, 1], n_samples) * rng.uniform(0.8, 1.5, n_samples)
    deltas = rng.uniform(-0.3, 0.3, n_samples)
    phi_dots = rng.uniform(-2.0, 2.0, n_samples)
    delta_dots = rng.uniform(-1.0, 1.0, n_samples)
    taus = rng.uniform(-50, 50, n_samples)

    states = np.column_stack([phis, deltas, phi_dots, delta_dots])
    s_nexts = np.empty_like(states)
    for i in range(n_samples):
        s_nexts[i] = dynamics.step(states[i], taus[i])

    return states, taus, s_nexts


def main():
    print("=" * 80)
    print("V6 SUBAGENT E: Uncertainty Calibration and OOD Detection")
    print("=" * 80)
    t0 = time.time()

    # ---- Step 1: Generate data ----
    print("\n[1] Generating data...")
    dyn = BicycleDynamics()
    total_samples = 5000
    all_states, all_actions, all_s_nexts = generate_data(total_samples, 42, dyn)

    # Split: train=2500, cal=1250, test=1250
    train_states = all_states[:2500]
    train_actions = all_actions[:2500]
    train_s_nexts = all_s_nexts[:2500]

    cal_states = all_states[2500:3750]
    cal_actions = all_actions[2500:3750]
    cal_s_nexts = all_s_nexts[2500:3750]

    test_states = all_states[3750:5000]
    test_actions = all_actions[3750:5000]
    test_s_nexts = all_s_nexts[3750:5000]

    print(f"  Train: {len(train_states)}, Cal: {len(cal_states)}, Test: {len(test_states)}")

    # Compute normalization stats from training set
    state_std = np.std(train_states, axis=0)
    action_std = float(np.std(train_actions))
    train_deltas = train_s_nexts - train_states
    delta_std = np.std(train_deltas, axis=0)

    # ---- Step 2: Train RDE-M ----
    print("\n[2] Training RDE-M (n_models=5, dagger_rounds=2)...")
    t1 = time.time()
    rde_m = RDEHybrid(
        n_models=5, n_epochs=50, residual_scale=0.3,
        dagger_rounds=2, n_segments=3, segment_length=500
    )
    rde_m.train(
        train_states, train_actions, train_deltas,
        state_std, action_std, delta_std,
        dynamics=dyn, make_tau_func=dyn.generate_lqr_tau
    )
    print(f"  RDE-M trained in {time.time() - t1:.1f}s")

    # ---- Step 3: Evaluate on test set (raw uncertainty) ----
    print("\n[3] Evaluating raw uncertainty on test set...")
    n_test = len(test_states)
    test_preds = np.empty_like(test_states)
    test_uncs = np.empty_like(test_states)
    for i in range(n_test):
        pred, unc = rde_m.predict_with_uncertainty(test_states[i], test_actions[i])
        test_preds[i] = pred
        test_uncs[i] = unc

    test_errors = test_preds - test_s_nexts

    # 3a: Pearson correlation
    raw_corr = compute_correlation(test_errors, test_uncs)
    print("\n  Raw Pearson correlation (|error| vs uncertainty):")
    for name, vals in raw_corr.items():
        print(f"    {name}: r={vals['pearson_r']:.4f}, p={vals['p_value']:.4e}")

    # 3b: Raw coverage
    raw_coverage = compute_raw_coverage(test_errors, test_uncs, CONFIDENCE_LEVELS)
    print("\n  Raw coverage:")
    for alpha, covs in raw_coverage.items():
        vals = [f"{covs[s]:.3f}" for s in STATE_NAMES + ['all']]
        print(f"    {alpha}: {', '.join(vals)}")

    # ---- Step 4: Calibrate on calibration set ----
    print("\n[4] Calibrating on calibration set...")
    n_cal = len(cal_states)
    cal_preds = np.empty_like(cal_states)
    cal_uncs = np.empty_like(cal_states)
    for i in range(n_cal):
        pred, unc = rde_m.predict_with_uncertainty(cal_states[i], cal_actions[i])
        cal_preds[i] = pred
        cal_uncs[i] = unc

    cal_errors = cal_preds - cal_s_nexts

    # 4a: Scalar scaling
    print("  Finding scalar scale factor...")
    scalar_scale, scalar_gap = find_scalar_scale(cal_errors, cal_uncs)
    print(f"    Scalar scale: {scalar_scale:.4f} (gap from 95%: {scalar_gap:.4f})")

    # 4b: Per-state scaling
    print("  Finding per-state scale factors...")
    per_state_scales, per_state_gaps = find_per_state_scale(cal_errors, cal_uncs)
    for i, name in enumerate(STATE_NAMES):
        print(f"    {name}: scale={per_state_scales[i]:.4f} (gap: {per_state_gaps[i]:.4f})")

    # ---- Step 5: Evaluate calibrated uncertainty on test set ----
    print("\n[5] Evaluating calibrated uncertainty on test set...")

    # Scalar calibrated coverage
    calibrated_test_uncs_scalar = test_uncs * scalar_scale
    cal_coverage_scalar = compute_raw_coverage(test_errors, calibrated_test_uncs_scalar, CONFIDENCE_LEVELS)
    print("\n  Scalar-calibrated coverage:")
    for alpha, covs in cal_coverage_scalar.items():
        vals = [f"{covs[s]:.3f}" for s in STATE_NAMES + ['all']]
        print(f"    {alpha}: {', '.join(vals)}")

    # Per-state calibrated coverage
    scale_arr = np.array(per_state_scales)
    calibrated_test_uncs_perstate = test_uncs * scale_arr
    cal_coverage_perstate = compute_raw_coverage(test_errors, calibrated_test_uncs_perstate, CONFIDENCE_LEVELS)
    print("\n  Per-state-calibrated coverage:")
    for alpha, covs in cal_coverage_perstate.items():
        vals = [f"{covs[s]:.3f}" for s in STATE_NAMES + ['all']]
        print(f"    {alpha}: {', '.join(vals)}")

    # 5b: ECE
    ece_raw = compute_ece(test_errors, test_uncs)
    ece_scalar = compute_ece(test_errors, calibrated_test_uncs_scalar)
    ece_perstate = compute_ece(test_errors, calibrated_test_uncs_perstate)
    print("\n  ECE (Expected Calibration Error):")
    print(f"    Raw:      {ece_raw['overall']:.4f}")
    print(f"    Scalar:   {ece_scalar['overall']:.4f}")
    print(f"    Per-state: {ece_perstate['overall']:.4f}")

    # ---- Step 6: OOD detection ----
    print("\n[6] OOD detection test...")
    ood_states, ood_actions, ood_s_nexts = generate_ood_data(1000, 99, dyn)

    ood_preds = np.empty_like(ood_states)
    ood_uncs = np.empty_like(ood_states)
    for i in range(len(ood_states)):
        pred, unc = rde_m.predict_with_uncertainty(ood_states[i], ood_actions[i])
        ood_preds[i] = pred
        ood_uncs[i] = unc

    ood_errors = ood_preds - ood_s_nexts

    # Use ID test set (same as above) vs OOD set
    ood_metrics = compute_ood_detection(test_errors, test_uncs, ood_errors, ood_uncs)
    print(f"  AUROC: {ood_metrics['auroc']:.4f}")
    print(f"  AUPRC: {ood_metrics['auprc']:.4f}")
    print(f"  ID mean uncertainty:  {ood_metrics['id_mean_uncertainty']:.6f}")
    print(f"  OOD mean uncertainty: {ood_metrics['ood_mean_uncertainty']:.6f}")
    print(f"  OOD/ID ratio:         {ood_metrics['ood_id_ratio']:.2f}x")

    # ---- Save results ----
    results = {
        'subagent': 'E',
        'task': 'Uncertainty Calibration and OOD Detection',
        'data_split': {
            'total': total_samples,
            'train': 2500,
            'cal': 1250,
            'test': 1250,
            'ood': 1000,
        },
        'raw_results': {
            'correlation': raw_corr,
            'coverage': raw_coverage,
        },
        'calibration': {
            'scalar_scale': scalar_scale,
            'scalar_gap': scalar_gap,
            'per_state_scales': {STATE_NAMES[i]: per_state_scales[i] for i in range(4)},
            'per_state_gaps': {STATE_NAMES[i]: per_state_gaps[i] for i in range(4)},
        },
        'calibrated_results': {
            'scalar_coverage': cal_coverage_scalar,
            'per_state_coverage': cal_coverage_perstate,
        },
        'ece': {
            'raw': ece_raw,
            'scalar': ece_scalar,
            'per_state': ece_perstate,
        },
        'ood_detection': ood_metrics,
        'elapsed_seconds': time.time() - t0,
    }

    output_path = Path(__file__).parent / "AGENT_E_RESULTS.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    print(f"Total time: {time.time() - t0:.1f}s")

    return results


if __name__ == "__main__":
    main()
