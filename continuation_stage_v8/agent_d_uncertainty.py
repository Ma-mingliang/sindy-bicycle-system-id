"""Agent D: Uncertainty Calibration & OOD Detection for 7D GP+NN ensemble."""
import sys
import types
import warnings
warnings.filterwarnings("ignore")

# --- Numpy 2.x compatibility patch ---
# Import torch first (before mocking pandas) to avoid torch._dynamo issues
import torch

# Now mock pandas and narwhals for sklearn compatibility
_pd_mock = types.ModuleType('pandas')
_pd_mock.DataFrame = type('DataFrame', (), {})
_pd_mock.Series = type('Series', (), {})
_pd_mock.__path__ = []
_pd_mock.__file__ = '<mock>'
import importlib.machinery
_pd_mock.__spec__ = importlib.machinery.ModuleSpec('pandas', None)
sys.modules['pandas'] = _pd_mock
_pd_compat = types.ModuleType('pandas.compat')
_pd_compat.__dict__['PY312'] = True
sys.modules['pandas.compat'] = _pd_compat

_nw_deps = types.ModuleType('narwhals.dependencies')
_nw_deps.is_into_dataframe = lambda x: False
_nw_deps.is_into_series = lambda x: False
_nw_stable_v2_deps = types.ModuleType('narwhals.stable.v2.dependencies')
_nw_stable_v2_deps.is_into_dataframe = lambda x: False
_nw_stable_v2_deps.is_into_series = lambda x: False
sys.modules.setdefault('narwhals', types.ModuleType('narwhals'))
sys.modules['narwhals'].dependencies = _nw_deps
sys.modules.setdefault('narwhals.dependencies', _nw_deps)
sys.modules.setdefault('narwhals.stable', types.ModuleType('narwhals.stable'))
sys.modules.setdefault('narwhals.stable.v2', types.ModuleType('narwhals.stable.v2'))
sys.modules['narwhals.stable.v2'].dependencies = _nw_stable_v2_deps
sys.modules.setdefault('narwhals.stable.v2.dependencies', _nw_stable_v2_deps)
# ----------------------------------------

sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')

import numpy as np
import json
import time
from pathlib import Path

from canonical_7d import (
    load_7d_data, DataConfig, E1OfflineNN, STATE_NAMES_7D
)
from canonical_7d.residual_models import GPModel, NNEnsemble7D
from canonical_7d.config import STATE_DIM
from sklearn.metrics import roc_auc_score, average_precision_score


def main():
    t_start = time.time()
    print("=" * 60)
    print("AGENT D: Uncertainty Calibration & OOD Detection")
    print("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────────
    print("\n[1/5] Loading 7D data...")
    data_cfg = DataConfig(data_path="D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz")
    data = load_7d_data(data_cfg)
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    print(f"  Train samples: {len(data['train_states'])}")
    print(f"  Test samples:  {len(data['test_states'])}")
    print(f"  State names:   {STATE_NAMES_7D}")

    # ── Step 2: Train E1 model (GP + offline NN ensemble) ──────────
    cache_path = Path("D:/系统辨识作业/sindy_bicycle/continuation_stage_v8/raw_results/agent_d_cache.npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n[2/5] Training E1 model (GP + offline NN residual, 5 members)...")
    t_train = time.time()

    if cache_path.exists():
        print("  Loading cached predictions...")
        cached = np.load(cache_path, allow_pickle=True)
        all_pred_next = cached['all_pred_next']
        all_uncertainty = cached['all_uncertainty']
        true_next = cached['true_next']
        t_train_done = time.time()
        t_pred_end = t_train_done
        n_test = len(all_pred_next)
        print(f"  Loaded {n_test} cached predictions")
        skip_training = True
    else:
        model = E1OfflineNN(n_models=5, n_epochs=50, residual_scale=0.3)
        model.train(
            data['train_states'], data['train_actions'], data['train_deltas'],
            state_std, action_std, delta_std
        )
        t_train_done = time.time()
        print(f"  Training completed in {t_train_done - t_train:.1f}s")
        skip_training = False

    # ── Step 3: Collect predictions + uncertainties on test set ─────
    print("\n[3/5] Collecting predictions on test set...")
    test_states = data['test_states']
    test_actions = data['test_actions']
    test_deltas = data['test_deltas']

    if not skip_training:
        n_test = len(test_states)
        # true next states
        true_next = test_states + test_deltas  # (N, 7)

        # predict with uncertainty
        all_pred_next = np.empty((n_test, 7))
        all_uncertainty = np.empty((n_test, 7))

        print(f"  Running {n_test} predictions...")
        t_pred_start = time.time()
        for i in range(n_test):
            pred_next, unc = model.predict_with_uncertainty(test_states[i], test_actions[i])
            all_pred_next[i] = pred_next
            all_uncertainty[i] = unc
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t_pred_start
                print(f"    {i+1}/{n_test} done ({elapsed:.1f}s)")
        t_pred_end = time.time()
        print(f"  Prediction completed in {t_pred_end - t_pred_start:.1f}s")

        # Cache for next run
        np.savez(cache_path, all_pred_next=all_pred_next, all_uncertainty=all_uncertainty, true_next=true_next)
        print(f"  Cached predictions to {cache_path}")
    else:
        test_states = data['test_states']
        t_pred_start = t_train_done
        print(f"  Using {n_test} cached predictions")

    # ── Step 4: Uncertainty Calibration Metrics ─────────────────────
    print("\n[4/5] Computing calibration metrics...")

    # Absolute errors per state
    abs_errors = np.abs(all_pred_next - true_next)  # (N, 7)

    # 4a. Pearson correlation: raw ensemble std vs actual error (per state)
    from scipy.stats import pearsonr
    pearson_per_state = {}
    for i, name in enumerate(STATE_NAMES_7D):
        r, p = pearsonr(all_uncertainty[:, i], abs_errors[:, i])
        pearson_per_state[name] = {"r": float(r), "p_value": float(p)}
        print(f"  Pearson r ({name}): {r:.4f} (p={p:.2e})")

    # Overall Pearson (flatten)
    flat_unc = all_uncertainty.flatten()
    flat_err = abs_errors.flatten()
    r_overall, p_overall = pearsonr(flat_unc, flat_err)
    pearson_overall = {"r": float(r_overall), "p_value": float(p_overall)}
    print(f"  Pearson r (overall): {r_overall:.4f} (p={p_overall:.2e})")

    # 4b. Optimal scalar scale factor for 95% coverage
    target_coverage = 0.95
    # Use coarse search first, then refine around the target
    scale_coarse = np.linspace(0.1, 50.0, 1000)

    # Precompute vectorized coverage for all scales (all states must be covered)
    # abs_errors: (N, 7), all_uncertainty: (N, 7)
    # For each scale s: covered[i] = all(abs_errors[i] <= s * all_uncertainty[i])
    max_err_ratio = np.max(abs_errors / (all_uncertainty + 1e-12), axis=1)  # (N,)

    def coverage_at_scale(s):
        return (max_err_ratio <= s).mean()

    # Find coarse optimum
    best_gap = float('inf')
    best_scale_coarse = None
    best_coverage_coarse = None
    for s in scale_coarse:
        cov = coverage_at_scale(s)
        gap = abs(cov - target_coverage)
        if gap < best_gap:
            best_gap = gap
            best_scale_coarse = float(s)
            best_coverage_coarse = float(cov)

    # Refine around the best coarse scale
    if best_scale_coarse is not None:
        lo = max(0.01, best_scale_coarse - 1.0)
        hi = best_scale_coarse + 1.0
        scale_search = np.linspace(lo, hi, 1000)
    else:
        scale_search = scale_coarse

    best_scale = best_scale_coarse
    best_coverage = best_coverage_coarse
    for s in scale_search:
        cov = coverage_at_scale(s)
        gap = abs(cov - target_coverage)
        if gap < best_gap:
            best_gap = gap
            best_scale = float(s)
            best_coverage = float(cov)

    # For each candidate scale, check what fraction of errors fall within scale * uncertainty
    best_scale = None
    best_gap = float('inf')
    for s in scale_search:
        covered = np.all(abs_errors <= s * all_uncertainty, axis=1)
        coverage = covered.mean()
        gap = abs(coverage - target_coverage)
        if gap < best_gap:
            best_gap = gap
            best_scale = float(s)
            best_coverage = float(coverage)

    print(f"  Optimal scalar scale: {best_scale:.3f} -> coverage={best_coverage:.4f} (target={target_coverage})")

    # 4c. Per-state scale factors for 95% marginal coverage
    # Use wide search for per-state (theta_dot needs very large scale)
    scale_wide = np.concatenate([np.linspace(0.1, 50.0, 1000), np.linspace(50, 200, 500)])
    per_state_scales = {}
    # Vectorized: compute err_ratios for all states at once
    err_ratios = abs_errors / (all_uncertainty + 1e-12)  # (N, 7)
    for i, name in enumerate(STATE_NAMES_7D):
        err_ratio_i = err_ratios[:, i]
        # Vectorized coverage computation over all scales
        # coverage[s] = mean(err_ratio_i <= scale_wide[s])
        # Shape: (len(scale_wide), N) -> sum over axis=1 / N
        covered_matrix = (err_ratio_i[None, :] <= scale_wide[:, None])  # (S, N)
        coverages = covered_matrix.mean(axis=1)  # (S,)
        best_idx = np.argmin(np.abs(coverages - target_coverage))
        best_s = float(scale_wide[best_idx])
        best_cov = float(coverages[best_idx])
        per_state_scales[name] = {"scale": best_s, "coverage": best_cov}
        print(f"  Per-state scale ({name}): {best_s:.3f} -> coverage={best_cov:.4f}")

    # 4d. Expected Calibration Error (ECE)
    n_bins = 10
    ece_values = {}
    for i, name in enumerate(STATE_NAMES_7D):
        # Normalize errors by uncertainty to get "z-scores"
        z = abs_errors[:, i] / (all_uncertainty[:, i] + 1e-12)
        # Bin z-scores by theoretical probability
        # For a Gaussian, z <= 1.0 => ~68% coverage, z <= 1.645 => 90%, z <= 1.96 => 95%, z <= 2.576 => 99%
        # We define bins by the empirical fraction of z <= threshold
        bin_edges = np.linspace(0, np.percentile(z, 99), n_bins + 1)
        ece = 0.0
        total = len(z)
        for b in range(n_bins):
            mask = (z >= bin_edges[b]) & (z < bin_edges[b + 1])
            if mask.sum() == 0:
                continue
            # Empirical coverage: fraction of errors within this z-range
            # vs theoretical expected: fraction of data in this bin under Gaussian
            n_in_bin = mask.sum()
            frac_in_bin = n_in_bin / total
            # Mean z in this bin
            mean_z = z[mask].mean()
            # Empirical coverage fraction up to this bin edge
            empirical_coverage = (z <= bin_edges[b + 1]).mean()
            # Theoretical coverage for Gaussian: P(|Z| <= z) = erf(z/sqrt(2))
            from math import erf, sqrt
            theoretical_coverage = erf(bin_edges[b + 1] / sqrt(2))
            ece += abs(empirical_coverage - theoretical_coverage) * frac_in_bin
        ece_values[name] = float(ece)
        print(f"  ECE ({name}): {ece:.4f}")

    ece_overall = float(np.mean(list(ece_values.values())))
    print(f"  ECE (overall): {ece_overall:.4f}")

    # 4e. Coverage at standard confidence levels
    confidence_levels = [0.68, 0.90, 0.95, 0.99]
    coverage_at_levels = {}
    for level in confidence_levels:
        covered = np.all(abs_errors <= best_scale * all_uncertainty, axis=1)
        coverage_at_levels[str(level)] = float(covered.mean())
        print(f"  Coverage at scale={best_scale:.3f} (target {level}): {covered.mean():.4f}")

    # What scale gives each confidence level?
    # Vectorized: compute coverage for all scales at once
    scale_for_level_coarse = np.linspace(0.1, 200, 3000)
    covered_all = np.array([(max_err_ratio <= s).mean() for s in scale_for_level_coarse])  # (3000,)
    scale_for_level = {}
    for level in confidence_levels:
        best_idx = np.argmin(np.abs(covered_all - level))
        scale_for_level[str(level)] = float(scale_for_level_coarse[best_idx])
    print(f"  Scales for confidence levels: {scale_for_level}")

    # 4f. Calibration quality: max absolute coverage deviation (MACD)
    # Compute MACD: max |empirical coverage - nominal level| across standard levels
    from scipy.special import erfinv as _erfinv
    z_all = abs_errors / (all_uncertainty + 1e-12)
    z_max_all = z_all.max(axis=1)  # max z across states
    nominal_levels = np.linspace(0.1, 0.98, 20)
    macd_values = []
    for nom in nominal_levels:
        try:
            z_thresh = np.sqrt(2) * _erfinv(nom)
            emp = (z_max_all <= z_thresh).mean()
            macd_values.append(abs(emp - nom))
        except (ValueError, OverflowError):
            pass
    macd = float(max(macd_values)) if macd_values else float('nan')
    print(f"  MACD (max absolute coverage deviation): {macd:.4f}")

    # ── Step 5: OOD Detection ──────────────────────────────────────
    print("\n[5/5] Evaluating OOD detection...")

    # OOD definition: any state dimension has |s_i| > 2 * std_i
    state_std_global = np.std(data['all_states'], axis=0)
    state_std_global[state_std_global < 1e-10] = 1.0
    ood_mask = np.any(np.abs(test_states) > 2 * state_std_global, axis=1)
    n_ood = ood_mask.sum()
    n_total = len(ood_mask)
    ood_fraction = n_ood / n_total
    print(f"  OOD samples: {n_ood}/{n_total} ({ood_fraction:.4f})")

    # OOD score: use max uncertainty across states (higher = more uncertain)
    ood_scores = np.max(all_uncertainty / (state_std + 1e-10), axis=1)

    # AUROC
    try:
        auroc_val = float(roc_auc_score(ood_mask.astype(int), ood_scores))
        auroc = None if np.isnan(auroc_val) else auroc_val
        print(f"  AUROC: {auroc}")
    except Exception as e:
        auroc = None
        print(f"  AUROC: FAILED ({e})")

    # AUPRC
    try:
        auprc_val = float(average_precision_score(ood_mask.astype(int), ood_scores))
        auprc = None if np.isnan(auprc_val) else auprc_val
        print(f"  AUPRC: {auprc}")
    except Exception as e:
        auprc = None
        print(f"  AUPRC: FAILED ({e})")

    # Per-state OOD AUROC
    per_state_auroc = {}
    per_state_auprc = {}
    for i, name in enumerate(STATE_NAMES_7D):
        state_ood = np.abs(test_states[:, i]) > 2 * state_std_global[i]
        if state_ood.sum() == 0 or state_ood.sum() == len(state_ood):
            per_state_auroc[name] = None
            per_state_auprc[name] = None
        else:
            try:
                per_state_auroc[name] = float(roc_auc_score(
                    state_ood.astype(int),
                    all_uncertainty[:, i] / (state_std[i] + 1e-10)
                ))
            except:
                per_state_auroc[name] = None
            try:
                per_state_auprc[name] = float(average_precision_score(
                    state_ood.astype(int),
                    all_uncertainty[:, i] / (state_std[i] + 1e-10)
                ))
            except:
                per_state_auprc[name] = None
        print(f"  {name} OOD AUROC: {per_state_auroc[name]}, AUPRC: {per_state_auprc[name]}")

    # Entropy-based OOD score (using ensemble std as proxy)
    ood_scores_entropy = np.mean(all_uncertainty, axis=1)
    try:
        auroc_entropy_val = float(roc_auc_score(ood_mask.astype(int), ood_scores_entropy))
        auroc_entropy = None if np.isnan(auroc_entropy_val) else auroc_entropy_val
        auprc_entropy_val = float(average_precision_score(ood_mask.astype(int), ood_scores_entropy))
        auprc_entropy = None if np.isnan(auprc_entropy_val) else auprc_entropy_val
        print(f"  Mean-uncertainty AUROC: {auroc_entropy}, AUPRC: {auprc_entropy}")
    except:
        auroc_entropy = None
        auprc_entropy = None

    # ── Summary statistics ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Pearson r (overall):       {r_overall:.4f}")
    print(f"  Optimal scalar scale:      {best_scale:.3f} (coverage={best_coverage:.4f})")
    print(f"  ECE (overall):             {ece_overall:.4f}")
    print(f"  MACD:                      {macd:.4f}")
    print(f"  OOD fraction:              {ood_fraction:.4f}")
    print(f"  OOD AUROC (max-unc):       {auroc}")
    print(f"  OOD AUPRC (max-unc):       {auprc}")
    print(f"  OOD AUROC (mean-unc):      {auroc_entropy}")
    print(f"  Mean absolute error:       {abs_errors.mean():.6f}")
    print(f"  Mean uncertainty:          {all_uncertainty.mean():.6f}")

    t_end = time.time()
    print(f"\nTotal time: {t_end - t_start:.1f}s")

    # ── Save results ────────────────────────────────────────────────
    results = {
        "agent": "D",
        "task": "uncertainty_calibration_ood_detection",
        "model": "E1 (GP + offline NN residual)",
        "n_test_samples": int(n_test),
        "n_models_ensemble": 5,
        "training_time_s": float(t_train_done - t_train),
        "prediction_time_s": float(t_pred_end - t_pred_start),
        "total_time_s": float(t_end - t_start),

        "calibration": {
            "pearson_per_state": pearson_per_state,
            "pearson_overall": pearson_overall,
            "optimal_scalar_scale": best_scale,
            "optimal_scalar_coverage": best_coverage,
            "per_state_scales": per_state_scales,
            "ece_per_state": ece_values,
            "ece_overall": ece_overall,
            "macd": macd,
            "target_coverage": target_coverage,
            "scales_for_confidence_levels": scale_for_level,
        },

        "ood_detection": {
            "ood_definition": "any |s_i| > 2*std_i",
            "n_ood": int(n_ood),
            "n_total": int(n_total),
            "ood_fraction": float(ood_fraction),
            "auroc_max_uncertainty": auroc,
            "auprc_max_uncertainty": auprc,
            "auroc_mean_uncertainty": auroc_entropy,
            "auprc_mean_uncertainty": auprc_entropy,
            "per_state_auroc": per_state_auroc,
            "per_state_auprc": per_state_auprc,
        },

        "per_state_summary": {},
    }

    for i, name in enumerate(STATE_NAMES_7D):
        results["per_state_summary"][name] = {
            "pearson_r": pearson_per_state[name]["r"],
            "pearson_p": pearson_per_state[name]["p_value"],
            "optimal_scale": per_state_scales[name]["scale"],
            "coverage": per_state_scales[name]["coverage"],
            "ece": ece_values[name],
            "mean_abs_error": float(abs_errors[:, i].mean()),
            "mean_uncertainty": float(all_uncertainty[:, i].mean()),
        }

    out_dir = Path("D:/系统辨识作业/sindy_bicycle/continuation_stage_v8/raw_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "AGENT_D_UNCERTAINTY.json"

    # Custom JSON encoder for NaN/Inf
    class NanEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, float):
                if np.isnan(obj):
                    return None
                if np.isinf(obj):
                    return None
            return super().default(obj)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NanEncoder)

    print(f"\nResults saved to: {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
