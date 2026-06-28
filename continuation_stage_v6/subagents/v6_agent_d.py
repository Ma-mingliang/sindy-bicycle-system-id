"""V6 Agent D: MPC/MPPI Cost & Ranking Prediction Quality.

Tests whether learned models can correctly predict:
1. The COST of candidate action sequences
2. The RANKING of candidate sequences by cost

Models tested: GP, RDE-L (local residual), RDE-M (hybrid DAgger)
Horizons tested: [10, 20, 50] (short-horizon MPC relevance)
"""
import sys, os, json, time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

sys.path.insert(0, str(Path(__file__).parent.parent))
from canonical_4d.config import DynamicsConfig, TrainingConfig, PlanConfig
from canonical_4d.dynamics import BicycleDynamics
from canonical_4d.models import GPModel
from canonical_4d.residual_models import RDELocal, RDEHybrid
from canonical_4d.runner import generate_training_data
from canonical_4d.metrics import compute_cost

OUTPUT = Path(__file__).parent
OUTPUT.mkdir(exist_ok=True)

# Cost weights from PlanConfig
COST_WEIGHTS = {
    'phi': 100.0, 'delta': 100.0,
    'phi_dot': 10.0, 'delta_dot': 1.0, 'u': 0.1
}


# ────────────────────────────────────────────────────────────
# Rollout helpers
# ────────────────────────────────────────────────────────────

def rollout_true(s0: np.ndarray, actions: np.ndarray, dynamics: BicycleDynamics):
    """Roll out with TRUE dynamics. Returns (states, diverged_step)."""
    states = [s0.copy()]
    s = s0.copy()
    for k in range(len(actions)):
        s_next = dynamics.step(s, actions[k])
        if dynamics.is_diverged(s_next):
            return np.array(states), k
        states.append(s_next.copy())
        s = s_next
    return np.array(states), len(actions)


def rollout_model(model, s0: np.ndarray, actions: np.ndarray):
    """Roll out with a learned model. Returns (states, diverged_step)."""
    states = [s0.copy()]
    s = s0.copy()
    for k in range(len(actions)):
        try:
            s_next = model.predict(s, actions[k])
        except Exception:
            return np.array(states), k
        if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
            return np.array(states), k
        if abs(s_next[0]) > np.pi / 3 or np.any(np.abs(s_next) > 100):
            return np.array(states), k
        states.append(s_next.copy())
        s = s_next
    return np.array(states), len(actions)


def compute_sequence_cost(states: np.ndarray, actions: np.ndarray, n_steps: int) -> float:
    """Compute cost for the first n_steps of a rollout."""
    n_use = min(n_steps, len(states) - 1, len(actions))
    if n_use <= 0:
        return 1e12  # diverged immediately -> huge cost
    return compute_cost(states[:n_use + 1], actions[:n_use], COST_WEIGHTS)


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("V6 AGENT D: MPC/MPPI Cost & Ranking Prediction Quality")
    print("=" * 80)
    t_start = time.time()

    horizons = [10, 20, 50]
    n_candidates = 50
    seq_len = 20  # max sequence length (we truncate to horizon)
    action_lo, action_hi = -30.0, 30.0

    # ── Step 1: Train models ──────────────────────────────
    print("\n[Step 1] Generating training data & training models...")
    dyn = BicycleDynamics()
    data = generate_training_data(n_samples=3000, seed=42, dynamics=dyn)
    states, actions, deltas, state_std, action_std, delta_std = data

    # GP (use fewer samples to avoid memory issues)
    t0 = time.time()
    gp = GPModel(max_samples=800, n_restarts=2)
    gp.train(states, actions, deltas, state_std, action_std, delta_std)
    print(f"  GP trained in {time.time() - t0:.1f}s")

    # RDE-L (local residual, 0 DAgger rounds)
    t0 = time.time()
    rde_l = RDELocal(n_models=3, n_epochs=30, residual_scale=0.3)
    rde_l.train(states, actions, deltas, state_std, action_std, delta_std)
    print(f"  RDE-L trained in {time.time() - t0:.1f}s")

    # RDE-M (hybrid DAgger, 2 rounds)
    t0 = time.time()
    rde_m = RDEHybrid(
        n_models=3, n_epochs=30, residual_scale=0.3,
        dagger_rounds=2, n_segments=3, segment_length=300
    )
    rde_m.train(states, actions, deltas, state_std, action_std, delta_std,
                dynamics=dyn, make_tau_func=dyn.generate_lqr_tau)
    print(f"  RDE-M trained in {time.time() - t0:.1f}s")

    models = {'gp': gp, 'rde_l': rde_l, 'rde_m': rde_m}

    # ── Step 2: Generate candidate sequences ──────────────
    print(f"\n[Step 2] Generating {n_candidates} candidate action sequences "
          f"(len={seq_len}, range=[{action_lo}, {action_hi}])...")
    rng_seq = np.random.RandomState(123)
    candidates = rng_seq.uniform(action_lo, action_hi, size=(n_candidates, seq_len))

    # Also add a few "expert-like" sequences (LQR-like small actions)
    for i in range(min(5, n_candidates)):
        candidates[i] = rng_seq.uniform(-5, 5, seq_len)

    # ── Step 3: Compute true costs ────────────────────────
    print("\n[Step 3] Computing TRUE costs for each candidate...")
    # Use a fixed initial state for all candidates (relevance: MPC starts from current state)
    s0 = np.array([0.1, 0.0, 0.0, 0.0])

    true_costs = np.full(n_candidates, 1e12)
    true_states_cache = {}  # cache true rollout states for model comparison

    for c_idx in range(n_candidates):
        actions_c = candidates[c_idx]
        states_true, n_valid = rollout_true(s0, actions_c, dyn)
        true_states_cache[c_idx] = (states_true, n_valid)
        if n_valid > 0:
            true_costs[c_idx] = compute_sequence_cost(states_true, actions_c, n_valid)

    # ── Step 4: Compute model costs at each horizon ───────
    print("\n[Step 4] Computing model-predicted costs...")

    all_results = {}  # horizon -> model_name -> metrics dict

    for horizon in horizons:
        print(f"\n  === Horizon H = {horizon} ===")
        # Filter candidates that survive at least `horizon` steps in true dynamics
        valid_mask = np.array([
            true_states_cache[i][1] >= horizon for i in range(n_candidates)
        ])
        n_valid_cands = int(np.sum(valid_mask))
        print(f"  Candidates surviving {horizon} steps (true): {n_valid_cands}/{n_candidates}")

        if n_valid_cands < 5:
            print(f"  SKIP: too few valid candidates for horizon {horizon}")
            all_results[horizon] = {'skip_reason': 'insufficient_valid_candidates'}
            continue

        # True costs at this horizon
        true_costs_h = np.array([
            compute_sequence_cost(true_states_cache[i][0], candidates[i], horizon)
            for i in range(n_candidates)
        ])

        # True ranking (lower cost = better)
        true_rank = np.argsort(np.argsort(true_costs_h))  # rank 0 = best

        for model_name, model in models.items():
            t0 = time.time()
            model_costs_h = np.full(n_candidates, 1e12)

            for c_idx in range(n_candidates):
                actions_c = candidates[c_idx][:horizon]
                states_model, n_valid = rollout_model(model, s0, actions_c)
                if n_valid > 0:
                    model_costs_h[c_idx] = compute_sequence_cost(states_model, actions_c, n_valid)

            # Model ranking (lower cost = better)
            model_rank = np.argsort(np.argsort(model_costs_h))

            # ── Metrics ──
            # 1. Spearman rank correlation
            spearman_corr, spearman_p = spearmanr(true_rank, model_rank)

            # 2. Kendall tau correlation
            kendall_corr, kendall_p = kendalltau(true_rank, model_rank)

            # 3. Cost prediction error (relative)
            # Only for candidates where both costs are finite
            both_finite = np.isfinite(true_costs_h) & np.isfinite(model_costs_h) & (true_costs_h > 1e-6)
            if np.sum(both_finite) > 0:
                rel_errors = np.abs(model_costs_h[both_finite] - true_costs_h[both_finite]) / true_costs_h[both_finite]
                mean_rel_error = float(np.mean(rel_errors))
                median_rel_error = float(np.median(rel_errors))
                max_rel_error = float(np.max(rel_errors))
            else:
                mean_rel_error = float('nan')
                median_rel_error = float('nan')
                max_rel_error = float('nan')

            # 4. Is TRUE best in model's top-5?
            true_best_idx = int(np.argmin(true_costs_h))
            model_top5 = set(np.argsort(model_costs_h)[:5])
            true_best_in_top5 = true_best_idx in model_top5

            # 5. Top-5 overlap (Jaccard of model top-5 vs true top-5)
            true_top5 = set(np.argsort(true_costs_h)[:5])
            top5_overlap = len(model_top5 & true_top5)
            top5_jaccard = top5_overlap / 5.0

            elapsed = time.time() - t0

            metrics = {
                'spearman_rho': round(spearman_corr, 4),
                'spearman_p': round(spearman_p, 6),
                'kendall_tau': round(kendall_corr, 4),
                'kendall_p': round(kendall_p, 6),
                'mean_rel_error': round(mean_rel_error, 4),
                'median_rel_error': round(median_rel_error, 4),
                'max_rel_error': round(max_rel_error, 4),
                'true_best_in_top5': true_best_in_top5,
                'top5_overlap': top5_overlap,
                'top5_jaccard': round(top5_jaccard, 4),
                'n_valid_candidates': n_valid_cands,
                'compute_time_s': round(elapsed, 2),
            }
            all_results.setdefault(horizon, {})[model_name] = metrics

            print(f"    {model_name:6s}: Spearman={spearman_corr:+.3f}, Kendall={kendall_corr:+.3f}, "
                  f"RelErr={mean_rel_error:.3f}, Top5={true_best_in_top5}, "
                  f"Overlap={top5_overlap}/5  ({elapsed:.1f}s)")

    # ── Step 5: Summary ───────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Best model per metric per horizon
    summary_rows = []
    for h in horizons:
        if h not in all_results or 'skip_reason' in all_results.get(h, {}):
            continue
        hm = all_results[h]
        for metric_key in ['spearman_rho', 'kendall_tau', 'mean_rel_error', 'top5_jaccard']:
            if metric_key in ('mean_rel_error',):
                # Lower is better
                best_model = min(hm.keys(), key=lambda m: hm[m].get(metric_key, 999))
            else:
                # Higher is better
                best_model = max(hm.keys(), key=lambda m: hm[m].get(metric_key, -999))
            best_val = hm[best_model].get(metric_key, None)
            row = {
                'horizon': h, 'metric': metric_key,
                'best_model': best_model, 'best_value': best_val,
                'all_values': {m: hm[m].get(metric_key) for m in hm}
            }
            summary_rows.append(row)
            print(f"  H={h:3d} | {metric_key:18s} | best={best_model:6s} ({best_val})")

    # Overall verdict
    print("\n--- Overall Verdict ---")
    for model_name in models:
        # Average Spearman across horizons
        spearmans = []
        kendalls = []
        for h in horizons:
            if h in all_results and model_name in all_results.get(h, {}):
                spearmans.append(all_results[h][model_name].get('spearman_rho', 0))
                kendalls.append(all_results[h][model_name].get('kendall_tau', 0))
        avg_spearman = np.mean(spearmans) if spearmans else 0
        avg_kendall = np.mean(kendalls) if kendalls else 0
        print(f"  {model_name:6s}: avg Spearman={avg_spearman:+.3f}, avg Kendall={avg_kendall:+.3f}")

    # ── Save ──────────────────────────────────────────────
    output = {
        'config': {
            'n_candidates': n_candidates,
            'seq_len': seq_len,
            'action_range': [action_lo, action_hi],
            'horizons': horizons,
            'cost_weights': COST_WEIGHTS,
            's0': s0.tolist(),
            'training_samples': 5000,
            'training_seed': 42,
        },
        'results_by_horizon': {
            str(h): all_results.get(h, {}) for h in horizons
        },
        'summary': summary_rows,
        'elapsed_s': round(time.time() - t_start, 1),
    }

    out_path = OUTPUT / "AGENT_D_RESULTS.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    elapsed = time.time() - t_start
    print(f"\nAgent D done in {elapsed:.1f}s")
    return output


if __name__ == "__main__":
    main()
