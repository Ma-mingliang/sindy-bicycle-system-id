"""Agent F: MPC/MPPI Planning Validation for 7D World Model.

Validates that the GP model correctly ranks candidate action sequences
by cost, which is the core requirement for model-based planning (MPC/MPPI).

Approach:
- Use nearest-neighbor lookup on training data as ground-truth dynamics oracle
- For each test starting state, generate 50 random candidate action sequences
- Rollout each candidate with GP model -> predicted cost
- Rollout each candidate with real-data nearest-neighbor -> true cost
- Compare cost rankings: Spearman rho, Kendall tau, top-k overlap
"""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')

import numpy as np
import json
import time
from scipy.stats import spearmanr, kendalltau
from scipy.spatial import cKDTree
from canonical_7d import (
    load_7d_data, DataConfig, GPModel, TrainingConfig,
    compute_cost, STATE_DIM, STATE_NAMES_7D,
)
from canonical_7d.data_loader import get_test_segments


# ---------------------------------------------------------------------------
# Cost weights (from task spec)
# ---------------------------------------------------------------------------
COST_WEIGHTS = {
    'e_y': 100.0, 'e_psi': 100.0, 'v': 1.0,
    'theta': 10.0, 'theta_dot': 1.0,
    'delta': 10.0, 'delta_dot': 1.0, 'u': 0.1
}

N_CANDIDATES = 50
HORIZONS = [10, 20]
N_START_STATES = 10  # number of test segments to use
K_NN = 5  # k for nearest-neighbor lookup


def build_nn_oracle(train_states, train_actions, train_deltas, k=5):
    """Build a k-nearest-neighbor oracle for real dynamics.

    Given (state, action), finds the k nearest (state, action) pairs
    in the training data and returns the average delta as the
    ground-truth next-state transition.
    """
    # Build feature matrix: [normalized_state, normalized_action]
    s_std = np.std(train_states, axis=0)
    s_std[s_std < 1e-10] = 1.0
    a_std = np.std(train_actions)
    if a_std < 1e-10:
        a_std = 1.0

    X_nn = np.column_stack([
        train_states / s_std,
        train_actions.reshape(-1, 1) / a_std
    ])
    tree = cKDTree(X_nn)

    return {
        'tree': tree,
        'deltas': train_deltas,
        's_std': s_std,
        'a_std': a_std,
        'k': k,
    }


def nn_predict(oracle, state, action):
    """Predict next state using nearest-neighbor oracle."""
    s_std = oracle['s_std']
    a_std = oracle['a_std']
    x = np.concatenate([state / s_std, [action / a_std]]).reshape(1, -1)
    dists, idxs = oracle['tree'].query(x, k=oracle['k'])
    # Weighted average of deltas by inverse distance
    dists = dists.flatten()
    idxs = idxs.flatten()
    weights = 1.0 / (dists + 1e-10)
    weights /= weights.sum()
    delta = np.average(oracle['deltas'][idxs], axis=0, weights=weights)
    return state + delta


def rollout_gp(model, s0, actions):
    """Rollout with GP model, return trajectory states."""
    trajectory = [s0.copy()]
    s = s0.copy()
    for a in actions:
        s = model.predict(s, a)
        trajectory.append(s.copy())
    return np.array(trajectory)


def rollout_nn(oracle, s0, actions):
    """Rollout with nearest-neighbor oracle, return trajectory states."""
    trajectory = [s0.copy()]
    s = s0.copy()
    for a in actions:
        s = nn_predict(oracle, s, a)
        trajectory.append(s.copy())
    return np.array(trajectory)


def compute_candidate_costs(model, oracle, s0, action_sequences):
    """Compute GP-predicted and NN-true costs for a set of candidate sequences.

    Returns:
        gp_costs: array of GP-predicted costs
        nn_costs: array of NN-true costs
    """
    gp_costs = []
    nn_costs = []

    for actions in action_sequences:
        # GP rollout
        traj_gp = rollout_gp(model, s0, actions)
        gp_cost = compute_cost(traj_gp, actions, COST_WEIGHTS)
        gp_costs.append(gp_cost)

        # NN (real-data) rollout
        traj_nn = rollout_nn(oracle, s0, actions)
        nn_cost = compute_cost(traj_nn, actions, COST_WEIGHTS)
        nn_costs.append(nn_cost)

    return np.array(gp_costs), np.array(nn_costs)


def ranking_metrics(true_costs, pred_costs, top_k=5):
    """Compute ranking similarity metrics."""
    n = len(true_costs)
    if n < 3:
        return {'spearman_rho': 0.0, 'kendall_tau': 0.0,
                'top_k_overlap': 0.0, 'mae_ratio': 0.0}

    # Spearman rank correlation
    spearman_rho, spearman_p = spearmanr(true_costs, pred_costs)

    # Kendall tau
    kendall_tau, kendall_p = kendalltau(true_costs, pred_costs)

    # Top-k overlap
    k = min(top_k, n)
    true_topk = set(np.argsort(true_costs)[:k])
    pred_topk = set(np.argsort(pred_costs)[:k])
    overlap = len(true_topk & pred_topk) / k

    # Relative cost prediction error
    # (mean |predicted - true| / mean |true|)
    mean_true = np.mean(np.abs(true_costs))
    if mean_true > 1e-10:
        mae_ratio = np.mean(np.abs(pred_costs - true_costs)) / mean_true
    else:
        mae_ratio = float('nan')

    return {
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(spearman_p),
        'kendall_tau': float(kendall_tau),
        'kendall_p': float(kendall_p),
        'top_k_overlap': float(overlap),
        'top_k': k,
        'mae_ratio': float(mae_ratio),
    }


def run_planning_validation():
    """Main validation loop."""
    t_start = time.time()
    print("=" * 70)
    print("AGENT F: MPC/MPPI PLANNING VALIDATION")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("\n[1] Loading 7D data...")
    cfg = DataConfig(data_path='D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz')
    data = load_7d_data(cfg)
    print(f"    Train: {len(data['train_states'])} samples, "
          f"Test: {len(data['test_states'])} samples")
    print(f"    Episodes: {data['n_episodes']}")

    # -----------------------------------------------------------------------
    # 2. Train GP model
    # -----------------------------------------------------------------------
    print("\n[2] Training GP model...")
    t0 = time.time()
    train_cfg = TrainingConfig()
    gp = GPModel(max_samples=train_cfg.gp_max_samples,
                 n_restarts=train_cfg.gp_n_restarts)
    gp.train(
        data['train_states'], data['train_actions'], data['train_deltas'],
        data['state_std'], data['action_std'], data['delta_std']
    )
    gp_train_time = time.time() - t0
    print(f"    GP trained in {gp_train_time:.1f}s")

    # -----------------------------------------------------------------------
    # 3. Build nearest-neighbor oracle (ground truth dynamics)
    # -----------------------------------------------------------------------
    print("\n[3] Building NN oracle for ground-truth dynamics...")
    oracle = build_nn_oracle(
        data['train_states'], data['train_actions'], data['train_deltas'],
        k=K_NN
    )
    print(f"    Oracle built with {len(data['train_states'])} transitions, k={K_NN}")

    # -----------------------------------------------------------------------
    # 4. Get test segments
    # -----------------------------------------------------------------------
    print("\n[4] Extracting test segments...")
    segments = get_test_segments(data, n_segments=N_START_STATES + 5,
                                 segment_length=max(HORIZONS) + 10, seed=42)
    print(f"    Got {len(segments)} test segments")

    # -----------------------------------------------------------------------
    # 5. MPC planning validation
    # -----------------------------------------------------------------------
    print("\n[5] Running MPC planning validation...")
    rng = np.random.RandomState(123)
    results = {}

    for horizon in HORIZONS:
        print(f"\n  --- Horizon H = {horizon} ---")
        horizon_results = {
            'horizon': horizon,
            'n_candidates': N_CANDIDATES,
            'n_start_states': 0,
            'ranking_metrics_per_state': [],
            'aggregated_ranking': {},
            'cost_stats': {},
        }

        all_spearman = []
        all_kendall = []
        all_topk = []
        all_mae_ratio = []
        all_gp_costs = []
        all_nn_costs = []

        n_used = 0
        for seg_idx, seg in enumerate(segments):
            if n_used >= N_START_STATES:
                break
            if len(seg['actions']) < horizon:
                continue

            s0 = seg['states'][0].copy()
            actions_actual = seg['actions'][:horizon]

            # Generate N_CANDIDATES random action sequences
            # Range [-0.1, 0.1] matching real action range
            action_seqs = []
            for _ in range(N_CANDIDATES):
                a = rng.uniform(-0.1, 0.1, size=horizon)
                action_seqs.append(a)

            # Compute costs: GP-predicted and NN-true
            gp_costs, nn_costs = compute_candidate_costs(
                gp, oracle, s0, action_seqs
            )

            # Compute ranking metrics for this starting state
            metrics = ranking_metrics(nn_costs, gp_costs, top_k=5)

            all_spearman.append(metrics['spearman_rho'])
            all_kendall.append(metrics['kendall_tau'])
            all_topk.append(metrics['top_k_overlap'])
            all_mae_ratio.append(metrics['mae_ratio'])
            all_gp_costs.extend(gp_costs.tolist())
            all_nn_costs.extend(nn_costs.tolist())

            horizon_results['ranking_metrics_per_state'].append({
                'segment_idx': seg_idx,
                'spearman_rho': metrics['spearman_rho'],
                'kendall_tau': metrics['kendall_tau'],
                'top_k_overlap': metrics['top_k_overlap'],
                'mae_ratio': metrics['mae_ratio'],
                'mean_gp_cost': float(np.mean(gp_costs)),
                'mean_nn_cost': float(np.mean(nn_costs)),
            })

            n_used += 1
            if n_used % 5 == 0:
                print(f"    Processed {n_used}/{N_START_STATES} starting states "
                      f"(latest Spearman={metrics['spearman_rho']:.3f})")

        horizon_results['n_start_states'] = n_used

        # Aggregate ranking metrics
        horizon_results['aggregated_ranking'] = {
            'spearman_rho_mean': float(np.mean(all_spearman)),
            'spearman_rho_std': float(np.std(all_spearman)),
            'kendall_tau_mean': float(np.mean(all_kendall)),
            'kendall_tau_std': float(np.std(all_kendall)),
            'top_k_overlap_mean': float(np.mean(all_topk)),
            'top_k_overlap_std': float(np.std(all_topk)),
            'mae_ratio_mean': float(np.mean(all_mae_ratio)),
            'mae_ratio_std': float(np.std(all_mae_ratio)),
        }

        # Cost prediction statistics
        all_gp = np.array(all_gp_costs)
        all_nn = np.array(all_nn_costs)
        valid = (np.isfinite(all_gp) & np.isfinite(all_nn))
        if valid.sum() > 0:
            gp_v = all_gp[valid]
            nn_v = all_nn[valid]
            horizon_results['cost_stats'] = {
                'gp_cost_mean': float(np.mean(gp_v)),
                'gp_cost_std': float(np.std(gp_v)),
                'nn_cost_mean': float(np.mean(nn_v)),
                'nn_cost_std': float(np.std(nn_v)),
                'cost_correlation': float(np.corrcoef(gp_v, nn_v)[0, 1])
                    if len(gp_v) > 1 else 0.0,
                'relative_error': float(np.mean(np.abs(gp_v - nn_v))
                    / (np.mean(np.abs(nn_v)) + 1e-10)),
            }

        results[f'horizon_{horizon}'] = horizon_results

        # Print summary
        ar = horizon_results['aggregated_ranking']
        cs = horizon_results['cost_stats']
        print(f"\n    Summary for H={horizon}:")
        print(f"      Spearman rho:  {ar['spearman_rho_mean']:.4f} "
              f"+/- {ar['spearman_rho_std']:.4f}")
        print(f"      Kendall tau:   {ar['kendall_tau_mean']:.4f} "
              f"+/- {ar['kendall_tau_std']:.4f}")
        print(f"      Top-5 overlap: {ar['top_k_overlap_mean']:.4f} "
              f"+/- {ar['top_k_overlap_std']:.4f}")
        print(f"      Cost MAE ratio:{ar['mae_ratio_mean']:.4f} "
              f"+/- {ar['mae_ratio_std']:.4f}")
        if cs:
            print(f"      Cost correlation: {cs['cost_correlation']:.4f}")
            print(f"      Relative cost error: {cs['relative_error']:.4f}")

    # -----------------------------------------------------------------------
    # 6. Additional validation: actual trajectory cost prediction
    # -----------------------------------------------------------------------
    print("\n[6] Validating actual trajectory cost prediction...")
    actual_traj_results = []
    for seg in segments[:N_START_STATES]:
        actual_actions = seg['actions'][:min(max(HORIZONS), len(seg['actions']))]
        if len(actual_actions) < min(HORIZONS):
            continue
        actual_states = seg['states'][:len(actual_actions) + 1]

        # True cost from real data
        true_cost = compute_cost(actual_states, actual_actions, COST_WEIGHTS)

        # GP-predicted cost
        traj_gp = rollout_gp(gp, actual_states[0], actual_actions)
        gp_cost = compute_cost(traj_gp, actual_actions, COST_WEIGHTS)

        actual_traj_results.append({
            'true_cost': float(true_cost),
            'gp_cost': float(gp_cost),
            'relative_error': float(abs(gp_cost - true_cost)
                / (abs(true_cost) + 1e-10)),
        })

    if actual_traj_results:
        true_costs_arr = np.array([r['true_cost'] for r in actual_traj_results])
        gp_costs_arr = np.array([r['gp_cost'] for r in actual_traj_results])
        rel_errors = np.array([r['relative_error'] for r in actual_traj_results])
        results['actual_trajectory_validation'] = {
            'n_segments': len(actual_traj_results),
            'mean_relative_error': float(np.mean(rel_errors)),
            'median_relative_error': float(np.median(rel_errors)),
            'max_relative_error': float(np.max(rel_errors)),
            'cost_correlation': float(np.corrcoef(true_costs_arr, gp_costs_arr)[0, 1])
                if len(true_costs_arr) > 1 else 0.0,
            'details': actual_traj_results,
        }
        print(f"    Mean relative cost error: {np.mean(rel_errors):.4f}")
        print(f"    Cost correlation: {results['actual_trajectory_validation']['cost_correlation']:.4f}")

    # -----------------------------------------------------------------------
    # 7. Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start
    summary = {
        'agent': 'F',
        'task': 'MPC/MPPI Planning Validation',
        'elapsed_seconds': round(elapsed, 1),
        'config': {
            'n_candidates': N_CANDIDATES,
            'horizons': HORIZONS,
            'n_start_states': N_START_STATES,
            'cost_weights': COST_WEIGHTS,
            'k_nn': K_NN,
            'gp_max_samples': train_cfg.gp_max_samples,
        },
    }
    results['summary'] = summary

    # Print overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    for h in HORIZONS:
        key = f'horizon_{h}'
        ar = results[key]['aggregated_ranking']
        verdict = "PASS" if ar['spearman_rho_mean'] > 0.3 else "FAIL"
        print(f"  H={h:3d}: Spearman={ar['spearman_rho_mean']:+.4f} "
              f"Kendall={ar['kendall_tau_mean']:+.4f} "
              f"Top5={ar['top_k_overlap_mean']:.2f} "
              f"MAE_ratio={ar['mae_ratio_mean']:.4f}  [{verdict}]")

    if 'actual_trajectory_validation' in results:
        atv = results['actual_trajectory_validation']
        print(f"\n  Actual trajectory cost prediction:")
        print(f"    Correlation: {atv['cost_correlation']:.4f}")
        print(f"    Mean relative error: {atv['mean_relative_error']:.4f}")

    print(f"\n  Total time: {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # 8. Save results
    # -----------------------------------------------------------------------
    out_path = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8/raw_results/AGENT_F_PLANNING.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_path}")

    return results


if __name__ == '__main__':
    run_planning_validation()
