"""Stage 1: LQR Q/R re-tuning sweep.

Tests different Q/R configurations to find safer LQR gains.
Prioritizes safety (ET, theta_rms, theta_max) over return.
"""
import sys, os, json, csv
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tracking_env import PathTrackingEnv, _compute_lqr_gains


def run_episode(env, seed, max_steps=1500):
    obs, _ = env.reset(seed=seed)
    total_r = 0
    early_term = False
    ey_list, epsi_list, theta_list = [], [], []

    for step in range(max_steps):
        action = np.array([0.0])
        obs, r, t, tr, info = env.step(action)
        total_r += r

        raw = info.get('raw_state', np.zeros(8))
        ey_list.append(abs(raw[0]))
        epsi_list.append(abs(raw[1]))
        theta_list.append(abs(raw[3]))

        if t:
            early_term = True
        if t or tr:
            break

    n = step + 1
    return {
        'return': total_r, 'length': n, 'early_term': early_term,
        'ey_rms': float(np.sqrt(np.mean(np.array(ey_list)**2))),
        'epsi_rms': float(np.sqrt(np.mean(np.array(epsi_list)**2))),
        'theta_rms': float(np.sqrt(np.mean(np.array(theta_list)**2))),
        'theta_max': float(np.max(np.array(theta_list))) if theta_list else 0.0,
    }


def evaluate_config(Q_diag, R_val, label, n_seeds=15):
    env = PathTrackingEnv(max_episode_steps=1500, residual_injection='stanley_ref',
                          lqr_Q=Q_diag, lqr_R=R_val)
    results = []
    for s in range(n_seeds):
        r = run_episode(env, seed=s)
        results.append(r)
    env.close()

    ret = np.mean([r['return'] for r in results])
    et = sum(r['early_term'] for r in results)
    ey = np.mean([r['ey_rms'] for r in results])
    ep = np.mean([r['epsi_rms'] for r in results])
    th = np.mean([r['theta_rms'] for r in results])
    th_max = max(r['theta_max'] for r in results)
    avg_len = np.mean([r['length'] for r in results])

    return {
        'label': label, 'Q_diag': Q_diag, 'R_val': R_val,
        'return': float(ret), 'ET': int(et), 'ET_rate': float(et / n_seeds),
        'ey_rms': float(ey), 'epsi_rms': float(ep),
        'theta_rms': float(th), 'theta_max': float(th_max),
        'avg_length': float(avg_len),
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 110)
    print("  Stage 1: LQR Q/R Re-tuning Sweep")
    print(f"  Timestamp: {timestamp}")
    print("=" * 110)

    # Current baseline: Q=[100,10,10,1], R=1.0
    # Goal: prioritize safety → reduce ET, reduce theta_rms

    configs = [
        # Baseline
        {'label': 'baseline',       'Q': [100, 10, 10, 1],  'R': 1.0},
        # Higher theta penalty
        {'label': 'Q200',           'Q': [200, 10, 10, 1],  'R': 1.0},
        {'label': 'Q500',           'Q': [500, 10, 10, 1],  'R': 1.0},
        {'label': 'Q1000',          'Q': [1000, 10, 10, 1], 'R': 1.0},
        # Higher theta_dot penalty (smoother)
        {'label': 'Q100_D20',       'Q': [100, 20, 10, 1],  'R': 1.0},
        {'label': 'Q100_D50',       'Q': [100, 50, 10, 1],  'R': 1.0},
        # Lower R (more aggressive control)
        {'label': 'R05',            'Q': [100, 10, 10, 1],  'R': 0.5},
        {'label': 'R02',            'Q': [100, 10, 10, 1],  'R': 0.2},
        # Combined: high theta + low R
        {'label': 'Q200_R05',       'Q': [200, 10, 10, 1],  'R': 0.5},
        {'label': 'Q500_R05',       'Q': [500, 10, 10, 1],  'R': 0.5},
        # Balanced: moderate increase across the board
        {'label': 'Q200_D20_R05',   'Q': [200, 20, 10, 1],  'R': 0.5},
        {'label': 'Q200_D20_R02',   'Q': [200, 20, 10, 1],  'R': 0.2},
        # Aggressive safety
        {'label': 'Q500_D50_R02',   'Q': [500, 50, 10, 1],  'R': 0.2},
        # Reduce delta penalty (allow more steering)
        {'label': 'Q100_D10_D5',    'Q': [100, 10, 5, 1],   'R': 1.0},
        {'label': 'Q200_D20_D5_R05','Q': [200, 20, 5, 1],   'R': 0.5},
    ]

    all_results = []
    n_seeds = 15

    print(f"\n  Testing {len(configs)} configs with {n_seeds} seeds each\n")
    print(f"  {'Config':<25} {'K_lqr':>30} {'ret':>8} {'ET':>5} {'ey':>7} {'ep':>7} {'th':>7} {'th_max':>7}")
    print(f"  {'-'*25} {'-'*30} {'-'*8} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for cfg in configs:
        r = evaluate_config(cfg['Q'], cfg['R'], cfg['label'], n_seeds)
        all_results.append(r)

        # Print K gains for reference
        K = _compute_lqr_gains(Q_diag=cfg['Q'], R_val=cfg['R'])
        k_str = f"[{K[0]:.1f}, {K[1]:.1f}, {K[2]:.1f}, {K[3]:.1f}]"

        print(f"  {cfg['label']:<25} {k_str:>30} {r['return']:>8.2f} {r['ET']:>5d} "
              f"{r['ey_rms']:>7.4f} {r['epsi_rms']:>7.4f} {r['theta_rms']:>7.4f} {r['theta_max']:>7.4f}")

    # Find best configs
    print("\n" + "=" * 110)
    print("  RANKING BY SAFETY (ET count, then theta_rms)")
    print("=" * 110)

    safe_rank = sorted(all_results, key=lambda x: (x['ET'], x['theta_rms']))
    for i, r in enumerate(safe_rank[:5]):
        print(f"  #{i+1}: {r['label']:<25} ET={r['ET']}, theta_rms={r['theta_rms']:.4f}, "
              f"return={r['return']:.2f}, ey_rms={r['ey_rms']:.4f}")

    print("\n  RANKING BY RETURN (highest first)")
    print("=" * 110)

    ret_rank = sorted(all_results, key=lambda x: -x['return'])
    for i, r in enumerate(ret_rank[:5]):
        print(f"  #{i+1}: {r['label']:<25} return={r['return']:.2f}, ET={r['ET']}, "
              f"theta_rms={r['theta_rms']:.4f}")

    # Safety score: weighted combination
    print("\n  RANKING BY COMPOSITE SCORE (safety-weighted)")
    print("=" * 110)

    # Normalize metrics
    baseline = next(r for r in all_results if r['label'] == 'baseline')
    for r in all_results:
        # Lower is better for ET, theta_rms; higher is better for return
        et_score = 1.0 - r['ET'] / max(x['ET'] for x in all_results) if max(x['ET'] for x in all_results) > 0 else 1.0
        th_score = 1.0 - (r['theta_rms'] - min(x['theta_rms'] for x in all_results)) / \
                   max(0.001, max(x['theta_rms'] for x in all_results) - min(x['theta_rms'] for x in all_results))
        ret_score = (r['return'] - min(x['return'] for x in all_results)) / \
                    max(0.01, max(x['return'] for x in all_results) - min(x['return'] for x in all_results))
        # Safety-weighted: ET 40%, theta_rms 30%, return 30%
        r['composite_score'] = 0.4 * et_score + 0.3 * th_score + 0.3 * ret_score

    comp_rank = sorted(all_results, key=lambda x: -x['composite_score'])
    for i, r in enumerate(comp_rank[:5]):
        print(f"  #{i+1}: {r['label']:<25} score={r['composite_score']:.3f}  "
              f"ET={r['ET']}, theta_rms={r['theta_rms']:.4f}, return={r['return']:.2f}")

    # Save results
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, f'stage1_lqr_sweep_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'n_seeds': n_seeds,
            'baseline_Q': [100, 10, 10, 1],
            'baseline_R': 1.0,
            'configs': [
                {k: v for k, v in r.items() if k != 'composite_score'}
                for r in all_results
            ],
            'best_safety': safe_rank[0]['label'],
            'best_return': ret_rank[0]['label'],
            'best_composite': comp_rank[0]['label'],
        }, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Save best config for later use
    best = comp_rank[0]
    best_path = os.path.join(base_dir, 'configs', 'best_lqr.json')
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    with open(best_path, 'w') as f:
        json.dump({
            'label': best['label'],
            'Q_diag': best['Q_diag'],
            'R_val': best['R_val'],
            'K_lqr': list(_compute_lqr_gains(Q_diag=best['Q_diag'], R_val=best['R_val'])),
            'metrics': {k: v for k, v in best.items() if k not in ('Q_diag', 'R_val', 'label')},
        }, f, indent=2)
    print(f"  Saved best LQR config: {best_path}")

    print("\n" + "=" * 110)


if __name__ == '__main__':
    main()
