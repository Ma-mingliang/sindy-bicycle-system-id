"""Safety-oriented LQR sweep.

Goal: minimize ET (theta hitting pi/2), not maximize return.
Tests higher Q[0]/Q[1] and lower R to keep theta under control.
"""
import sys, os, json
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tracking_env import PathTrackingEnv, _compute_lqr_gains


def run_episode(env, seed, max_steps=1500):
    obs, _ = env.reset(seed=seed)
    total_r = 0
    early_term = False
    ey_list, epsi_list, theta_list = [], [], []
    u_list = []

    for step in range(max_steps):
        action = np.array([0.0])
        obs, r, t, tr, info = env.step(action)
        total_r += r

        raw = info.get('raw_state', np.zeros(8))
        ey_list.append(abs(raw[0]))
        epsi_list.append(abs(raw[1]))
        theta_list.append(abs(raw[3]))
        u_list.append(abs(info.get('u_total', 0)))

        if t:
            early_term = True
        if t or tr:
            break

    return {
        'return': total_r, 'length': step + 1, 'early_term': early_term,
        'ey_rms': float(np.sqrt(np.mean(np.array(ey_list)**2))),
        'epsi_rms': float(np.sqrt(np.mean(np.array(epsi_list)**2))),
        'theta_rms': float(np.sqrt(np.mean(np.array(theta_list)**2))),
        'theta_max': float(np.max(np.array(theta_list))) if theta_list else 0.0,
        'u_mean': float(np.mean(u_list)),
        'u_max': float(np.max(u_list)),
    }


def evaluate(Q_diag, R_val, label, n_seeds=25):
    env = PathTrackingEnv(max_episode_steps=1500, residual_injection='stanley_ref',
                          lqr_Q=Q_diag, lqr_R=R_val)
    results = []
    for s in range(n_seeds):
        results.append(run_episode(env, seed=s))
    env.close()

    K = _compute_lqr_gains(Q_diag=Q_diag, R_val=R_val)
    return {
        'label': label, 'Q_diag': Q_diag, 'R_val': R_val, 'K': list(K),
        'return': float(np.mean([r['return'] for r in results])),
        'ET': int(sum(r['early_term'] for r in results)),
        'ET_rate': float(sum(r['early_term'] for r in results) / n_seeds),
        'ey_rms': float(np.mean([r['ey_rms'] for r in results])),
        'epsi_rms': float(np.mean([r['epsi_rms'] for r in results])),
        'theta_rms': float(np.mean([r['theta_rms'] for r in results])),
        'theta_max': float(max(r['theta_max'] for r in results)),
        'u_mean': float(np.mean([r['u_mean'] for r in results])),
        'u_max': float(max(r['u_max'] for r in results)),
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    n_seeds = 25

    print("=" * 120)
    print("  Safety-Oriented LQR Sweep")
    print("  Goal: minimize ET, theta_rms; keep epsi_rms stable")
    print(f"  Timestamp: {timestamp}")
    print("=" * 120)

    # Current best: Q500_D50_R02, ET=5/20(25%)
    # Try more aggressive safety configs
    configs = [
        # Current best (reference)
        {'label': 'Q500_D50_R02',     'Q': [500, 50, 10, 1],  'R': 0.2},
        # Higher theta penalty
        {'label': 'Q800_D50_R02',     'Q': [800, 50, 10, 1],  'R': 0.2},
        {'label': 'Q1000_D50_R02',    'Q': [1000, 50, 10, 1], 'R': 0.2},
        {'label': 'Q1500_D50_R02',    'Q': [1500, 50, 10, 1], 'R': 0.2},
        # Higher theta_dot penalty (smoother)
        {'label': 'Q800_D100_R02',    'Q': [800, 100, 10, 1], 'R': 0.2},
        {'label': 'Q1000_D100_R02',   'Q': [1000, 100, 10, 1],'R': 0.2},
        # Even lower R
        {'label': 'Q800_D50_R01',     'Q': [800, 50, 10, 1],  'R': 0.1},
        {'label': 'Q1000_D50_R01',    'Q': [1000, 50, 10, 1], 'R': 0.1},
        # Combined aggressive
        {'label': 'Q1000_D100_R01',   'Q': [1000, 100, 10, 1],'R': 0.1},
        {'label': 'Q1500_D100_R01',   'Q': [1500, 100, 10, 1],'R': 0.1},
        # Very aggressive
        {'label': 'Q2000_D200_R01',   'Q': [2000, 200, 10, 1],'R': 0.1},
        # Lower delta penalty (allow more steering)
        {'label': 'Q1000_D100_D5_R01','Q': [1000, 100, 5, 1], 'R': 0.1},
    ]

    all_results = []

    print(f"\n  Testing {len(configs)} configs × {n_seeds} seeds each\n")
    print(f"  {'Config':<25} {'K_lqr':>35} {'ret':>8} {'ET':>5} {'ET%':>6} {'ey':>7} {'ep':>7} {'th':>7} {'th_max':>7} {'u_max':>8}")
    print(f"  {'-'*25} {'-'*35} {'-'*8} {'-'*5} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    for cfg in configs:
        r = evaluate(cfg['Q'], cfg['R'], cfg['label'], n_seeds)
        all_results.append(r)

        k_str = f"[{r['K'][0]:.0f}, {r['K'][1]:.0f}, {r['K'][2]:.0f}, {r['K'][3]:.0f}]"
        print(f"  {cfg['label']:<25} {k_str:>35} {r['return']:>8.1f} {r['ET']:>5d} {r['ET_rate']:>5.0%} "
              f"{r['ey_rms']:>7.4f} {r['epsi_rms']:>7.4f} {r['theta_rms']:>7.4f} {r['theta_max']:>7.4f} {r['u_max']:>8.1f}")

    # Rank by safety (ET, then theta_rms)
    print("\n" + "=" * 120)
    print("  RANKING BY SAFETY (ET count, then theta_rms)")
    print("=" * 120)

    safe_rank = sorted(all_results, key=lambda x: (x['ET'], x['theta_rms']))
    for i, r in enumerate(safe_rank[:5]):
        print(f"  #{i+1}: {r['label']:<25} ET={r['ET']}/{n_seeds}({r['ET_rate']:.0%})  "
              f"theta_rms={r['theta_rms']:.4f}  return={r['return']:.1f}  epsi_rms={r['epsi_rms']:.4f}")

    # Check if any config achieves ET < 25%
    best = safe_rank[0]
    print(f"\n  Best safety: {best['label']} (ET={best['ET']}/{n_seeds}={best['ET_rate']:.0%})")

    if best['ET_rate'] < 0.20:
        print("  TARGET MET: ET < 20%")
    elif best['ET_rate'] < 0.25:
        print("  TARGET MET: ET < 25%")
    else:
        print(f"  TARGET NOT MET: best ET={best['ET_rate']:.0%}")

    # Save
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, f'stage_safety_lqr_sweep_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp, 'n_seeds': n_seeds,
            'configs': [{k: v for k, v in r.items()} for r in all_results],
            'best_safety': best['label'],
        }, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Save best safety config
    best_path = os.path.join(base_dir, 'configs', 'best_lqr_safety.json')
    with open(best_path, 'w') as f:
        json.dump({
            'label': best['label'],
            'Q_diag': best['Q_diag'],
            'R_val': best['R_val'],
            'K_lqr': best['K'],
            'metrics': {k: v for k, v in best.items() if k not in ('Q_diag', 'R_val', 'label', 'K')},
        }, f, indent=2)
    print(f"  Saved best safety config: {best_path}")

    print("=" * 120)


if __name__ == '__main__':
    main()
