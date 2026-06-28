"""50-seed baseline evaluation with Q1000_D100_R02 safety LQR config."""
import sys, os, json, csv
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tracking_env import PathTrackingEnv


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
        'control_energy': float(np.mean(np.array(u_list)**2)),
        'u_mean': float(np.mean(u_list)),
        'u_max': float(np.max(u_list)),
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load LQR config from command line arg or default
    cfg_name = sys.argv[1] if len(sys.argv) > 1 else 'best_lqr_safety.json'
    cfg_path = os.path.join(base_dir, 'configs', cfg_name)
    with open(cfg_path) as f:
        lqr_cfg = json.load(f)

    print("=" * 90)
    print("  50-Seed Baseline Evaluation (Safety LQR)")
    print(f"  LQR: {lqr_cfg['label']} (Q={lqr_cfg['Q_diag']}, R={lqr_cfg['R_val']})")
    print(f"  K: {lqr_cfg['K_lqr']}")
    print(f"  Timestamp: {timestamp}")
    print("=" * 90)

    env = PathTrackingEnv(max_episode_steps=1500, residual_injection='stanley_ref',
                          lqr_Q=lqr_cfg['Q_diag'], lqr_R=lqr_cfg['R_val'])

    n_episodes = 50
    results = []

    for ep in range(n_episodes):
        r = run_episode(env, seed=ep)
        results.append(r)

        if (ep + 1) % 10 == 0:
            avg_ret = np.mean([x['return'] for x in results])
            et = sum(x['early_term'] for x in results)
            print(f"  [{ep+1}/{n_episodes}] avg_return={avg_ret:.2f}, ET={et}/{ep+1}")

    env.close()

    # Statistics
    ret = np.array([r['return'] for r in results])
    et_count = sum(r['early_term'] for r in results)
    ey_rms = np.array([r['ey_rms'] for r in results])
    epsi_rms = np.array([r['epsi_rms'] for r in results])
    theta_rms = np.array([r['theta_rms'] for r in results])
    theta_max = np.array([r['theta_max'] for r in results])
    u_mean = np.array([r['u_mean'] for r in results])
    u_max = np.array([r['u_max'] for r in results])

    print("\n" + "=" * 90)
    print("  RESULTS")
    print("=" * 90)
    print(f"  Return:      {np.mean(ret):.2f} ± {np.std(ret):.2f}")
    print(f"  ET:          {et_count}/{n_episodes} ({100*et_count/n_episodes:.1f}%)")
    print(f"  ey_rms:      {np.mean(ey_rms):.4f} ± {np.std(ey_rms):.4f}")
    print(f"  epsi_rms:    {np.mean(epsi_rms):.4f} ± {np.std(epsi_rms):.4f}")
    print(f"  theta_rms:   {np.mean(theta_rms):.4f} ± {np.std(theta_rms):.4f}")
    print(f"  theta_max:   {np.max(theta_max):.4f}")
    print(f"  u_mean:      {np.mean(u_mean):.4f}")
    print(f"  u_max:       {np.max(u_max):.1f}")

    # Save
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    summary = {
        'timestamp': timestamp,
        'lqr_config': lqr_cfg,
        'n_episodes': n_episodes,
        'return_mean': float(np.mean(ret)),
        'return_std': float(np.std(ret)),
        'early_termination_count': int(et_count),
        'early_termination_rate': float(et_count / n_episodes),
        'ey_rms_mean': float(np.mean(ey_rms)),
        'epsi_rms_mean': float(np.mean(epsi_rms)),
        'theta_rms_mean': float(np.mean(theta_rms)),
        'theta_max_max': float(np.max(theta_max)),
        'u_mean': float(np.mean(u_mean)),
        'u_max': float(np.max(u_max)),
        'per_seed': [
            {
                'seed': i, 'return': float(results[i]['return']),
                'length': int(results[i]['length']),
                'early_term': bool(results[i]['early_term']),
                'ey_rms': float(results[i]['ey_rms']),
                'epsi_rms': float(results[i]['epsi_rms']),
                'theta_rms': float(results[i]['theta_rms']),
                'theta_max': float(results[i]['theta_max']),
            }
            for i in range(n_episodes)
        ],
    }

    json_path = os.path.join(results_dir, f'baseline_safety_lqr_50seeds_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {json_path}")
    print("=" * 90)


if __name__ == '__main__':
    main()
