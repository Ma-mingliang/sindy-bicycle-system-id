"""Stage 2: Seed=42 debug framework.

Runs a single episode with seed=42 and logs detailed per-step diagnostics.
Used to verify that the control pipeline works correctly.
"""
import sys, os, json
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tracking_env import PathTrackingEnv


def run_debug_episode(env, seed=42, max_steps=200, action_mode='zero'):
    """Run a single episode with detailed per-step logging."""
    obs, _ = env.reset(seed=seed)
    log = []
    total_r = 0

    for step in range(max_steps):
        if action_mode == 'zero':
            action = np.array([0.0])
        elif action_mode == 'small_positive':
            action = np.array([0.05])
        elif action_mode == 'small_negative':
            action = np.array([-0.05])
        elif action_mode == 'alternating':
            action = np.array([0.05 if step % 2 == 0 else -0.05])
        else:
            action = np.array([0.0])

        obs, r, t, tr, info = env.step(action)
        total_r += r

        raw = info.get('raw_state', np.zeros(8))
        entry = {
            'step': step,
            'ey': float(raw[0]),
            'epsi': float(raw[1]),
            'v': float(raw[2]),
            'theta': float(raw[3]),
            'theta_dot': float(raw[4]),
            'k': float(raw[5]),
            'delta': float(raw[6]),
            'delta_dot': float(raw[7]),
            'u_total': float(info.get('u_total', 0)),
            'u_stanley': float(info.get('u_stanley', 0) or 0),
            'u_prior': float(info.get('u_prior', 0) or 0),
            'u_residual': float(info.get('u_residual', 0)),
            'epsilon': float(info.get('epsilon', 0)),
            'reward': float(r),
            'terminated': bool(t),
            'truncated': bool(tr),
            'injection_mode': info.get('injection_mode', ''),
        }
        log.append(entry)

        if t or tr:
            break

    return {
        'seed': seed,
        'action_mode': action_mode,
        'total_reward': total_r,
        'steps': step + 1,
        'terminated': bool(t),
        'truncated': bool(tr),
        'per_step': log,
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load best LQR config if available
    best_lqr_path = os.path.join(base_dir, 'configs', 'best_lqr.json')
    lqr_kwargs = {}
    if os.path.exists(best_lqr_path):
        with open(best_lqr_path) as f:
            best = json.load(f)
        lqr_kwargs = {'lqr_Q': best['Q_diag'], 'lqr_R': best['R_val']}
        print(f"  Using LQR config: {best['label']} (Q={best['Q_diag']}, R={best['R_val']})")
    else:
        print("  Using default LQR config")

    print("=" * 80)
    print(f"  Stage 2: Seed=42 Debug Framework")
    print(f"  Timestamp: {timestamp}")
    print("=" * 80)

    all_results = {}

    for mode in ['stanley_ref', 'theta_target', 'final_action']:
        print(f"\n  --- Injection mode: {mode} ---")
        env = PathTrackingEnv(max_episode_steps=1500, residual_injection=mode, **lqr_kwargs)

        for action_mode in ['zero', 'small_positive', 'small_negative']:
            result = run_debug_episode(env, seed=42, max_steps=200, action_mode=action_mode)
            key = f"{mode}_{action_mode}"
            all_results[key] = result

            steps = result['per_step']
            ey_mean = np.mean([abs(s['ey']) for s in steps])
            theta_mean = np.mean([abs(s['theta']) for s in steps])
            theta_max = max(abs(s['theta']) for s in steps)
            u_mean = np.mean([abs(s['u_total']) for s in steps])

            print(f"    {action_mode:<20} steps={result['steps']:>4}  "
                  f"rwd={result['total_reward']:>8.2f}  "
                  f"ey_mean={ey_mean:.4f}  theta_mean={theta_mean:.4f}  "
                  f"theta_max={theta_max:.4f}  u_mean={u_mean:.4f}  "
                  f"{'ET' if result['terminated'] else 'OK'}")

        env.close()

    # Verify consistency across modes for zero action
    print("\n" + "=" * 80)
    print("  CROSS-MODE CONSISTENCY CHECK (zero action, seed=42)")
    print("=" * 80)

    zero_results = {k: v for k, v in all_results.items() if 'zero' in k}
    for key, result in zero_results.items():
        steps = result['per_step']
        print(f"\n  {key}:")
        print(f"    Steps: {result['steps']}, Terminated: {result['terminated']}")
        print(f"    ey: mean={np.mean([abs(s['ey']) for s in steps]):.4f}, "
              f"max={max(abs(s['ey']) for s in steps):.4f}")
        print(f"    theta: mean={np.mean([abs(s['theta']) for s in steps]):.4f}, "
              f"max={max(abs(s['theta']) for s in steps):.4f}")
        print(f"    u_total: mean={np.mean([abs(s['u_total']) for s in steps]):.4f}, "
              f"max={max(abs(s['u_total']) for s in steps):.4f}")

    # Save
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Save only summary (not per-step for all modes to keep file small)
    summary = {
        'timestamp': timestamp,
        'seed': 42,
        'lqr_config': lqr_kwargs,
        'modes': {},
    }
    for key, result in all_results.items():
        steps = result['per_step']
        summary['modes'][key] = {
            'steps': result['steps'],
            'total_reward': result['total_reward'],
            'terminated': result['terminated'],
            'ey_mean': float(np.mean([abs(s['ey']) for s in steps])),
            'theta_mean': float(np.mean([abs(s['theta']) for s in steps])),
            'theta_max': float(max(abs(s['theta']) for s in steps)),
            'u_total_mean': float(np.mean([abs(s['u_total']) for s in steps])),
            'u_total_max': float(max(abs(s['u_total']) for s in steps)),
        }

    json_path = os.path.join(results_dir, f'stage2_seed42_debug_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Save detailed per-step for the main mode (stanley_ref + zero)
    main_key = 'stanley_ref_zero'
    if main_key in all_results:
        detail_path = os.path.join(results_dir, f'stage2_seed42_detail_{main_key}.json')
        with open(detail_path, 'w') as f:
            json.dump(all_results[main_key], f, indent=2)
        print(f"  Saved detail: {detail_path}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
