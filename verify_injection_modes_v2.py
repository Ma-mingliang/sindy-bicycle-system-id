"""Stage 3: Verify residual injection modes with best LQR config.

Compares all three injection modes across multiple seeds to confirm
that stanley_ref is still the best mode with the new LQR gains.
"""
import sys, os, json
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tracking_env import PathTrackingEnv


def run_episode(env, seed, action_fn, max_steps=1500):
    obs, _ = env.reset(seed=seed)
    total_r = 0
    early_term = False
    ey_list, epsi_list, theta_list = [], [], []
    u_list = []

    for step in range(max_steps):
        action = action_fn(step)
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

    n = step + 1
    return {
        'return': total_r, 'length': n, 'early_term': early_term,
        'ey_rms': float(np.sqrt(np.mean(np.array(ey_list)**2))),
        'epsi_rms': float(np.sqrt(np.mean(np.array(epsi_list)**2))),
        'theta_rms': float(np.sqrt(np.mean(np.array(theta_list)**2))),
        'theta_max': float(np.max(np.array(theta_list))) if theta_list else 0.0,
        'control_energy': float(np.mean(np.array(u_list)**2)),
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load best LQR config
    best_lqr_path = os.path.join(base_dir, 'configs', 'best_lqr.json')
    lqr_kwargs = {}
    if os.path.exists(best_lqr_path):
        with open(best_lqr_path) as f:
            best = json.load(f)
        lqr_kwargs = {'lqr_Q': best['Q_diag'], 'lqr_R': best['R_val']}
        print(f"  LQR: {best['label']} (Q={best['Q_diag']}, R={best['R_val']})")
    else:
        print("  LQR: default")

    print("=" * 90)
    print("  Stage 3: Injection Mode Verification")
    print(f"  Timestamp: {timestamp}")
    print("=" * 90)

    modes = ['stanley_ref', 'theta_target', 'final_action']
    n_seeds = 20

    # Action modes to test
    action_configs = {
        'zero': lambda s: np.array([0.0]),
        'small_pos': lambda s: np.array([0.05]),
        'small_neg': lambda s: np.array([-0.05]),
        'noise': lambda s: np.array([np.random.uniform(-0.1, 0.1)]),
    }

    all_results = {}

    for mode in modes:
        print(f"\n  --- {mode} ---")
        env = PathTrackingEnv(max_episode_steps=1500, residual_injection=mode, **lqr_kwargs)

        for action_name, action_fn in action_configs.items():
            results = []
            for s in range(n_seeds):
                r = run_episode(env, seed=s, action_fn=action_fn)
                results.append(r)

            ret = np.mean([r['return'] for r in results])
            et = sum(r['early_term'] for r in results)
            ey = np.mean([r['ey_rms'] for r in results])
            ep = np.mean([r['epsi_rms'] for r in results])
            th = np.mean([r['theta_rms'] for r in results])
            th_max = max(r['theta_max'] for r in results)
            ctrl = np.mean([r['control_energy'] for r in results])

            key = f"{mode}_{action_name}"
            all_results[key] = {
                'mode': mode, 'action': action_name,
                'return': float(ret), 'ET': int(et),
                'ey_rms': float(ey), 'epsi_rms': float(ep),
                'theta_rms': float(th), 'theta_max': float(th_max),
                'control_energy': float(ctrl),
            }

            print(f"    {action_name:<12} ret={ret:>8.2f}  ET={et:>3}/{n_seeds}  "
                  f"ey={ey:.4f}  ep={ep:.4f}  th={th:.4f}  th_max={th_max:.4f}")

        env.close()

    # Summary table
    print("\n" + "=" * 90)
    print("  SUMMARY (zero action, 20 seeds)")
    print("=" * 90)

    print(f"\n  {'Mode':<15} {'Return':>8} {'ET':>5} {'ey_rms':>8} {'ep_rms':>8} {'th_rms':>8} {'th_max':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for mode in modes:
        key = f"{mode}_zero"
        r = all_results[key]
        print(f"  {mode:<15} {r['return']:>8.2f} {r['ET']:>5d} "
              f"{r['ey_rms']:>8.4f} {r['epsi_rms']:>8.4f} {r['theta_rms']:>8.4f} {r['theta_max']:>8.4f}")

    # Check if injection modes affect zero-action behavior
    print("\n  CONSISTENCY CHECK: All modes with zero action should produce identical results")
    zero_returns = [all_results[f"{m}_zero"]['return'] for m in modes]
    if max(zero_returns) - min(zero_returns) < 1.0:
        print("  PASS: Zero-action returns are consistent across modes")
    else:
        print(f"  WARNING: Zero-action returns differ: {[f'{r:.2f}' for r in zero_returns]}")

    # Check if non-zero actions have any effect
    print("\n  ACTION SENSITIVITY CHECK:")
    for mode in modes:
        zero_ret = all_results[f"{mode}_zero"]['return']
        pos_ret = all_results[f"{mode}_small_pos"]['return']
        neg_ret = all_results[f"{mode}_small_neg"]['return']
        delta = max(abs(pos_ret - zero_ret), abs(neg_ret - zero_ret))
        sensitivity = "HIGH" if delta > 5 else "LOW" if delta < 0.5 else "MEDIUM"
        print(f"    {mode:<15} zero={zero_ret:.2f}  +0.05={pos_ret:.2f}  -0.05={neg_ret:.2f}  "
              f"delta={delta:.2f}  sensitivity={sensitivity}")

    # Save
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, f'stage3_injection_verify_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'n_seeds': n_seeds,
            'lqr_config': lqr_kwargs,
            'results': all_results,
        }, f, indent=2)
    print(f"\n  Saved: {json_path}")

    print("\n" + "=" * 90)


if __name__ == '__main__':
    main()
