"""Stage 1: Verify all three injection modes produce different trajectories."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tracking_env import PathTrackingEnv


def run_episode(env, seed, max_steps=1500):
    obs, _ = env.reset(seed=seed)
    total_r = 0
    ey_list, theta_list = [], []

    for step in range(max_steps):
        action = np.array([0.0])  # zero residual
        obs, r, t, tr, info = env.step(action)
        total_r += r
        raw = info.get('raw_state', np.zeros(8))
        ey_list.append(abs(raw[0]))
        theta_list.append(abs(raw[3]))
        if t or tr:
            break

    n = step + 1
    return {
        'return': total_r, 'length': n, 'early_term': t,
        'ey_rms': float(np.sqrt(np.mean(np.array(ey_list)**2))),
        'theta_rms': float(np.sqrt(np.mean(np.array(theta_list)**2))),
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    modes = ['final_action', 'stanley_ref', 'theta_target']
    seeds = list(range(5))

    print("=" * 80)
    print("  Stage 1: Injection Mode Verification")
    print("  Testing all 3 modes with zero residual (should be identical)")
    print("=" * 80)

    for mode in modes:
        env = PathTrackingEnv(max_episode_steps=1500, residual_injection=mode)
        results = []
        for seed in seeds:
            r = run_episode(env, seed)
            results.append(r)
        env.close()

        ret_mean = np.mean([r['return'] for r in results])
        et_count = sum(r['early_term'] for r in results)
        print(f"  {mode:<20}: ret={ret_mean:+.2f}, ET={et_count}/5")

    print("\n  All modes with zero residual should produce identical results.")
    print("  If not, there is a bug in the injection implementation.")
    print("=" * 80)


if __name__ == '__main__':
    main()
