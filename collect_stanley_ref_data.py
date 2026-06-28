"""Stage 2: Collect training data with stanley_ref injection mode.

Collects transitions where random epsilon_path is added to Stanley reference:
  delta_stanley_res = delta_stanley + epsilon_path
  theta_target = steady_state(delta_stanley_res)
  u_total = LQR(theta_target)

Saves: obs, action (epsilon_path), next_obs, reward, done, metadata.
"""
import sys, os
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tracking_env import PathTrackingEnv
from reward_fn import compute_tracking_reward


def collect_episode(env, seed, max_steps=1500):
    """Collect one episode of transitions."""
    obs, _ = env.reset(seed=seed)
    transitions = []

    for step in range(max_steps):
        # Random epsilon_path noise
        epsilon_path = np.random.uniform(-0.1, 0.1)
        action = np.array([epsilon_path])

        prev_obs = obs.copy()
        obs, r, t, tr, info = env.step(action)

        transitions.append({
            'obs': prev_obs,
            'action': np.array([epsilon_path]),
            'next_obs': obs.copy(),
            'reward': r,
            'done': t,
            'seed': seed,
            'step': step,
        })

        if t or tr:
            break

    return transitions


def main():
    sys.stdout.reconfigure(line_buffering=True)

    # Parse target steps from command line
    target_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load best LQR config if available
    base_dir = os.path.dirname(os.path.abspath(__file__))
    best_lqr_path = os.path.join(base_dir, 'configs', 'best_lqr.json')
    lqr_kwargs = {}
    lqr_label = 'default'
    if os.path.exists(best_lqr_path):
        import json
        with open(best_lqr_path) as f:
            best = json.load(f)
        lqr_kwargs = {'lqr_Q': best['Q_diag'], 'lqr_R': best['R_val']}
        lqr_label = best['label']

    print("=" * 80)
    print(f"  Collecting Stanley Reference Data")
    print(f"  Target: {target_steps} steps")
    print(f"  Injection: stanley_ref, epsilon_range=[-0.1, 0.1]")
    print(f"  LQR: {lqr_label}")
    print(f"  Timestamp: {timestamp}")
    print("=" * 80)

    env = PathTrackingEnv(max_episode_steps=1500, residual_injection='stanley_ref',
                          **lqr_kwargs)

    all_obs, all_action, all_next_obs, all_reward, all_done = [], [], [], [], []
    all_meta = []
    total_steps = 0
    episode_count = 0

    while total_steps < target_steps:
        seed = episode_count
        transitions = collect_episode(env, seed)

        for t in transitions:
            all_obs.append(t['obs'])
            all_action.append(t['action'])
            all_next_obs.append(t['next_obs'])
            all_reward.append(t['reward'])
            all_done.append(t['done'])
            all_meta.append({
                'seed': t['seed'], 'step': t['step'],
                'injection_mode': 'stanley_ref',
            })

        total_steps += len(transitions)
        episode_count += 1

        if episode_count % 10 == 0:
            print(f"  Episodes: {episode_count}, Steps: {total_steps}/{target_steps}")

    env.close()

    # Convert to arrays
    obs_arr = np.array(all_obs[:target_steps], dtype=np.float32)
    action_arr = np.array(all_action[:target_steps], dtype=np.float32)
    next_obs_arr = np.array(all_next_obs[:target_steps], dtype=np.float32)
    reward_arr = np.array(all_reward[:target_steps], dtype=np.float32)
    done_arr = np.array(all_done[:target_steps], dtype=bool)

    # Save
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)

    filename = f'stanley_ref_dataset_{target_steps//1000}k.npz'
    filepath = os.path.join(data_dir, filename)

    np.savez_compressed(filepath,
        obs=obs_arr, action=action_arr, next_obs=next_obs_arr,
        reward=reward_arr, done=done_arr,
        injection_mode='stanley_ref',
        epsilon_range='[-0.1, 0.1]',
        timestamp=timestamp,
        n_episodes=episode_count,
        n_steps=target_steps,
        lqr_label=lqr_label,
        lqr_Q=np.array(lqr_kwargs.get('lqr_Q', [100, 10, 10, 1])),
        lqr_R=np.array([lqr_kwargs.get('lqr_R', 1.0)]),
    )

    print(f"\n  Saved: {filepath}")
    print(f"  Episodes: {episode_count}")
    print(f"  Steps: {target_steps}")
    print(f"  Obs shape: {obs_arr.shape}")
    print(f"  Action shape: {action_arr.shape}")
    print(f"  Done rate: {done_arr.mean():.1%}")
    print("=" * 80)


if __name__ == '__main__':
    main()
