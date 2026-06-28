"""Stage 1: Train SINDy+MPPI attitude controller on straight-line environment.

Pipeline:
  Phase 0: Collect data (straight-line + progressive disturbance + random residual)
  Phase 1: Train ensemble world model (SINDy + NN residual)
  Phase 2: MPPI evaluation (zero-residual vs MPPI)
  Phase 3: Save best model

Reference: HRRL train_attitude.py (TD3) → replaced with SINDy+MPPI
"""
import sys, os, json
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attitude_control_env import AttitudeControlEnv
from residual_mppi import EnsembleWorldModel, ConservativeEnsembleResidualMPPI
from reward_fn import compute_attitude_reward_torch


def collect_data(env, target_steps, epsilon_range=0.1):
    """Collect transitions with random residual on steering torque."""
    all_obs, all_action, all_next_obs, all_reward, all_done = [], [], [], [], []
    total_steps = 0
    episode_count = 0

    while total_steps < target_steps:
        seed = episode_count
        obs, _ = env.reset(seed=seed)

        for step in range(env.max_episode_steps):
            epsilon = np.random.uniform(-epsilon_range, epsilon_range)
            action = np.array([epsilon])

            prev_obs = obs.copy()
            obs, r, t, tr, info = env.step(action)

            all_obs.append(prev_obs)
            all_action.append(action)
            all_next_obs.append(obs.copy())
            all_reward.append(r)
            all_done.append(t)

            total_steps += 1
            if t or tr:
                break

        episode_count += 1
        if episode_count % 10 == 0:
            print(f"  Episodes: {episode_count}, Steps: {total_steps}/{target_steps}")

    return {
        'obs': np.array(all_obs[:target_steps], dtype=np.float32),
        'action': np.array(all_action[:target_steps], dtype=np.float32),
        'next_obs': np.array(all_next_obs[:target_steps], dtype=np.float32),
        'reward': np.array(all_reward[:target_steps], dtype=np.float32),
        'done': np.array(all_done[:target_steps], dtype=bool),
        'n_episodes': episode_count,
    }


class SimpleBuffer:
    """Simple replay buffer from collected data."""
    def __init__(self, data):
        self.obs = data['obs']
        self.action = data['action']
        self.next_obs = data['next_obs']
        self.size = len(self.obs)


def evaluate_baseline(env, n_episodes=10):
    """Evaluate zero-residual baseline."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total_r = 0
        et = False
        theta_list = []

        for step in range(env.max_episode_steps):
            action = np.array([0.0])
            obs, r, t, tr, info = env.step(action)
            total_r += r
            theta_list.append(abs(info['raw_state'][3]))
            if t:
                et = True
            if t or tr:
                break

        results.append({
            'return': total_r,
            'length': step + 1,
            'early_term': et,
            'theta_rms': float(np.sqrt(np.mean(np.array(theta_list)**2))),
            'theta_max': float(np.max(np.array(theta_list))),
        })

    ret = np.mean([r['return'] for r in results])
    et_count = sum(r['early_term'] for r in results)
    theta_rms = np.mean([r['theta_rms'] for r in results])
    return {
        'return': ret,
        'ET': et_count,
        'ET_rate': et_count / n_episodes,
        'theta_rms': theta_rms,
    }


def evaluate_mppi(env, agent, n_episodes=3, max_steps=200):
    """Evaluate MPPI agent (reduced for speed)."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        agent.reset_planning()
        total_r = 0
        et = False
        theta_list = []
        eps_list = []

        for step in range(min(max_steps, env.max_episode_steps)):
            eps, best_G, zero_G, unc = agent.act(obs, eval_mode=True)
            obs, r, t, tr, info = env.step(eps)
            total_r += r
            theta_list.append(abs(info['raw_state'][3]))
            eps_list.append(float(np.abs(eps[0])))
            if t:
                et = True
            if t or tr:
                break

        results.append({
            'return': total_r,
            'length': step + 1,
            'early_term': et,
            'theta_rms': float(np.sqrt(np.mean(np.array(theta_list)**2))),
            'theta_max': float(np.max(np.array(theta_list))),
            'mean_eps': float(np.mean(eps_list)),
        })

    ret = np.mean([r['return'] for r in results])
    et_count = sum(r['early_term'] for r in results)
    theta_rms = np.mean([r['theta_rms'] for r in results])
    mean_eps = np.mean([r['mean_eps'] for r in results])
    return {
        'return': ret,
        'ET': et_count,
        'ET_rate': et_count / n_episodes,
        'theta_rms': theta_rms,
        'mean_eps': mean_eps,
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Parse args
    target_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 150000
    skip_collection = '--skip-collection' in sys.argv

    # Load LQR config
    best_lqr_path = os.path.join(base_dir, 'configs', 'best_lqr.json')
    lqr_kwargs = {}
    lqr_label = 'default'
    if os.path.exists(best_lqr_path):
        with open(best_lqr_path) as f:
            best = json.load(f)
        lqr_kwargs = {'lqr_Q': best['Q_diag'], 'lqr_R': best['R_val']}
        lqr_label = best['label']

    print("=" * 80)
    print("  Stage 1: SINDy+MPPI Attitude Control Training")
    print(f"  Target: {target_steps} steps")
    print(f"  LQR: {lqr_label}")
    print(f"  Timestamp: {timestamp}")
    print("=" * 80)

    # ============================================================
    # Phase 0: Data Collection
    # ============================================================
    data_dir = os.path.join(base_dir, 'data')
    data_path = os.path.join(data_dir, f'stage1_dataset_{target_steps//1000}k.npz')

    if skip_collection and os.path.exists(data_path):
        print("\n" + "=" * 80)
        print("  Phase 0: Loading Existing Dataset")
        print("=" * 80)
        loaded = np.load(data_path, allow_pickle=True)
        data = {
            'obs': loaded['obs'],
            'action': loaded['action'],
            'next_obs': loaded['next_obs'],
            'reward': loaded['reward'],
            'done': loaded['done'],
            'n_episodes': int(loaded['n_episodes']),
        }
        print(f"  Loaded: {data_path}")
        print(f"  Episodes: {data['n_episodes']}, Steps: {len(data['obs'])}")
        print(f"  Done rate: {data['done'].mean():.1%}")
    else:
        print("\n" + "=" * 80)
        print("  Phase 0: Data Collection")
        print("=" * 80)

        env = AttitudeControlEnv(max_episode_steps=1000, **lqr_kwargs)
        data = collect_data(env, target_steps, epsilon_range=0.1)

        # Save data
        os.makedirs(data_dir, exist_ok=True)
        np.savez_compressed(data_path,
            obs=data['obs'], action=data['action'], next_obs=data['next_obs'],
            reward=data['reward'], done=data['done'],
            n_episodes=data['n_episodes'], n_steps=target_steps,
            lqr_label=lqr_label,
            lqr_Q=np.array(lqr_kwargs.get('lqr_Q', [100, 10, 10, 1])),
            lqr_R=np.array([lqr_kwargs.get('lqr_R', 1.0)]),
            timestamp=timestamp,
        )
        print(f"  Saved: {data_path}")
        print(f"  Done rate: {data['done'].mean():.1%}")

    # ============================================================
    # Phase 1: Train Ensemble World Model
    # ============================================================
    print("\n" + "=" * 80)
    print("  Phase 1: Training Ensemble World Model")
    print("=" * 80)

    # Create env for evaluation (needed for Phase 2)
    env = AttitudeControlEnv(max_episode_steps=1000, **lqr_kwargs)

    sindy_data = np.load(os.path.join(base_dir, 'sindy_model_improved.npz'))
    sindy_Xi = sindy_data['coefficients']
    action_scale = float(sindy_data['action_scale'])

    cfg = {
        'obs_dim': 8, 'action_dim': 1, 'mlp_dim': 256,
        'ensemble_size': 3,
    }
    agent = ConservativeEnsembleResidualMPPI(
        cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)

    buffer = SimpleBuffer(data)
    results = agent.train_world_model(buffer, epochs=50, batch_size=256, val_ratio=0.2)

    # Model errors
    errors = agent.compute_model_errors(buffer)
    print(f"  SINDy error: {errors['sindy_error']:.6f}")
    print(f"  Model error: {errors['model_error']:.6f}")

    # Uncertainty
    val_unc = agent.compute_val_uncertainty_percentiles(buffer, val_ratio=0.2)
    print(f"  Val p95 uncertainty: {val_unc['p95']:.6f}")

    # Rollout
    rollout = agent.validate_rollout(buffer, horizon=5, n_samples=100)
    print(f"  5-step error: {rollout['multi_step_error'][-1]:.6f}")
    print(f"  Diverges: {rollout['diverges']}")

    # ============================================================
    # Phase 2: MPPI Evaluation (fast: 3 episodes × 200 steps)
    # ============================================================
    print("\n" + "=" * 80)
    print("  Phase 2: MPPI Evaluation (fast)")
    print("=" * 80)

    # Baseline
    baseline = evaluate_baseline(env, n_episodes=3)
    print(f"  Baseline: return={baseline['return']:.2f}, ET={baseline['ET']}/3, "
          f"theta_rms={baseline['theta_rms']:.4f}")

    # MPPI evaluation (reduced parameters for speed)
    mppi_cfg = {
        'obs_dim': 8, 'action_dim': 1, 'mlp_dim': 256,
        'ensemble_size': 3,
        'horizon': 6, 'num_samples': 64, 'num_elites': 8,
        'iterations': 4, 'temperature': 0.5,
        'epsilon_max': 0.1, 'epsilon_std': 0.08,
        'gamma': 0.95,
        'lambda_res': 2.0, 'lambda_smooth': 5.0,
        'lambda_uncertainty': 5.0,
        'uncertainty_threshold': val_unc['p95'],
        'risk_mode': 'tolerance',
        'ey_tol': 0.1, 'theta_tol': 0.05,
        'margin': 0.01,
    }
    mppi_agent = ConservativeEnsembleResidualMPPI(
        mppi_cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
    # Copy trained ensemble
    mppi_agent.ensemble = agent.ensemble

    mppi_result = evaluate_mppi(env, mppi_agent, n_episodes=3, max_steps=200)
    advantage = mppi_result['return'] - baseline['return']
    print(f"  MPPI: return={mppi_result['return']:.2f}, ET={mppi_result['ET']}/3, "
          f"theta_rms={mppi_result['theta_rms']:.4f}")
    print(f"  Advantage: {advantage:.2f}")

    # ============================================================
    # Phase 3: Save Best Model
    # ============================================================
    print("\n" + "=" * 80)
    print("  Phase 3: Save Best Model")
    print("=" * 80)

    ckpt_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'stage1_ensemble.pt')

    agent.save(ckpt_path, extra_meta={
        'stage': 1,
        'injection_mode': 'torque_residual',
        'train_data_source': data_path,
        'reward_version': 'attitude_hrrl',
        'created_at': timestamp,
        'dataset_size': target_steps,
        'n_samples': len(data['obs']),
        'sindy_error': errors['sindy_error'],
        'model_error': errors['model_error'],
        'val_uncertainty_p95': val_unc['p95'],
        'rollout_h5_error': float(rollout['multi_step_error'][-1]),
        'rollout_diverges': rollout['diverges'],
        'baseline_return': float(baseline['return']),
        'baseline_ET': int(baseline['ET']),
        'mppi_return': float(mppi_result['return']),
        'mppi_ET': int(mppi_result['ET']),
        'mppi_advantage': float(advantage),
        'lqr_label': lqr_label,
        'lqr_Q': lqr_kwargs.get('lqr_Q', [100, 10, 10, 1]),
        'lqr_R': lqr_kwargs.get('lqr_R', 1.0),
    })

    # Save config
    config_path = os.path.join(base_dir, 'configs', 'stage1_best.json')
    with open(config_path, 'w') as f:
        json.dump({
            'stage': 1,
            'checkpoint': 'stage1_ensemble.pt',
            'obs_dim': 8,
            'action_dim': 1,
            'mlp_dim': 256,
            'ensemble_size': 3,
            'lqr_label': lqr_label,
            'lqr_Q': lqr_kwargs.get('lqr_Q', [100, 10, 10, 1]),
            'lqr_R': lqr_kwargs.get('lqr_R', 1.0),
            'horizon': 8,
            'epsilon_max': 0.1,
            'uncertainty_threshold': float(val_unc['p95']),
            'baseline': baseline,
            'mppi': mppi_result,
            'advantage': float(advantage),
            'timestamp': timestamp,
        }, f, indent=2)

    print(f"  Saved checkpoint: {ckpt_path}")
    print(f"  Saved config: {config_path}")
    print(f"  Advantage: {advantage:.2f}")
    print("=" * 80)

    env.close()


if __name__ == '__main__':
    main()
