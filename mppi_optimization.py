"""MPPI Optimization: Systematic parameter tuning for Stage 2.

Tests:
  1. MPPI parameters (horizon, samples, iterations, temperature)
  2. Residual scaling (epsilon_max, alpha)
  3. Reward weights (lambda_res, lambda_smooth, lambda_uncertainty)
  4. On-policy data collection
"""
import sys, os, json
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tracking_env import PathTrackingEnv
from residual_mppi import ConservativeEnsembleResidualMPPI


class SimpleBuffer:
    def __init__(self, data):
        self.obs = data['obs']
        self.action = data['action']
        self.next_obs = data['next_obs']
        self.size = len(self.obs)


def evaluate_mppi(env, agent, n_episodes=3, max_steps=1500):
    """Evaluate MPPI agent."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        agent.reset_planning()
        total_r = 0
        et = False
        ey_list = []
        epsi_list = []
        theta_list = []

        for step in range(min(max_steps, env.max_episode_steps)):
            eps, best_G, zero_G, unc = agent.act(obs, eval_mode=True)
            obs, r, t, tr, info = env.step(eps)
            total_r += r
            raw = info['raw_state']
            ey_list.append(abs(raw[0]))
            epsi_list.append(abs(raw[1]))
            theta_list.append(abs(raw[3]))
            if t:
                et = True
            if t or tr:
                break

        results.append({
            'return': total_r,
            'length': step + 1,
            'early_term': et,
            'ey_rms': float(np.sqrt(np.mean(np.array(ey_list)**2))),
            'epsi_rms': float(np.sqrt(np.mean(np.array(epsi_list)**2))),
            'theta_rms': float(np.sqrt(np.mean(np.array(theta_list)**2))),
        })

    ret = np.mean([r['return'] for r in results])
    et_count = sum(r['early_term'] for r in results)
    ey_rms = np.mean([r['ey_rms'] for r in results])
    epsi_rms = np.mean([r['epsi_rms'] for r in results])
    theta_rms = np.mean([r['theta_rms'] for r in results])
    return {
        'return': ret,
        'ET': et_count,
        'ET_rate': et_count / n_episodes,
        'ey_rms': ey_rms,
        'epsi_rms': epsi_rms,
        'theta_rms': theta_rms,
    }


def evaluate_baseline(env, n_episodes=3, max_steps=1500):
    """Evaluate zero-residual baseline."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total_r = 0
        et = False
        ey_list = []
        epsi_list = []
        theta_list = []

        for step in range(min(max_steps, env.max_episode_steps)):
            action = np.array([0.0])
            obs, r, t, tr, info = env.step(action)
            total_r += r
            raw = info['raw_state']
            ey_list.append(abs(raw[0]))
            epsi_list.append(abs(raw[1]))
            theta_list.append(abs(raw[3]))
            if t:
                et = True
            if t or tr:
                break

        results.append({
            'return': total_r,
            'length': step + 1,
            'early_term': et,
            'ey_rms': float(np.sqrt(np.mean(np.array(ey_list)**2))),
            'epsi_rms': float(np.sqrt(np.mean(np.array(epsi_list)**2))),
            'theta_rms': float(np.sqrt(np.mean(np.array(theta_list)**2))),
        })

    ret = np.mean([r['return'] for r in results])
    et_count = sum(r['early_term'] for r in results)
    ey_rms = np.mean([r['ey_rms'] for r in results])
    epsi_rms = np.mean([r['epsi_rms'] for r in results])
    theta_rms = np.mean([r['theta_rms'] for r in results])
    return {
        'return': ret,
        'ET': et_count,
        'ET_rate': et_count / n_episodes,
        'ey_rms': ey_rms,
        'epsi_rms': epsi_rms,
        'theta_rms': theta_rms,
    }


def collect_on_policy_data(env, agent, n_episodes=10, max_steps=1500):
    """Collect data using MPPI agent (on-policy)."""
    all_obs, all_action, all_next_obs = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep + 100)  # Different seeds from evaluation
        agent.reset_planning()

        for step in range(min(max_steps, env.max_episode_steps)):
            eps, _, _, _ = agent.act(obs, eval_mode=False)  # Training mode
            prev_obs = obs.copy()
            obs, r, t, tr, info = env.step(eps)

            all_obs.append(prev_obs)
            all_action.append(eps)
            all_next_obs.append(obs.copy())

            if t or tr:
                break

    return {
        'obs': np.array(all_obs, dtype=np.float32),
        'action': np.array(all_action, dtype=np.float32),
        'next_obs': np.array(all_next_obs, dtype=np.float32),
    }


def sweep_mppi_params(env, sindy_Xi, action_scale, base_cfg, baseline_return):
    """Sweep MPPI parameters."""
    print("\n" + "=" * 80)
    print("  Sweeping MPPI Parameters")
    print("=" * 80)

    best_advantage = float('-inf')
    best_params = None
    best_result = None

    # Parameter combinations to test
    param_configs = [
        # (horizon, num_samples, num_elites, iterations, temperature)
        (6, 64, 8, 4, 0.5),    # Current baseline
        (8, 64, 8, 4, 0.5),    # Longer horizon
        (10, 64, 8, 4, 0.5),   # Even longer horizon
        (6, 128, 16, 4, 0.5),  # More samples
        (6, 64, 8, 6, 0.5),    # More iterations
        (6, 64, 8, 4, 0.3),    # Lower temperature (more exploitation)
        (6, 64, 8, 4, 0.8),    # Higher temperature (more exploration)
        (8, 128, 16, 6, 0.5),  # Combined: longer + more samples + more iterations
    ]

    for i, (horizon, num_samples, num_elites, iterations, temperature) in enumerate(param_configs):
        cfg = base_cfg.copy()
        cfg.update({
            'horizon': horizon,
            'num_samples': num_samples,
            'num_elites': num_elites,
            'iterations': iterations,
            'temperature': temperature,
        })

        agent = ConservativeEnsembleResidualMPPI(
            cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)

        # Load trained ensemble
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'checkpoints', 'stage2_ensemble.pt')
        agent.load(ckpt_path)

        result = evaluate_mppi(env, agent, n_episodes=3, max_steps=1500)
        advantage = result['return'] - baseline_return

        print(f"  Config {i+1}: H={horizon}, S={num_samples}, E={num_elites}, "
              f"I={iterations}, T={temperature} | "
              f"return={result['return']:.2f}, adv={advantage:.2f}")

        if advantage > best_advantage:
            best_advantage = advantage
            best_params = {
                'horizon': horizon,
                'num_samples': num_samples,
                'num_elites': num_elites,
                'iterations': iterations,
                'temperature': temperature,
            }
            best_result = result

    return best_params, best_result, best_advantage


def sweep_epsilon_scaling(env, sindy_Xi, action_scale, base_cfg, baseline_return):
    """Sweep epsilon_max and alpha scaling."""
    print("\n" + "=" * 80)
    print("  Sweeping Epsilon Scaling")
    print("=" * 80)

    best_advantage = float('-inf')
    best_params = None
    best_result = None

    # Epsilon configurations
    epsilon_configs = [
        # (epsilon_max, alpha)
        (0.05, 0.3),   # Small residual
        (0.1, 0.3),    # Current
        (0.15, 0.3),   # Larger residual
        (0.2, 0.3),    # Even larger
        (0.1, 0.5),    # Higher alpha
        (0.1, 0.7),    # Even higher alpha
        (0.15, 0.5),   # Combined
    ]

    for i, (epsilon_max, alpha) in enumerate(epsilon_configs):
        cfg = base_cfg.copy()
        cfg.update({
            'epsilon_max': epsilon_max,
            'alpha': alpha,
        })

        agent = ConservativeEnsembleResidualMPPI(
            cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)

        # Load trained ensemble
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'checkpoints', 'stage2_ensemble.pt')
        agent.load(ckpt_path)

        result = evaluate_mppi(env, agent, n_episodes=3, max_steps=1500)
        advantage = result['return'] - baseline_return

        print(f"  Config {i+1}: eps_max={epsilon_max}, alpha={alpha} | "
              f"return={result['return']:.2f}, adv={advantage:.2f}")

        if advantage > best_advantage:
            best_advantage = advantage
            best_params = {
                'epsilon_max': epsilon_max,
                'alpha': alpha,
            }
            best_result = result

    return best_params, best_result, best_advantage


def sweep_reward_weights(env, sindy_Xi, action_scale, base_cfg, baseline_return):
    """Sweep reward weights."""
    print("\n" + "=" * 80)
    print("  Sweeping Reward Weights")
    print("=" * 80)

    best_advantage = float('-inf')
    best_params = None
    best_result = None

    # Reward weight configurations
    reward_configs = [
        # (lambda_res, lambda_smooth, lambda_uncertainty)
        (2.0, 5.0, 5.0),    # Current
        (1.0, 3.0, 5.0),    # Lower res weight
        (3.0, 5.0, 5.0),    # Higher res weight
        (2.0, 2.0, 5.0),    # Lower smooth weight
        (2.0, 5.0, 2.0),    # Lower uncertainty weight
        (2.0, 3.0, 3.0),    # Balanced lower
        (4.0, 8.0, 8.0),    # All higher
    ]

    for i, (lambda_res, lambda_smooth, lambda_uncertainty) in enumerate(reward_configs):
        cfg = base_cfg.copy()
        cfg.update({
            'lambda_res': lambda_res,
            'lambda_smooth': lambda_smooth,
            'lambda_uncertainty': lambda_uncertainty,
        })

        agent = ConservativeEnsembleResidualMPPI(
            cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)

        # Load trained ensemble
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'checkpoints', 'stage2_ensemble.pt')
        agent.load(ckpt_path)

        result = evaluate_mppi(env, agent, n_episodes=3, max_steps=1500)
        advantage = result['return'] - baseline_return

        print(f"  Config {i+1}: λ_res={lambda_res}, λ_smooth={lambda_smooth}, "
              f"λ_unc={lambda_uncertainty} | "
              f"return={result['return']:.2f}, adv={advantage:.2f}")

        if advantage > best_advantage:
            best_advantage = advantage
            best_params = {
                'lambda_res': lambda_res,
                'lambda_smooth': lambda_smooth,
                'lambda_uncertainty': lambda_uncertainty,
            }
            best_result = result

    return best_params, best_result, best_advantage


def test_on_policy_improvement(env, sindy_Xi, action_scale, base_cfg, baseline_return):
    """Test if on-policy data collection improves performance."""
    print("\n" + "=" * 80)
    print("  Testing On-Policy Data Collection")
    print("=" * 80)

    # Load current agent
    agent = ConservativeEnsembleResidualMPPI(
        base_cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'checkpoints', 'stage2_ensemble.pt')
    agent.load(ckpt_path)

    # Collect on-policy data
    print("  Collecting on-policy data (10 episodes)...")
    on_policy_data = collect_on_policy_data(env, agent, n_episodes=10, max_steps=1500)

    # Add to existing data
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'data', 'stage2_dataset_150k.npz')
    loaded = np.load(data_path, allow_pickle=True)

    # Combine data
    combined_obs = np.concatenate([loaded['obs'], on_policy_data['obs']])
    combined_action = np.concatenate([loaded['action'], on_policy_data['action']])
    combined_next_obs = np.concatenate([loaded['next_obs'], on_policy_data['next_obs']])

    combined_data = {
        'obs': combined_obs,
        'action': combined_action,
        'next_obs': combined_next_obs,
    }

    print(f"  Combined data: {len(combined_obs)} samples "
          f"({len(loaded['obs'])} off-policy + {len(on_policy_data['obs'])} on-policy)")

    # Retrain world model
    print("  Retraining world model with combined data...")
    buffer = SimpleBuffer(combined_data)

    # Create new agent for retraining
    new_agent = ConservativeEnsembleResidualMPPI(
        base_cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
    new_agent.train_world_model(buffer, epochs=30, batch_size=256, val_ratio=0.2)

    # Evaluate
    result = evaluate_mppi(env, new_agent, n_episodes=3, max_steps=1500)
    advantage = result['return'] - baseline_return

    print(f"  On-policy result: return={result['return']:.2f}, adv={advantage:.2f}")

    return result, advantage


def main():
    sys.stdout.reconfigure(line_buffering=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 80)
    print("  MPPI Optimization: Systematic Parameter Tuning")
    print(f"  Timestamp: {timestamp}")
    print("=" * 80)

    # ============================================================
    # Phase 0: Load Model and Environment
    # ============================================================
    print("\n" + "=" * 80)
    print("  Phase 0: Loading Model and Environment")
    print("=" * 80)

    sindy_data = np.load(os.path.join(base_dir, 'sindy_model_improved.npz'))
    sindy_Xi = sindy_data['coefficients']
    action_scale = float(sindy_data['action_scale'])

    # Load LQR config
    best_lqr_path = os.path.join(base_dir, 'configs', 'best_lqr.json')
    lqr_kwargs = {}
    if os.path.exists(best_lqr_path):
        with open(best_lqr_path) as f:
            best = json.load(f)
        lqr_kwargs = {'lqr_Q': best['Q_diag'], 'lqr_R': best['R_val']}

    env = PathTrackingEnv(
        max_episode_steps=1500,
        residual_injection='theta_target',
        **lqr_kwargs,
    )

    # Base MPPI config
    base_cfg = {
        'obs_dim': 8, 'action_dim': 1, 'mlp_dim': 256,
        'ensemble_size': 3,
        'horizon': 6, 'num_samples': 64, 'num_elites': 8,
        'iterations': 4, 'temperature': 0.5,
        'epsilon_max': 0.1, 'epsilon_std': 0.08,
        'gamma': 0.95,
        'lambda_res': 2.0, 'lambda_smooth': 5.0,
        'lambda_uncertainty': 5.0,
        'uncertainty_threshold': 1e6,  # Disable
        'risk_mode': 'hard_limit_only',
        'ey_tol': 0.1, 'theta_tol': 0.05,
        'margin': 0.0,  # Disable
    }

    # ============================================================
    # Phase 1: Baseline Evaluation
    # ============================================================
    print("\n" + "=" * 80)
    print("  Phase 1: Baseline Evaluation")
    print("=" * 80)

    baseline = evaluate_baseline(env, n_episodes=3, max_steps=1500)
    print(f"  Baseline: return={baseline['return']:.2f}, ET={baseline['ET']}/3, "
          f"ey_rms={baseline['ey_rms']:.4f}")

    # ============================================================
    # Phase 2: Sweep MPPI Parameters
    # ============================================================
    best_mppi_params, best_mppi_result, best_mppi_adv = sweep_mppi_params(
        env, sindy_Xi, action_scale, base_cfg, baseline['return'])

    # ============================================================
    # Phase 3: Sweep Epsilon Scaling
    # ============================================================
    best_eps_params, best_eps_result, best_eps_adv = sweep_epsilon_scaling(
        env, sindy_Xi, action_scale, base_cfg, baseline['return'])

    # ============================================================
    # Phase 4: Sweep Reward Weights
    # ============================================================
    best_reward_params, best_reward_result, best_reward_adv = sweep_reward_weights(
        env, sindy_Xi, action_scale, base_cfg, baseline['return'])

    # ============================================================
    # Phase 5: Test On-Policy Data Collection
    # ============================================================
    on_policy_result, on_policy_adv = test_on_policy_improvement(
        env, sindy_Xi, action_scale, base_cfg, baseline['return'])

    # ============================================================
    # Phase 6: Final Comparison
    # ============================================================
    print("\n" + "=" * 80)
    print("  Phase 6: Final Comparison")
    print("=" * 80)

    print(f"\n  Baseline: return={baseline['return']:.2f}")
    print(f"\n  Best MPPI Params: {best_mppi_params}")
    print(f"  Best MPPI: return={best_mppi_result['return']:.2f}, adv={best_mppi_adv:.2f}")
    print(f"\n  Best Epsilon Params: {best_eps_params}")
    print(f"  Best Epsilon: return={best_eps_result['return']:.2f}, adv={best_eps_adv:.2f}")
    print(f"\n  Best Reward Params: {best_reward_params}")
    print(f"  Best Reward: return={best_reward_result['return']:.2f}, adv={best_reward_adv:.2f}")
    print(f"\n  On-Policy: return={on_policy_result['return']:.2f}, adv={on_policy_adv:.2f}")

    # Find overall best
    all_results = [
        ('MPPI Params', best_mppi_adv, best_mppi_params),
        ('Epsilon Scaling', best_eps_adv, best_eps_params),
        ('Reward Weights', best_reward_adv, best_reward_params),
        ('On-Policy', on_policy_adv, {}),
    ]
    best_method = max(all_results, key=lambda x: x[1])

    print(f"\n  Overall Best: {best_method[0]}")
    print(f"  Advantage: {best_method[1]:.2f}")
    print(f"  Params: {best_method[2]}")

    # Save results
    results_path = os.path.join(base_dir, 'configs', 'mppi_optimization.json')
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'baseline': baseline,
            'best_mppi_params': best_mppi_params,
            'best_mppi_result': best_mppi_result,
            'best_mppi_advantage': float(best_mppi_adv),
            'best_eps_params': best_eps_params,
            'best_eps_result': best_eps_result,
            'best_eps_advantage': float(best_eps_adv),
            'best_reward_params': best_reward_params,
            'best_reward_result': best_reward_result,
            'best_reward_advantage': float(best_reward_adv),
            'on_policy_result': on_policy_result,
            'on_policy_advantage': float(on_policy_adv),
            'overall_best_method': best_method[0],
            'overall_best_advantage': float(best_method[1]),
        }, f, indent=2)

    print(f"\n  Saved results: {results_path}")
    print("=" * 80)

    env.close()


if __name__ == '__main__':
    main()
