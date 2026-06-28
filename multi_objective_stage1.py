"""Stage 1: Multi-objective optimization for attitude control.

Objectives:
  1. Stability: small theta (tilt angle)
  2. Energy efficiency: small actions
  3. Smoothness: small action changes
  4. Angular velocity: small theta_dot

Key insight: LQR optimizes a single objective (quadratic cost).
We can find a better trade-off by optimizing multiple objectives simultaneously.
"""
import sys, os, json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attitude_control_env import AttitudeControlEnv
from residual_mppi import ConservativeEnsembleResidualMPPI


class MultiObjectiveReward:
    """Multi-objective reward function for attitude control.

    Balances:
      - stability: -|theta|
      - energy: -|action|
      - smoothness: -|action - prev_action|
      - angular_velocity: -|theta_dot|
    """
    def __init__(self, w_stability=1.0, w_energy=0.1, w_smoothness=0.5, w_angular_vel=0.01):
        self.w_stability = w_stability
        self.w_energy = w_energy
        self.w_smoothness = w_smoothness
        self.w_angular_vel = w_angular_vel
        self.prev_action = 0.0

    def reset(self):
        self.prev_action = 0.0

    def compute(self, raw_state, action, terminated=False):
        """Compute multi-objective reward.

        Args:
            raw_state: [ey, epsi, v, theta, theta_dot, k, delta, delta_dot]
            action: scalar action value
            terminated: whether episode terminated early

        Returns:
            reward: scalar
        """
        theta = raw_state[3]
        theta_dot = raw_state[4]

        if terminated:
            return -10.0  # Large penalty for termination

        # Stability: penalize tilt
        r_stability = -abs(theta)

        # Energy: penalize large actions
        r_energy = -abs(action)

        # Smoothness: penalize action changes
        r_smoothness = -abs(action - self.prev_action)

        # Angular velocity: penalize fast rotation
        r_angular_vel = -abs(theta_dot)

        # Combined reward
        reward = (self.w_stability * r_stability +
                  self.w_energy * r_energy +
                  self.w_smoothness * r_smoothness +
                  self.w_angular_vel * r_angular_vel)

        self.prev_action = action
        return reward


def evaluate_with_reward(env, agent, reward_fn, n_episodes=5, max_steps=200):
    """Evaluate agent with multi-objective reward."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        reward_fn.reset()
        agent.reset_planning()
        total_r = 0
        et = False
        theta_list = []
        action_list = []
        smoothness_list = []

        for step in range(min(max_steps, env.max_episode_steps)):
            eps, best_G, zero_G, unc = agent.act(obs, eval_mode=True)
            obs, r_env, t, tr, info = env.step(eps)

            # Compute multi-objective reward
            r_multi = reward_fn.compute(info['raw_state'], float(eps[0]), t)
            total_r += r_multi

            theta_list.append(abs(info['raw_state'][3]))
            action_list.append(abs(float(eps[0])))
            if step > 0:
                smoothness_list.append(abs(float(eps[0]) - float(eps_prev[0])))
            eps_prev = eps

            if t:
                et = True
            if t or tr:
                break

        results.append({
            'return': total_r,
            'length': step + 1,
            'early_term': et,
            'theta_rms': float(np.sqrt(np.mean(np.array(theta_list)**2))),
            'mean_action': float(np.mean(action_list)),
            'mean_smoothness': float(np.mean(smoothness_list)) if smoothness_list else 0.0,
        })

    ret = np.mean([r['return'] for r in results])
    et_count = sum(r['early_term'] for r in results)
    theta_rms = np.mean([r['theta_rms'] for r in results])
    mean_action = np.mean([r['mean_action'] for r in results])
    mean_smoothness = np.mean([r['mean_smoothness'] for r in results])
    return {
        'return': ret,
        'ET': et_count,
        'ET_rate': et_count / n_episodes,
        'theta_rms': theta_rms,
        'mean_action': mean_action,
        'mean_smoothness': mean_smoothness,
    }


def evaluate_baseline(env, reward_fn, n_episodes=5, max_steps=200):
    """Evaluate zero-residual baseline with multi-objective reward."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        reward_fn.reset()
        total_r = 0
        et = False
        theta_list = []

        for step in range(min(max_steps, env.max_episode_steps)):
            action = np.array([0.0])
            obs, r_env, t, tr, info = env.step(action)

            # Compute multi-objective reward
            r_multi = reward_fn.compute(info['raw_state'], 0.0, t)
            total_r += r_multi

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


def sweep_reward_weights(env, agent, n_episodes=3, max_steps=200):
    """Sweep different reward weight combinations."""
    print("\n  Sweeping reward weights...")

    best_advantage = float('-inf')
    best_weights = None
    best_result = None

    # Weight combinations to try
    weight_configs = [
        # (w_stability, w_energy, w_smoothness, w_angular_vel)
        (1.0, 0.1, 0.5, 0.01),   # Balanced
        (1.0, 0.01, 0.1, 0.01),  # Stability-focused
        (1.0, 0.5, 1.0, 0.01),   # Energy+smoothness focused
        (0.5, 0.1, 0.5, 0.1),    # Angular velocity focused
        (2.0, 0.05, 0.2, 0.005), # High stability weight
        (1.0, 0.2, 0.2, 0.05),   # Equal weights
    ]

    for i, (w_s, w_e, w_sm, w_av) in enumerate(weight_configs):
        reward_fn = MultiObjectiveReward(
            w_stability=w_s, w_energy=w_e,
            w_smoothness=w_sm, w_angular_vel=w_av
        )

        baseline = evaluate_baseline(env, reward_fn, n_episodes=n_episodes, max_steps=max_steps)
        mppi_result = evaluate_with_reward(env, agent, reward_fn, n_episodes=n_episodes, max_steps=max_steps)

        advantage = mppi_result['return'] - baseline['return']
        print(f"  Config {i+1}: w=({w_s},{w_e},{w_sm},{w_av}) | "
              f"baseline={baseline['return']:.3f}, mppi={mppi_result['return']:.3f}, "
              f"adv={advantage:.3f}")

        if advantage > best_advantage:
            best_advantage = advantage
            best_weights = (w_s, w_e, w_sm, w_av)
            best_result = {
                'baseline': baseline,
                'mppi': mppi_result,
                'advantage': advantage,
            }

    return best_weights, best_result


def main():
    sys.stdout.reconfigure(line_buffering=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 80)
    print("  Stage 1: Multi-objective Optimization")
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

    env = AttitudeControlEnv(max_episode_steps=1000, **lqr_kwargs)

    # Load MPPI agent
    mppi_cfg = {
        'obs_dim': 8, 'action_dim': 1, 'mlp_dim': 256,
        'ensemble_size': 3,
        'horizon': 6, 'num_samples': 64, 'num_elites': 8,
        'iterations': 4, 'temperature': 0.5,
        'epsilon_max': 0.1,
    }
    agent = ConservativeEnsembleResidualMPPI(
        mppi_cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
    agent.load(os.path.join(base_dir, 'checkpoints', 'stage1_ensemble.pt'))
    print("  Loaded MPPI agent")

    # ============================================================
    # Phase 1: Sweep Reward Weights
    # ============================================================
    print("\n" + "=" * 80)
    print("  Phase 1: Sweeping Reward Weights")
    print("=" * 80)

    best_weights, best_result = sweep_reward_weights(env, agent, n_episodes=3, max_steps=200)

    print(f"\n  Best weights: {best_weights}")
    print(f"  Best advantage: {best_result['advantage']:.3f}")

    # ============================================================
    # Phase 2: Detailed Evaluation with Best Weights
    # ============================================================
    print("\n" + "=" * 80)
    print("  Phase 2: Detailed Evaluation with Best Weights")
    print("=" * 80)

    reward_fn = MultiObjectiveReward(
        w_stability=best_weights[0],
        w_energy=best_weights[1],
        w_smoothness=best_weights[2],
        w_angular_vel=best_weights[3],
    )

    # Run longer evaluation
    baseline = evaluate_baseline(env, reward_fn, n_episodes=5, max_steps=500)
    mppi_result = evaluate_with_reward(env, agent, reward_fn, n_episodes=5, max_steps=500)
    advantage = mppi_result['return'] - baseline['return']

    print(f"  Baseline: return={baseline['return']:.3f}, ET={baseline['ET']}/5, "
          f"theta_rms={baseline['theta_rms']:.4f}")
    print(f"  MPPI: return={mppi_result['return']:.3f}, ET={mppi_result['ET']}/5, "
          f"theta_rms={mppi_result['theta_rms']:.4f}, "
          f"mean_action={mppi_result['mean_action']:.4f}, "
          f"smoothness={mppi_result['mean_smoothness']:.4f}")
    print(f"  Advantage: {advantage:.3f}")

    # ============================================================
    # Phase 3: Save Results
    # ============================================================
    print("\n" + "=" * 80)
    print("  Phase 3: Save Results")
    print("=" * 80)

    results_path = os.path.join(base_dir, 'configs', 'stage1_multi_objective.json')
    with open(results_path, 'w') as f:
        json.dump({
            'best_weights': list(best_weights),
            'baseline': baseline,
            'mppi': mppi_result,
            'advantage': float(advantage),
            'timestamp': timestamp,
        }, f, indent=2)

    print(f"  Saved results: {results_path}")
    print(f"  Advantage: {advantage:.3f}")
    print("=" * 80)

    env.close()


if __name__ == '__main__':
    main()
