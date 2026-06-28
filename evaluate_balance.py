"""
Evaluate trained SINDy-TD-MPC2 bicycle balance agent.

Generates:
1. Episode statistics (return, length, max tilt)
2. Balance trajectory plots
3. Comparison with random/LQR baselines
"""

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman', 'DejaVu Serif']
rcParams['font.serif'] = ['Times New Roman', 'SimSun', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sindy_env import SINDyBicycleEnv, _load_sindy_model
from sindy_tdmpc2 import SINDyTDMPC2


def run_episode(agent, env, max_steps=300):
    """Run one episode, record full trajectory."""
    obs, _ = env.reset()
    trajectory = {
        'obs': [obs.copy()],
        'actions': [],
        'rewards': [],
    }
    total_reward = 0

    for step in range(max_steps):
        action = agent.act(obs, t0=(step == 0), eval_mode=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        trajectory['obs'].append(obs.copy())
        trajectory['actions'].append(action.copy())
        trajectory['rewards'].append(reward)
        if terminated or truncated:
            break

    trajectory['obs'] = np.array(trajectory['obs'])
    trajectory['actions'] = np.array(trajectory['actions'])
    trajectory['rewards'] = np.array(trajectory['rewards'])
    trajectory['total_reward'] = total_reward
    trajectory['length'] = len(trajectory['rewards'])
    trajectory['terminated'] = terminated

    return trajectory


def random_policy(obs, env):
    return env.action_space.sample()


def lqr_policy(obs, env):
    """Simple LQR-like balance controller as baseline."""
    theta = obs[3] * 1.57  # denormalize
    theta_dot = obs[4] * 10.0
    delta = obs[6] * 0.785
    kp, kd = 8.0, 2.0
    u = -kp * theta - kd * theta_dot - 1.5 * delta
    return np.array([np.clip(u / 10, -1, 1)], dtype=np.float32)


def evaluate_and_plot(agent, fig_dir, n_episodes=20, max_steps=300):
    """Full evaluation with plots."""
    os.makedirs(fig_dir, exist_ok=True)
    env = SINDyBicycleEnv(max_episode_steps=max_steps)

    # Run episodes with SINDy-TD-MPC2
    trajectories = []
    for i in range(n_episodes):
        traj = run_episode(agent, env, max_steps)
        trajectories.append(traj)

    # Run random baseline
    random_returns = []
    random_lengths = []
    for _ in range(10):
        obs, _ = env.reset()
        total_r = 0
        for step in range(max_steps):
            obs, r, t, tr, _ = env.step(random_policy(obs, env))
            total_r += r
            if t or tr:
                break
        random_returns.append(total_r)
        random_lengths.append(step + 1)

    # Run LQR baseline
    lqr_returns = []
    lqr_lengths = []
    for _ in range(10):
        obs, _ = env.reset()
        total_r = 0
        for step in range(max_steps):
            obs, r, t, tr, _ = env.step(lqr_policy(obs, env))
            total_r += r
            if t or tr:
                break
        lqr_returns.append(total_r)
        lqr_lengths.append(step + 1)

    env.close()

    # Statistics
    agent_returns = [t['total_reward'] for t in trajectories]
    agent_lengths = [t['length'] for t in trajectories]

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n{'Policy':<20} {'Mean Return':>12} {'Std':>8} {'Mean Len':>10}")
    print("-" * 55)
    print(f"{'SINDy-TD-MPC2':<20} {np.mean(agent_returns):>12.2f} {np.std(agent_returns):>8.2f} {np.mean(agent_lengths):>10.1f}")
    print(f"{'LQR':<20} {np.mean(lqr_returns):>12.2f} {np.std(lqr_returns):>8.2f} {np.mean(lqr_lengths):>10.1f}")
    print(f"{'Random':<20} {np.mean(random_returns):>12.2f} {np.std(random_returns):>8.2f} {np.mean(random_lengths):>10.1f}")

    # Plot 1: Episode return distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SINDy-TD-MPC2 Balance Evaluation\n(SINDy-TD-MPC2 平衡评估)', fontsize=14)

    ax = axes[0, 0]
    ax.bar(['SINDy-TD-MPC2', 'LQR', 'Random'],
           [np.mean(agent_returns), np.mean(lqr_returns), np.mean(random_returns)],
           yerr=[np.std(agent_returns), np.std(lqr_returns), np.std(random_returns)],
           color=['steelblue', 'coral', 'gray'], capsize=5)
    ax.set_title('Mean Episode Return')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Episode length distribution
    ax = axes[0, 1]
    ax.bar(['SINDy-TD-MPC2', 'LQR', 'Random'],
           [np.mean(agent_lengths), np.mean(lqr_lengths), np.mean(random_lengths)],
           yerr=[np.std(agent_lengths), np.std(lqr_lengths), np.std(random_lengths)],
           color=['steelblue', 'coral', 'gray'], capsize=5)
    ax.set_title('Mean Episode Length')
    ax.set_ylabel('Steps')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Best episode trajectory (roll angle)
    ax = axes[1, 0]
    best_idx = np.argmax(agent_returns)
    best_traj = trajectories[best_idx]
    theta = best_traj['obs'][:, 3] * 1.57  # denormalize
    ax.plot(theta, 'b-', linewidth=1.5, label=f'Best ep (R={agent_returns[best_idx]:.1f})')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Termination (0.5 rad)')
    ax.axhline(y=-0.5, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Roll Angle Trajectory (Best Episode)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Roll angle (rad)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Action distribution
    ax = axes[1, 1]
    all_actions = np.concatenate([t['actions'].flatten() for t in trajectories])
    ax.hist(all_actions, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.set_title('Action Distribution')
    ax.set_xlabel('Action')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_balance_evaluation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: fig_balance_evaluation.png")

    # Plot 5: Detailed trajectory for best episode
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Best Episode Trajectory (Return={agent_returns[best_idx]:.2f}, Length={agent_lengths[best_idx]})\n'
                 f'(最优回合轨迹)', fontsize=13)

    t = np.arange(len(best_traj['obs']) - 1)

    # Roll angle
    theta = best_traj['obs'][:-1, 3] * 1.57
    axes[0].plot(t, theta, 'b-', linewidth=1.2)
    axes[0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_ylabel('Roll angle (rad)')
    axes[0].set_title('theta')
    axes[0].grid(True, alpha=0.3)

    # Roll rate
    theta_dot = best_traj['obs'][:-1, 4] * 10.0
    axes[1].plot(t, theta_dot, 'g-', linewidth=1.2)
    axes[1].set_ylabel('Roll rate (rad/s)')
    axes[1].set_title('theta_dot')
    axes[1].grid(True, alpha=0.3)

    # Actions
    axes[2].plot(t, best_traj['actions'].flatten(), 'r-', linewidth=1.2)
    axes[2].set_ylabel('Action')
    axes[2].set_title('Steering action')
    axes[2].grid(True, alpha=0.3)

    # Rewards
    axes[3].plot(t, best_traj['rewards'], 'm-', linewidth=1.2)
    axes[3].set_ylabel('Reward')
    axes[3].set_xlabel('Step')
    axes[3].set_title('Reward')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_best_trajectory.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_best_trajectory.png")

    return {
        'agent_returns': agent_returns,
        'agent_lengths': agent_lengths,
        'lqr_returns': lqr_returns,
        'random_returns': random_returns,
    }


def main():
    print("=" * 60)
    print("  SINDy-TD-MPC2 Balance Evaluation")
    print("=" * 60)

    # Load SINDy model
    sindy = _load_sindy_model()
    Xi = sindy['coefficients']
    action_scale = sindy['action_scale']

    # Create agent (same config as training)
    cfg = {
        'latent_dim': 64,
        'mlp_dim': 256,
        'lr': 3e-4,
        'horizon': 3,
        'num_samples': 64,
        'num_elites': 8,
        'num_pi_trajs': 8,
        'iterations': 6,
        'temperature': 0.5,
        'max_std': 0.5,
        'min_std': 0.05,
        'sindy_coef': 5.0,
        'tau': 0.005,
        'gamma': 0.99,
        'grad_clip': 10.0,
    }
    agent = SINDyTDMPC2(cfg, Xi, action_scale, obs_dim=8, action_dim=1)

    # Load checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoint_balance.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoint_balance_final.pt')
    if os.path.exists(ckpt_path):
        agent.load(ckpt_path)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("WARNING: No checkpoint found, evaluating untrained agent.")

    fig_dir = os.path.join(os.path.dirname(__file__), 'figures', 'balance')
    results = evaluate_and_plot(agent, fig_dir, n_episodes=20)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    return results


if __name__ == '__main__':
    main()
