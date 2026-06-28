"""Compare Stanley-only vs Stanley+RL path tracking performance.

Generates comparison plots and statistical analysis.
"""

import sys
import os
import numpy as np
import torch
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from path_tracking_env import PathTrackingEnv
from sindy_tdmpc2 import SINDyTDMPC2


def collect_episode(agent, seed, max_steps=500):
    """Run one episode, collect trajectory data."""
    env = PathTrackingEnv(max_episode_steps=max_steps)
    obs, _ = env.reset(seed=seed)

    trajectory = {
        'x': [], 'y': [], 'lateral_error': [], 'course_error': [],
        'theta': [], 'theta_dot': [], 'curvature': [], 'reward': [],
        'target_roll': [],
    }
    total_r = 0

    for step in range(max_steps):
        if agent is not None:
            action = agent.act(obs, t0=(step == 0), eval_mode=True)
        else:
            action = np.array([0.0])

        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward

        path_info = info.get('path_info', {})
        trajectory['x'].append(env._x)
        trajectory['y'].append(env._y)
        trajectory['lateral_error'].append(path_info.get('lateral_error', 0))
        trajectory['course_error'].append(path_info.get('course_error_angle', 0))
        trajectory['theta'].append(env._theta)
        trajectory['theta_dot'].append(env._theta_dot)
        trajectory['curvature'].append(path_info.get('curvature', 0))
        trajectory['reward'].append(reward)
        trajectory['target_roll'].append(info.get('target_roll', 0))

        if terminated or truncated:
            break

    env.close()
    trajectory['total_return'] = total_r
    trajectory['length'] = step + 1
    return trajectory


def draw_complex_path(ax):
    """Draw the reference path on a matplotlib axis."""
    ax.plot([0, 55], [0, 0], 'k--', linewidth=1.5, alpha=0.5)
    angles2 = np.linspace(-math.pi/2, 0, 50)
    ax.plot(55 + 15*np.cos(angles2), 15 + 15*np.sin(angles2), 'k--', linewidth=1.5, alpha=0.5)
    ax.plot([70, 70], [15, 35], 'k--', linewidth=1.5, alpha=0.5)
    angles4 = np.linspace(0, math.pi/2, 50)
    ax.plot(55 + 15*np.cos(angles4), 35 + 15*np.sin(angles4), 'k--', linewidth=1.5, alpha=0.5)
    angles5 = np.linspace(math.pi/2, math.pi, 50)
    ax.plot(28 + 12*np.cos(angles5), 35 + 12*np.sin(angles5), 'k--', linewidth=1.5, alpha=0.5)
    ax.plot([28, 0], [23, 23], 'k--', linewidth=1.5, alpha=0.5)


def run_comparison():
    sys.stdout.reconfigure(line_buffering=True)

    print("Loading Stage 2 agent...")
    stage2_cfg = {
        'latent_dim': 64, 'mlp_dim': 256, 'lr': 3e-4,
        'horizon': 5, 'num_samples': 64, 'num_elites': 8,
        'num_pi_trajs': 8, 'iterations': 6, 'temperature': 0.5,
        'max_std': 0.5, 'min_std': 0.05, 'sindy_coef': 0.0,
        'tau': 0.005, 'gamma': 0.99, 'grad_clip': 10.0,
    }
    action_scale = 1.0
    dummy_Xi = np.zeros((55, 6), dtype=np.float32)
    rl_agent = SINDyTDMPC2(stage2_cfg, dummy_Xi, action_scale, obs_dim=6, action_dim=1)
    stage2_ckpt = os.path.join(os.path.dirname(__file__), 'checkpoint_path_tracking.pt')
    if os.path.exists(stage2_ckpt):
        rl_agent.load(stage2_ckpt)
        print("  Stage 2 agent loaded.")
    else:
        print("  WARNING: No Stage 2 checkpoint found!")
        return

    n_episodes = 50
    max_steps = 500

    print(f"\nRunning {n_episodes} episodes each...")
    stanley_returns, rl_returns = [], []
    stanley_lengths, rl_lengths = [], []
    stanley_lat_errors, rl_lat_errors = [], []

    for ep in range(n_episodes):
        traj_s = collect_episode(None, seed=ep, max_steps=max_steps)
        stanley_returns.append(traj_s['total_return'])
        stanley_lengths.append(traj_s['length'])
        stanley_lat_errors.append(np.mean(np.abs(traj_s['lateral_error'])))

        traj_r = collect_episode(rl_agent, seed=ep, max_steps=max_steps)
        rl_returns.append(traj_r['total_return'])
        rl_lengths.append(traj_r['length'])
        rl_lat_errors.append(np.mean(np.abs(traj_r['lateral_error'])))

        if ep % 10 == 0:
            print(f"  Episode {ep}: Stanley={traj_s['total_return']:.2f}, RL={traj_r['total_return']:.2f}")

    # Statistics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nStanley-only:")
    print(f"  Return: {np.mean(stanley_returns):.2f} +/- {np.std(stanley_returns):.2f}")
    print(f"  Length: {np.mean(stanley_lengths):.1f} +/- {np.std(stanley_lengths):.1f}")
    print(f"  Lateral Error: {np.mean(stanley_lat_errors):.3f} +/- {np.std(stanley_lat_errors):.3f}")

    print(f"\nStanley + RL:")
    print(f"  Return: {np.mean(rl_returns):.2f} +/- {np.std(rl_returns):.2f}")
    print(f"  Length: {np.mean(rl_lengths):.1f} +/- {np.std(rl_lengths):.1f}")
    print(f"  Lateral Error: {np.mean(rl_lat_errors):.3f} +/- {np.std(rl_lat_errors):.3f}")

    from scipy import stats
    t_ret, p_ret = stats.ttest_rel(rl_returns, stanley_returns)
    t_lat, p_lat = stats.ttest_rel(stanley_lat_errors, rl_lat_errors)

    print(f"\nPaired t-test (return): t={t_ret:.3f}, p={p_ret:.4f}")
    if p_ret < 0.05:
        print(f"  -> RL is {'better' if t_ret > 0 else 'worse'} than Stanley (p<0.05)")
    else:
        print(f"  -> No significant difference (p>=0.05)")

    print(f"Paired t-test (lateral error): t={t_lat:.3f}, p={p_lat:.4f}")
    if p_lat < 0.05:
        print(f"  -> RL has {'lower' if t_lat > 0 else 'higher'} lateral error (p<0.05)")
    else:
        print(f"  -> No significant difference (p>=0.05)")

    # Generate plots
    print("\nGenerating comparison plots...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: Trajectories
        ax = axes[0, 0]
        draw_complex_path(ax)
        for seed in [0, 1, 2]:
            traj_s = collect_episode(None, seed=seed, max_steps=max_steps)
            ax.plot(traj_s['x'], traj_s['y'], 'b-', alpha=0.6, linewidth=1)
            traj_r = collect_episode(rl_agent, seed=seed, max_steps=max_steps)
            ax.plot(traj_r['x'], traj_r['y'], 'r-', alpha=0.6, linewidth=1)
        ax.plot([], [], 'b-', label='Stanley')
        ax.plot([], [], 'r-', label='Stanley+RL')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectory Comparison')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Plot 2: Lateral error over time
        ax = axes[0, 1]
        traj_s = collect_episode(None, seed=0, max_steps=max_steps)
        traj_r = collect_episode(rl_agent, seed=0, max_steps=max_steps)
        t_s = np.arange(len(traj_s['lateral_error'])) * (1/30)
        t_r = np.arange(len(traj_r['lateral_error'])) * (1/30)
        ax.plot(t_s, np.abs(traj_s['lateral_error']), 'b-', alpha=0.7, label='Stanley')
        ax.plot(t_r, np.abs(traj_r['lateral_error']), 'r-', alpha=0.7, label='Stanley+RL')
        ax.axhline(y=0.05, color='g', linestyle='--', alpha=0.5, label='On-path')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('|Lateral Error| (m)')
        ax.set_title('Lateral Error Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Roll angle
        ax = axes[0, 2]
        ax.plot(t_s, np.array(traj_s['theta'])*180/math.pi, 'b-', alpha=0.7, label='Stanley')
        ax.plot(t_r, np.array(traj_r['theta'])*180/math.pi, 'r-', alpha=0.7, label='Stanley+RL')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Roll Angle (deg)')
        ax.set_title('Roll Angle')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Return boxplot
        ax = axes[1, 0]
        bp = ax.boxplot([stanley_returns, rl_returns], labels=['Stanley', 'Stanley+RL'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Episode Return')
        ax.set_title(f'Return (p={p_ret:.4f})')
        ax.grid(True, alpha=0.3)

        # Plot 5: Lateral error boxplot
        ax = axes[1, 1]
        bp = ax.boxplot([stanley_lat_errors, rl_lat_errors], labels=['Stanley', 'Stanley+RL'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Mean |Lateral Error| (m)')
        ax.set_title(f'Lateral Error (p={p_lat:.4f})')
        ax.grid(True, alpha=0.3)

        # Plot 6: Episode length
        ax = axes[1, 2]
        bp = ax.boxplot([stanley_lengths, rl_lengths], labels=['Stanley', 'Stanley+RL'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.set_ylabel('Episode Length (steps)')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(os.path.dirname(__file__), 'comparison_path_tracking.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

        # Reward plot
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
        traj_s = collect_episode(None, seed=0, max_steps=max_steps)
        traj_r = collect_episode(rl_agent, seed=0, max_steps=max_steps)
        window = 30
        r_s = np.convolve(traj_s['reward'], np.ones(window)/window, mode='valid')
        r_r = np.convolve(traj_r['reward'], np.ones(window)/window, mode='valid')
        t_s2 = np.arange(len(r_s)) * (1/30)
        t_r2 = np.arange(len(r_r)) * (1/30)
        ax2.plot(t_s2, r_s, 'b-', alpha=0.7, label='Stanley')
        ax2.plot(t_r2, r_r, 'r-', alpha=0.7, label='Stanley+RL')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Smoothed Reward')
        ax2.set_title('Reward Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        save_path2 = os.path.join(os.path.dirname(__file__), 'reward_comparison.png')
        plt.savefig(save_path2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path2}")

    except ImportError as e:
        print(f"  matplotlib not available: {e}")

    print("\nDone!")


if __name__ == '__main__':
    run_comparison()
