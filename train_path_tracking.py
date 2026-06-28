"""Training script: SINDy-TD-MPC2 for Stage 2 path tracking.

Architecture:
  Stanley controller → steering angle (baseline)
  RL residual → small correction on steering angle
  Kinematic bicycle model → position update

State (6D): [lateral_error, course_error_angle, v, theta, theta_dot, curvature]
Action (1D): RL residual [-1, 1]
"""

import sys
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from path_tracking_env import PathTrackingEnv
from sindy_tdmpc2 import SINDyTDMPC2


class ReplayBuffer:
    def __init__(self, capacity=200000, obs_dim=6, action_dim=1):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, terminated):
        self.obs[self.idx] = obs
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.terminated[self.idx] = terminated
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idx]),
            torch.tensor(self.action[idx]),
            torch.tensor(self.reward[idx]),
            torch.tensor(self.next_obs[idx]),
            torch.tensor(self.terminated[idx]),
        )


def evaluate(agent, n_episodes=10, max_steps=500):
    env = PathTrackingEnv(max_episode_steps=max_steps)
    returns, lengths, lat_errors = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total_r = 0
        ep_lat = []
        for step in range(max_steps):
            action = agent.act(obs, t0=(step == 0), eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            ep_lat.append(abs(info.get('path_info', {}).get('lateral_error', 0)))
            if terminated or truncated:
                break
        returns.append(total_r)
        lengths.append(step + 1)
        lat_errors.append(np.mean(ep_lat))

    env.close()
    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_length': np.mean(lengths),
        'mean_lateral_error': np.mean(lat_errors),
    }


def evaluate_stanley_only(n_episodes=10, max_steps=500):
    env = PathTrackingEnv(max_episode_steps=max_steps)
    returns, lengths, lat_errors = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total_r = 0
        ep_lat = []
        for step in range(max_steps):
            action = np.array([0.0])
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            ep_lat.append(abs(info.get('path_info', {}).get('lateral_error', 0)))
            if terminated or truncated:
                break
        returns.append(total_r)
        lengths.append(step + 1)
        lat_errors.append(np.mean(ep_lat))

    env.close()
    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_length': np.mean(lengths),
        'mean_lateral_error': np.mean(lat_errors),
    }


def train():
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 60)
    print("  SINDy-TD-MPC2 Path Tracking Training (Stage 2)")
    print("=" * 60)

    # Stage 2 config (6D state)
    cfg = {
        'latent_dim': 64,
        'mlp_dim': 256,
        'lr': 3e-4,
        'horizon': 5,
        'num_samples': 64,
        'num_elites': 8,
        'num_pi_trajs': 8,
        'iterations': 6,
        'temperature': 0.5,
        'max_std': 0.5,
        'min_std': 0.05,
        'sindy_coef': 0.0,  # No SINDy for 6D (mismatched with 8D model)
        'tau': 0.005,
        'gamma': 0.95,
        'grad_clip': 10.0,
    }

    total_steps = 200000
    batch_size = 128
    seed_steps = 3000
    eval_every = 10000
    log_every = 500
    save_every = 50000

    # Create agent
    print("\nCreating Stage 2 agent...")
    action_scale = 1.0
    dummy_Xi = np.zeros((55, 6), dtype=np.float32)
    agent = SINDyTDMPC2(cfg, dummy_Xi, action_scale, obs_dim=6, action_dim=1)

    env = PathTrackingEnv(max_episode_steps=500)
    buffer = ReplayBuffer(capacity=50000, obs_dim=6, action_dim=1)

    ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoint_path_tracking.pt')
    if os.path.exists(ckpt_path):
        agent.load(ckpt_path)
        print(f"  Loaded checkpoint from {ckpt_path}")

    # Baseline
    print("\nEvaluating Stanley-only baseline...")
    stanley_stats = evaluate_stanley_only(n_episodes=10)
    print(f"  Stanley-only: return={stanley_stats['mean_return']:.2f} +/- {stanley_stats['std_return']:.2f}")
    print(f"  Mean lateral error: {stanley_stats['mean_lateral_error']:.3f}")

    print(f"\nTraining config:")
    print(f"  Total steps: {total_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Horizon: {cfg['horizon']}")
    print(f"  Device: {agent.device}")

    print("\n" + "=" * 60)
    print("Pre-filling buffer with Stanley-only data...")
    print("=" * 60)

    # Pre-fill buffer with Stanley-only (zero action) data
    # This teaches Q-network that zero action → positive returns
    pre_fill_steps = 5000
    obs, _ = env.reset()
    for i in range(pre_fill_steps):
        action = np.array([0.0], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    print(f"  Pre-filled {buffer.size} Stanley-only transitions")

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    obs, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    episode_returns = []
    t_start = time.time()
    best_eval_return = -np.inf
    no_improve_count = 0

    for step in range(1, total_steps + 1):
        if step < 1000:
            # Small random exploration (don't disturb Stanley too much)
            action = np.random.uniform(-0.2, 0.2, size=(1,)).astype(np.float32)
        elif step < seed_steps:
            # Mixed: policy + noise (helps discover non-zero actions are useful)
            action = agent.act_policy(obs)
            noise = np.random.randn(*action.shape) * 0.2
            action = np.clip(action + noise, -1.0, 1.0).astype(np.float32)
        else:
            action = agent.act_policy(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, float(terminated))

        episode_reward += reward
        obs = next_obs

        if terminated or truncated:
            episode_returns.append(episode_reward)
            episode_count += 1
            obs, _ = env.reset()
            episode_reward = 0

        if step >= seed_steps and buffer.size >= batch_size:
            batch = buffer.sample(batch_size)
            losses = agent.update(batch, step=step)

            if step % log_every == 0:
                elapsed = time.time() - t_start
                sps = step / elapsed
                avg_ret = np.mean(episode_returns[-20:]) if episode_returns else 0
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    mean, log_std = agent.pi(obs_t)
                    q1 = agent.Q1(obs_t, mean)
                    q_zero = agent.Q1(obs_t, torch.zeros_like(mean))
                print(f"Step {step:>7d}/{total_steps} | "
                      f"Ep {episode_count:>4d} | "
                      f"AvgRet {avg_ret:>8.2f} | "
                      f"Loss {losses['total_loss']:.4f} | "
                      f"Q {losses['q_loss']:.4f} | "
                      f"Q_act({q1.item():.3f}) | "
                      f"Q_zero({q_zero.item():.3f}) | "
                      f"pi({mean.item():.3f},{log_std.item():.3f}) | "
                      f"SPS {sps:.0f}")

        if step % eval_every == 0:
            eval_stats = evaluate(agent, n_episodes=10)
            print(f"\n  [Eval @ step {step}] "
                  f"Return: {eval_stats['mean_return']:.2f} +/- {eval_stats['std_return']:.2f} | "
                  f"Length: {eval_stats['mean_length']:.1f} | "
                  f"LatErr: {eval_stats['mean_lateral_error']:.3f}")
            print(f"  [Stanley baseline] "
                  f"Return: {stanley_stats['mean_return']:.2f} +/- {stanley_stats['std_return']:.2f}")
            if eval_stats['mean_return'] > best_eval_return:
                best_eval_return = eval_stats['mean_return']
                agent.save(ckpt_path)
                no_improve_count = 0
                print(f"  [Checkpoint saved]")
            else:
                no_improve_count += 1
                print(f"  [No improvement for {no_improve_count} evals]")
            print()

        if step % save_every == 0:
            agent.save(os.path.join(os.path.dirname(__file__), f'checkpoint_path_tracking_{step}.pt'))

    agent.save(os.path.join(os.path.dirname(__file__), 'checkpoint_path_tracking_final.pt'))
    env.close()

    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_eval = evaluate(agent, n_episodes=20)
    print(f"  RL+Stanley: return={final_eval['mean_return']:.2f} +/- {final_eval['std_return']:.2f}")
    print(f"  Mean lateral error: {final_eval['mean_lateral_error']:.3f}")
    print(f"  Mean length: {final_eval['mean_length']:.1f}")
    print(f"\n  Stanley baseline: return={stanley_stats['mean_return']:.2f} +/- {stanley_stats['std_return']:.2f}")
    print(f"  Stanley lateral error: {stanley_stats['mean_lateral_error']:.3f}")
    print(f"\n  Best eval return: {best_eval_return:.2f}")
    print(f"  Training time: {(time.time() - t_start) / 60:.1f} min")
    print("=" * 60)

    return agent


if __name__ == '__main__':
    train()
