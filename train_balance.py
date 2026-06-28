"""
Training script: SINDy-TD-MPC2 for bicycle balancing (residual RL on LQR baseline).

Uses the analytical bicycle environment with LQR baseline.
The RL agent learns a residual steering correction on top of LQR.
SINDy prior regularizes the learned world model.
"""

import sys
import os
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bicycle_env_analytical import AnalyticalBicycleEnv
from sindy_env import _load_sindy_model
from sindy_tdmpc2 import SINDyTDMPC2


class ReplayBuffer:
    """Simple circular replay buffer."""

    def __init__(self, capacity=100000, obs_dim=8, action_dim=1):
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


def evaluate(agent, n_episodes=10, max_steps=800):
    """Evaluate agent performance using MPPI planning."""
    env = AnalyticalBicycleEnv(max_episode_steps=max_steps)
    returns = []
    lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = agent.act(obs, t0=(step == 0), eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        returns.append(total_reward)
        lengths.append(step + 1)

    env.close()
    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_length': np.mean(lengths),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
    }


def train():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 60)
    print("  SINDy-TD-MPC2 Bicycle Balance Training")
    print("=" * 60)

    # Load SINDy model
    print("\nLoading SINDy model...")
    sindy = _load_sindy_model()
    Xi = sindy['coefficients']
    action_scale = sindy['action_scale']
    print(f"  Coefficient matrix: {Xi.shape}")
    print(f"  Active terms: {np.sum(np.abs(Xi) > 1e-6)}")

    # Config
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

    total_steps = 100000
    batch_size = 256
    seed_steps = 2000  # More random exploration before training
    eval_every = 5000
    log_every = 500
    save_every = 20000

    # Setup
    print("\nCreating agent and environment...")
    agent = SINDyTDMPC2(cfg, Xi, action_scale, obs_dim=8, action_dim=1)
    env = AnalyticalBicycleEnv(max_episode_steps=800)
    buffer = ReplayBuffer(capacity=100000, obs_dim=8, action_dim=1)

    # Load existing checkpoint if available
    ckpt_path = os.path.join(os.path.dirname(__file__), 'checkpoint_balance.pt')
    if os.path.exists(ckpt_path):
        agent.load(ckpt_path)
        print(f"  Loaded checkpoint from {ckpt_path}")

    print(f"\nTraining config:")
    print(f"  Total steps: {total_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Seed steps (random): {seed_steps}")
    print(f"  SINDy coefficient: {cfg['sindy_coef']}")
    print(f"  Latent dim: {cfg['latent_dim']}")
    print(f"  Device: {agent.device}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    obs, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    episode_returns = []
    t_start = time.time()
    best_eval_return = -np.inf

    for step in range(1, total_steps + 1):
        # Select action (policy-only during training, MPPI for eval)
        if step < seed_steps:
            action = env.action_space.sample()
        else:
            action = agent.act_policy(obs)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, float(terminated))

        episode_reward += reward
        obs = next_obs

        if terminated or truncated:
            episode_returns.append(episode_reward)
            episode_count += 1
            obs, _ = env.reset()
            episode_reward = 0

        # Train
        if step >= seed_steps and buffer.size >= batch_size:
            batch = buffer.sample(batch_size)
            losses = agent.update(batch, step=step)

            if step % log_every == 0:
                elapsed = time.time() - t_start
                sps = step / elapsed
                avg_ret = np.mean(episode_returns[-20:]) if episode_returns else 0
                # Diagnostic: check Q-values and policy output
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    mean, log_std = agent.pi(obs_t)
                    q1 = agent.Q1(obs_t, mean)
                    q_zero = agent.Q1(obs_t, torch.zeros_like(mean))
                print(f"Step {step:>6d}/{total_steps} | "
                      f"Ep {episode_count:>3d} | "
                      f"AvgRet {avg_ret:>7.2f} | "
                      f"Loss {losses['total_loss']:.4f} | "
                      f"Q {losses['q_loss']:.4f} | "
                      f"Q_act({q1.item():.3f}) | "
                      f"Q_zero({q_zero.item():.3f}) | "
                      f"pi({mean.item():.3f},{log_std.item():.3f}) | "
                      f"SPS {sps:.0f}")

        # Evaluate
        if step % eval_every == 0:
            eval_stats = evaluate(agent, n_episodes=10)
            print(f"\n  [Eval @ step {step}] "
                  f"Return: {eval_stats['mean_return']:.2f} +/- {eval_stats['std_return']:.2f} | "
                  f"Length: {eval_stats['mean_length']:.1f}")
            if eval_stats['mean_return'] > best_eval_return:
                best_eval_return = eval_stats['mean_return']
                agent.save(ckpt_path)
                print(f"  [Checkpoint saved: {ckpt_path}]")
            print()

        # Save periodic checkpoint
        if step % save_every == 0:
            agent.save(os.path.join(os.path.dirname(__file__), f'checkpoint_balance_{step}.pt'))

    # Final save
    agent.save(os.path.join(os.path.dirname(__file__), 'checkpoint_balance_final.pt'))
    env.close()

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_eval = evaluate(agent, n_episodes=20)
    print(f"  Mean return: {final_eval['mean_return']:.2f} +/- {final_eval['std_return']:.2f}")
    print(f"  Mean episode length: {final_eval['mean_length']:.1f}")
    print(f"  Best eval return: {best_eval_return:.2f}")
    print(f"  Training time: {(time.time() - t_start) / 60:.1f} min")
    print("=" * 60)

    return agent


if __name__ == '__main__':
    train()
