"""Stage 1: Dyna-style model-based residual policy learning.

Approach:
  1. Use SINDy world model to generate synthetic data
  2. Train residual policy network on synthetic data
  3. Evaluate vs LQR baseline

Key insight: LQR is already near-optimal for attitude control.
We learn a small residual to handle nonlinearities LQR can't capture.
"""
import sys, os, json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attitude_control_env import AttitudeControlEnv
from residual_mppi import EnsembleWorldModel


class ResidualPolicy(nn.Module):
    """Small residual policy network.

    Input: 8D normalized state
    Output: 1D residual action [-1, 1]
    """
    def __init__(self, obs_dim=8, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, obs):
        return self.net(obs)


def generate_synthetic_data(world_model, n_samples=100000, epsilon_range=0.1):
    """Generate synthetic transitions using world model.

    Args:
        world_model: Trained ensemble world model
        n_samples: Number of samples to generate
        epsilon_range: Range of random residual actions

    Returns:
        dict with obs, action, next_obs arrays
    """
    print(f"  Generating {n_samples} synthetic samples...")

    # Sample random states from uniform distribution (normalized)
    obs = np.random.uniform(-1, 1, (n_samples, 8)).astype(np.float32)

    # Sample random actions
    actions = np.random.uniform(-epsilon_range, epsilon_range, (n_samples, 1)).astype(np.float32)

    # Convert to tensors
    obs_t = torch.tensor(obs, device=world_model.device)
    act_t = torch.tensor(actions, device=world_model.device)

    # Predict next states using world model
    with torch.no_grad():
        mean_pred, var_pred, _ = world_model.predict_next_state(obs_t, act_t)
        next_obs = mean_pred.cpu().numpy()

    print(f"  Generated: obs={obs.shape}, action={actions.shape}, next_obs={next_obs.shape}")
    return {
        'obs': obs,
        'action': actions,
        'next_obs': next_obs,
    }


def compute_reward_from_transitions(obs, next_obs, actions):
    """Compute attitude reward for synthetic transitions.

    Reward = -|theta_next| - 0.01*|theta_dot_next| - 0.1*|action|
    (Encourage small tilt, small angular velocity, small actions)
    """
    # Extract theta and theta_dot from normalized state
    # theta is index 3, theta_dot is index 4
    theta_next = next_obs[:, 3] * 1.57  # Denormalize
    theta_dot_next = next_obs[:, 4] * 10.0  # Denormalize

    reward = -np.abs(theta_next) - 0.01 * np.abs(theta_dot_next) - 0.1 * np.abs(actions[:, 0])
    return reward


def train_policy_on_synthetic_data(policy, world_model, n_iterations=10,
                                    n_samples_per_iter=10000,
                                    batch_size=256, lr=3e-4):
    """Train residual policy using Dyna-style approach.

    For each iteration:
    1. Generate synthetic data using world model
    2. Compute rewards
    3. Train policy to maximize reward (behavioral cloning on good actions)
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    best_reward = float('-inf')

    for iteration in range(n_iterations):
        # Generate synthetic data
        data = generate_synthetic_data(world_model, n_samples=n_samples_per_iter)

        # Compute rewards
        rewards = compute_reward_from_transitions(data['obs'], data['next_obs'], data['action'])

        # Select top-k% transitions (best actions)
        k_percent = 0.2  # Top 20%
        n_select = int(len(rewards) * k_percent)
        top_indices = np.argsort(rewards)[-n_select:]

        obs_top = data['obs'][top_indices]
        act_top = data['action'][top_indices]
        rewards_top = rewards[top_indices]

        # Create dataset
        dataset = TensorDataset(
            torch.tensor(obs_top, dtype=torch.float32),
            torch.tensor(act_top, dtype=torch.float32),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train policy (behavioral cloning on good actions)
        policy.train()
        total_loss = 0
        n_batches = 0

        for obs_batch, act_batch in dataloader:
            # Forward pass
            pred_act = policy(obs_batch)

            # Loss: MSE between predicted and good actions
            loss = F.mse_loss(pred_act, act_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_reward = np.mean(rewards_top)

        print(f"  Iter {iteration+1}/{n_iterations}: "
              f"loss={avg_loss:.6f}, avg_reward={avg_reward:.4f}, "
              f"n_selected={n_select}")

        if avg_reward > best_reward:
            best_reward = avg_reward

    return best_reward


def evaluate_policy(env, policy, n_episodes=5, max_steps=200):
    """Evaluate residual policy vs LQR baseline."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total_r = 0
        et = False
        theta_list = []
        eps_list = []

        policy.eval()
        for step in range(min(max_steps, env.max_episode_steps)):
            # Get policy action
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = policy(obs_t).squeeze(0).numpy()

            obs, r, t, tr, info = env.step(action)
            total_r += r
            theta_list.append(abs(info['raw_state'][3]))
            eps_list.append(float(np.abs(action[0])))
            if t:
                et = True
            if t or tr:
                break

        results.append({
            'return': total_r,
            'length': step + 1,
            'early_term': et,
            'theta_rms': float(np.sqrt(np.mean(np.array(theta_list)**2))),
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


def evaluate_baseline(env, n_episodes=5, max_steps=200):
    """Evaluate zero-residual baseline."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total_r = 0
        et = False
        theta_list = []

        for step in range(min(max_steps, env.max_episode_steps)):
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


def main():
    sys.stdout.reconfigure(line_buffering=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 80)
    print("  Stage 1: Dyna-style Model-based Residual Policy")
    print(f"  Timestamp: {timestamp}")
    print("=" * 80)

    # ============================================================
    # Phase 0: Load World Model
    # ============================================================
    print("\n" + "=" * 80)
    print("  Phase 0: Loading World Model")
    print("=" * 80)

    sindy_data = np.load(os.path.join(base_dir, 'sindy_model_improved.npz'))
    sindy_Xi = sindy_data['coefficients']
    action_scale = float(sindy_data['action_scale'])

    # Load trained ensemble from Stage 1
    ckpt_path = os.path.join(base_dir, 'checkpoints', 'stage1_ensemble.pt')
    if not os.path.exists(ckpt_path):
        print(f"  ERROR: Stage 1 checkpoint not found: {ckpt_path}")
        return

    cfg = {
        'obs_dim': 8, 'action_dim': 1, 'mlp_dim': 256,
        'ensemble_size': 3,
    }
    world_model = EnsembleWorldModel(
        obs_dim=8, action_dim=1, mlp_dim=256,
        sindy_Xi=sindy_Xi, action_scale=action_scale,
        ensemble_size=3,
    )
    world_model.load(ckpt_path)
    print(f"  Loaded world model from {ckpt_path}")

    # ============================================================
    # Phase 1: Train Residual Policy (Dyna-style)
    # ============================================================
    print("\n" + "=" * 80)
    print("  Phase 1: Training Residual Policy (Dyna-style)")
    print("=" * 80)

    policy = ResidualPolicy(obs_dim=8, hidden_dim=128)
    best_reward = train_policy_on_synthetic_data(
        policy, world_model,
        n_iterations=10,
        n_samples_per_iter=10000,
        batch_size=256,
        lr=3e-4,
    )
    print(f"  Best reward: {best_reward:.4f}")

    # ============================================================
    # Phase 2: Evaluate Policy
    # ============================================================
    print("\n" + "=" * 80)
    print("  Phase 2: Evaluation")
    print("=" * 80)

    # Load LQR config
    best_lqr_path = os.path.join(base_dir, 'configs', 'best_lqr.json')
    lqr_kwargs = {}
    if os.path.exists(best_lqr_path):
        with open(best_lqr_path) as f:
            best = json.load(f)
        lqr_kwargs = {'lqr_Q': best['Q_diag'], 'lqr_R': best['R_val']}

    env = AttitudeControlEnv(max_episode_steps=1000, **lqr_kwargs)

    # Baseline
    baseline = evaluate_baseline(env, n_episodes=5, max_steps=200)
    print(f"  Baseline: return={baseline['return']:.2f}, ET={baseline['ET']}/5, "
          f"theta_rms={baseline['theta_rms']:.4f}")

    # Policy
    policy_result = evaluate_policy(env, policy, n_episodes=5, max_steps=200)
    advantage = policy_result['return'] - baseline['return']
    print(f"  Policy: return={policy_result['return']:.2f}, ET={policy_result['ET']}/5, "
          f"theta_rms={policy_result['theta_rms']:.4f}")
    print(f"  Advantage: {advantage:.2f}")

    # ============================================================
    # Phase 3: Save Policy
    # ============================================================
    print("\n" + "=" * 80)
    print("  Phase 3: Save Policy")
    print("=" * 80)

    ckpt_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    policy_path = os.path.join(ckpt_dir, 'stage1_residual_policy.pt')

    torch.save({
        'policy_state': policy.state_dict(),
        'obs_dim': 8,
        'hidden_dim': 128,
        'best_reward': best_reward,
        'baseline': baseline,
        'policy_result': policy_result,
        'advantage': float(advantage),
        'timestamp': timestamp,
    }, policy_path)

    print(f"  Saved policy: {policy_path}")
    print(f"  Advantage: {advantage:.2f}")
    print("=" * 80)

    env.close()


if __name__ == '__main__':
    main()
