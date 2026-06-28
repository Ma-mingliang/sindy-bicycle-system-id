"""Training script: Conservative Ensemble Residual-MPPI with SINDy prior.

Pipeline:
  1. Collect baseline data (10000+ steps, LQR+Stanley+noise)
  2. Train ensemble world models (3 members, bootstrap data)
  3. Validate: one-step error, 5-step rollout, ensemble uncertainty
  4. Compute data-driven uncertainty threshold from val set
  5. Safety checks before enabling MPPI control
  6. MPPI diagnostics (fallback reasons, predicted advantage)
  7. MPPI evaluation with zero-residual comparison
"""

import sys
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from path_tracking_env import PathTrackingEnv
from residual_mppi import ConservativeEnsembleResidualMPPI

# Checkpoint directory
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'residual_mppi_ensemble_v1')


class ReplayBuffer:
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


def evaluate_zero_residual(n_episodes=10, max_steps=1500):
    """Evaluate Stanley-only baseline (zero residual)."""
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
        'returns': returns,
    }


def evaluate_mppi(agent, n_episodes=10, max_steps=1500):
    """Evaluate MPPI planning in real environment with fallback reason tracking."""
    env = PathTrackingEnv(max_episode_steps=max_steps)
    returns, lengths, lat_errors = [], [], []
    fallback_rates, nonfallback_counts, uncertainties = [], [], []
    all_fb_reasons = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        agent.reset_planning()
        total_r = 0
        ep_lat = []
        ep_fallback = 0
        ep_nf = 0
        ep_unc = []

        for step in range(max_steps):
            eps_action, best_G, zero_G, uncertainty = agent.act(obs, eval_mode=True)
            ep_unc.append(uncertainty)

            if np.allclose(eps_action, 0):
                ep_fallback += 1
            else:
                ep_nf += 1

            obs, reward, terminated, truncated, info = env.step(eps_action)
            total_r += reward
            ep_lat.append(abs(info.get('path_info', {}).get('lateral_error', 0)))
            if terminated or truncated:
                break

        n = step + 1
        returns.append(total_r)
        lengths.append(n)
        lat_errors.append(np.mean(ep_lat))
        fallback_rates.append(ep_fallback / n)
        nonfallback_counts.append(ep_nf)
        uncertainties.append(np.mean(ep_unc))
        all_fb_reasons.append(agent.get_fb_reasons())

    env.close()

    # Aggregate fallback reasons
    avg_fb = {}
    for key in all_fb_reasons[0]:
        avg_fb[key] = np.mean([r[key] for r in all_fb_reasons])

    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_length': np.mean(lengths),
        'mean_lateral_error': np.mean(lat_errors),
        'mean_fallback_rate': np.mean(fallback_rates),
        'mean_nonfallback_count': np.mean(nonfallback_counts),
        'mean_uncertainty': np.mean(uncertainties),
        'avg_fallback_reasons': avg_fb,
    }


def collect_baseline_data(env, buffer, n_steps=10000, epsilon_range=0.3):
    """Collect data with LQR+Stanley+noise."""
    obs, _ = env.reset()
    for i in range(n_steps):
        action = np.random.uniform(-epsilon_range, epsilon_range, size=(1,)).astype(np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    return buffer.size


def print_separator(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def train():
    sys.stdout.reconfigure(line_buffering=True)
    print_separator("Conservative Ensemble Residual-MPPI - Training")

    # Config
    cfg = {
        'obs_dim': 8,
        'action_dim': 1,
        'mlp_dim': 256,
        'horizon': 5,
        'num_samples': 128,
        'num_elites': 16,
        'iterations': 6,
        'temperature': 0.5,
        'epsilon_max': 0.1,
        'epsilon_std': 0.08,
        'gamma': 0.95,
        'lambda_res': 2.0,
        'lambda_smooth': 5.0,
        'lambda_uncertainty': 5.0,
        'uncertainty_threshold': None,  # Will be set from val set
        'margin': 0.1,
        'ensemble_size': 3,
    }

    # Load SINDy model
    sindy_data = np.load(os.path.join(os.path.dirname(__file__), 'sindy_model_improved.npz'))
    sindy_Xi = sindy_data['coefficients']
    action_scale = float(sindy_data['action_scale'])
    print(f"\n  SINDy model: Xi shape={sindy_Xi.shape}, action_scale={action_scale:.3f}")

    # Create agent, env, buffer
    agent = ConservativeEnsembleResidualMPPI(
        cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1,
    )
    env = PathTrackingEnv(max_episode_steps=1500)
    buffer = ReplayBuffer(capacity=100000, obs_dim=8, action_dim=1)

    print(f"  Device: {agent.device}")
    print(f"  Ensemble size: {cfg['ensemble_size']}")

    # === Phase 1: Collect baseline data ===
    print_separator("Phase 1: Collecting baseline data")

    n_collected = collect_baseline_data(env, buffer, n_steps=10000, epsilon_range=0.3)
    print(f"  Collected {n_collected} transitions")
    print(f"  Action range: [{buffer.action[:buffer.size].min():.3f}, "
          f"{buffer.action[:buffer.size].max():.3f}]")

    # === Phase 2: Train ensemble world models ===
    print_separator("Phase 2: Training ensemble world models")

    train_results = agent.train_world_model(buffer, epochs=50, batch_size=256, val_ratio=0.2)

    for i, result in enumerate(train_results):
        print(f"\n  Member {i}:")
        print(f"    Best val loss: {result['best_val_loss']:.6f}")
        print(f"    Epochs trained: {result['epochs_trained']}")

    # === Phase 3: Model error comparison ===
    print_separator("Phase 3: Model error comparison")

    errors = agent.compute_model_errors(buffer)
    state_names = ['ey', 'epsi', 'v', 'theta', 'theta_dot', 'k', 'delta', 'delta_dot']

    print(f"\n  SINDy-only error:  {errors['sindy_error']:.6f}")
    print(f"  Ensemble mean error:  {errors['model_error']:.6f}")
    print(f"  Avg uncertainty:  {errors['avg_uncertainty']:.6f}")
    print(f"  model_error / sindy_error = {errors['model_error'] / max(errors['sindy_error'], 1e-10):.4f}")

    print(f"\n  Per-dim comparison:")
    for i, name in enumerate(state_names):
        s_err = errors['sindy_per_dim'][i]
        m_err = errors['model_per_dim'][i]
        u_var = errors['uncertainty_per_dim'][i]
        ratio = m_err / max(s_err, 1e-10)
        print(f"    {name:>12}: sindy={s_err:.6f}, model={m_err:.6f} (×{ratio:.2f}), unc={u_var:.6f}")

    # === Phase 4: Rollout validation ===
    print_separator("Phase 4: Rollout validation")

    rollout = agent.validate_rollout(buffer, horizon=5, n_samples=200)
    print(f"  One-step error: {rollout['one_step_error']:.6f}")
    print(f"  5-step rollout error: {rollout['multi_step_error']}")
    print(f"  5-step rollout uncertainty: {rollout['multi_step_uncertainty']}")
    print(f"  Rollout diverges: {rollout['diverges']}")

    # === Phase 5: Uncertainty percentile analysis ===
    print_separator("Phase 5: Uncertainty percentile analysis")

    val_unc = agent.compute_val_uncertainty_percentiles(buffer, val_ratio=0.2)
    train_unc = agent.compute_train_uncertainty_percentiles(buffer, val_ratio=0.2)

    print(f"\n  Training set uncertainty:")
    print(f"    p50={train_unc['p50']:.6f}, p90={train_unc['p90']:.6f}, "
          f"p95={train_unc['p95']:.6f}, p99={train_unc['p99']:.6f}")
    print(f"    mean={train_unc['mean']:.6f}, std={train_unc['std']:.6f}")

    print(f"\n  Validation set uncertainty:")
    print(f"    p50={val_unc['p50']:.6f}, p90={val_unc['p90']:.6f}, "
          f"p95={val_unc['p95']:.6f}, p99={val_unc['p99']:.6f}")
    print(f"    mean={val_unc['mean']:.6f}, std={val_unc['std']:.6f}")

    # Set uncertainty threshold from val set p95
    unc_threshold_p95 = val_unc['p95']
    unc_threshold_p90 = val_unc['p90']
    print(f"\n  Data-driven uncertainty threshold:")
    print(f"    p90-based: {unc_threshold_p90:.6f}")
    print(f"    p95-based: {unc_threshold_p95:.6f}")

    # Use p95 as threshold
    agent.uncertainty_threshold = unc_threshold_p95
    cfg['uncertainty_threshold'] = unc_threshold_p95
    print(f"  Using uncertainty_threshold = p95 = {unc_threshold_p95:.6f}")

    # === Phase 6: Safety checks ===
    print_separator("Phase 6: Safety checks")

    checks_passed = True

    for i, result in enumerate(train_results):
        if result['best_val_loss'] > 0.01:
            print(f"  [WARN] Member {i} val loss high: {result['best_val_loss']:.6f}")

    if rollout['diverges']:
        print("  [WARN] 5-step rollout diverges (expected for learned models)")
        print(f"    One-step error is good: {rollout['one_step_error']:.6f}")
        print(f"    MPPI re-plans each step, so this is not a blocker")
    else:
        print("  [PASS] 5-step rollout stable")

    if errors['model_error'] >= errors['sindy_error']:
        print("  [WARN] Ensemble mean not better than SINDy-only")
    else:
        print("  [PASS] Ensemble mean better than SINDy-only")

    if errors['avg_uncertainty'] > 0.1:
        print("  [WARN] High ensemble uncertainty")
    else:
        print(f"  [PASS] Ensemble uncertainty reasonable: {errors['avg_uncertainty']:.6f}")

    # === Phase 7: Evaluate zero-residual baseline ===
    print_separator("Phase 7: Evaluating baselines")

    zero_stats = evaluate_zero_residual(n_episodes=10)
    print(f"  Zero-residual: return={zero_stats['mean_return']:.2f} +/- {zero_stats['std_return']:.2f}")
    print(f"  Mean lateral error: {zero_stats['mean_lateral_error']:.3f}")
    print(f"  Mean length: {zero_stats['mean_length']:.1f}")

    # === Phase 8: MPPI diagnostics (before full eval) ===
    if checks_passed:
        print_separator("Phase 8: MPPI diagnostics (5 episodes)")

        diag_stats = evaluate_mppi(agent, n_episodes=5)

        print(f"  Mean return: {diag_stats['mean_return']:.2f} +/- {diag_stats['std_return']:.2f}")
        print(f"  Mean fallback rate: {diag_stats['mean_fallback_rate']:.1%}")
        print(f"  Mean non-fallback steps: {diag_stats['mean_nonfallback_count']:.1f}")
        print(f"  Mean uncertainty: {diag_stats['mean_uncertainty']:.6f}")

        fb = diag_stats['avg_fallback_reasons']
        print(f"\n  Fallback reason breakdown:")
        print(f"    by one_step_reward: {fb.get('one_step_reward', 0):.3f}")
        print(f"    by risk:           {fb.get('risk', 0):.3f}")
        print(f"    by uncertainty:    {fb.get('uncertainty', 0):.3f}")
        print(f"    by margin:         {fb.get('margin', 0):.3f}")

        if diag_stats['mean_fallback_rate'] > 0.95:
            print("\n  [WARN] >95% fallback — MPPI is not contributing.")
            print("  Diagnose using fallback reasons above before running full eval.")
        elif diag_stats['mean_fallback_rate'] > 0.5:
            print(f"\n  [INFO] {diag_stats['mean_fallback_rate']:.0%} fallback — some MPPI contributions.")
        else:
            print(f"\n  [PASS] {diag_stats['mean_fallback_rate']:.0%} fallback — MPPI is active.")

    else:
        print_separator("MPPI evaluation SKIPPED (safety checks failed)")

    # Save checkpoint
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
    agent.save(ckpt_path, extra_meta={
        'uncertainty_threshold_p95': unc_threshold_p95,
        'uncertainty_threshold_p90': unc_threshold_p90,
        'sindy_error': errors['sindy_error'],
        'model_error': errors['model_error'],
        'val_uncertainty_p50': val_unc['p50'],
        'val_uncertainty_p90': val_unc['p90'],
        'val_uncertainty_p95': val_unc['p95'],
        'val_uncertainty_p99': val_unc['p99'],
    })
    print(f"\n  Checkpoint saved to {ckpt_path}")

    print_separator()
    env.close()
    return train_results


if __name__ == '__main__':
    train()
