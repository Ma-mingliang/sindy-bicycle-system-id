"""Compare LQR-only vs LQR+RL (trained agent) performance."""
import numpy as np
import torch
from bicycle_env_analytical import AnalyticalBicycleEnv
from sindy_tdmpc2 import SINDyTDMPC2
from sindy_env import _load_sindy_model


def evaluate(agent=None, n_episodes=50, max_steps=800, label=""):
    """Evaluate with fixed seeds for fair comparison."""
    env = AnalyticalBicycleEnv(max_episode_steps=max_steps)
    returns, lengths, successes = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)  # Same seeds for fair comparison
        total_r = 0
        for step in range(max_steps):
            if agent is not None:
                action = agent.act(obs, t0=(step == 0), eval_mode=True)
            else:
                action = np.array([0.0])  # LQR-only
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            if terminated or truncated:
                break
        returns.append(total_r)
        lengths.append(step + 1)
        successes.append(float(info.get('success', 0)))

    env.close()
    print(f"\n{label}:")
    print(f"  Return:  {np.mean(returns):.3f} +/- {np.std(returns):.3f}")
    print(f"  Length:  {np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")
    print(f"  Survival: {sum(1 for l in lengths if l >= max_steps)/len(lengths)*100:.0f}%")
    print(f"  Success: {np.mean(successes)*100:.1f}%")
    return returns, lengths


# Load trained agent
print("Loading SINDy model...")
sindy = _load_sindy_model()
Xi = sindy['coefficients']
action_scale = sindy['action_scale']

cfg = {
    'latent_dim': 64, 'mlp_dim': 256, 'lr': 3e-4,
    'horizon': 3, 'num_samples': 64, 'num_elites': 8,
    'num_pi_trajs': 8, 'iterations': 6, 'temperature': 0.5,
    'max_std': 0.5, 'min_std': 0.05, 'sindy_coef': 5.0,
    'tau': 0.005, 'gamma': 0.99, 'grad_clip': 10.0,
}
agent = SINDyTDMPC2(cfg, Xi, action_scale, obs_dim=8, action_dim=1)
agent.load('checkpoint_balance.pt')
print("Agent loaded.\n")

# Run comparison with same seeds
lqr_ret, lqr_len = evaluate(agent=None, n_episodes=50, label="LQR-only")
rl_ret, rl_len = evaluate(agent=agent, n_episodes=50, label="LQR+RL")

# Statistical comparison
print("\n" + "="*50)
print("COMPARISON")
print("="*50)
print(f"  Return diff: {np.mean(rl_ret) - np.mean(lqr_ret):.3f}")
print(f"  Length diff:  {np.mean(rl_len) - np.mean(lqr_len):.1f}")

# Paired t-test
from scipy import stats
t_stat, p_val = stats.ttest_rel(rl_ret, lqr_ret)
print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
if p_val < 0.05:
    print(f"  -> RL is {'better' if t_stat > 0 else 'worse'} than LQR (p<0.05)")
else:
    print(f"  -> No significant difference (p>=0.05)")
