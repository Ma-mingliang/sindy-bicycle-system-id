"""Test that different actions produce meaningfully different rewards."""
import numpy as np
from bicycle_env_analytical import AnalyticalBicycleEnv

env = AnalyticalBicycleEnv(max_episode_steps=800)

returns = {0.0: [], 0.5: [], -0.5: []}
for _ in range(50):
    obs, _ = env.reset()
    s0 = env._state.copy()
    for a_val in [0.0, 0.5, -0.5]:
        env._state = s0.copy()
        env._step_count = 0
        total_r = 0
        for step in range(20):
            action = np.array([a_val])
            obs, r, t, tr, info = env.step(action)
            total_r += r
            if t or tr:
                break
        returns[a_val].append(total_r)

print("Action impact on reward (50 eps x 20 steps):")
for a_val, rets in returns.items():
    print(f"  a={a_val:>5.1f}: mean={np.mean(rets):.4f}, std={np.std(rets):.4f}")

diff = np.mean(returns[0.0]) - np.mean(returns[0.5])
print(f"  Reward diff (a=0 vs a=0.5): {diff:.4f}")
print(f"  Signal-to-noise: {abs(diff)/np.std(returns[0.0]):.2f}")
