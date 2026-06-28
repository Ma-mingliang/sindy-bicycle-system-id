"""Quick LQR survival test with disturbances."""
import numpy as np
from bicycle_env_analytical import AnalyticalBicycleEnv

env = AnalyticalBicycleEnv(max_episode_steps=800)
returns, lengths = [], []
for _ in range(50):
    obs, _ = env.reset()
    total_r = 0
    for step in range(800):
        action = np.array([0.0])  # zero residual
        obs, r, t, tr, info = env.step(action)
        total_r += r
        if t or tr:
            break
    returns.append(total_r)
    lengths.append(step + 1)
print(f"LQR-only (disturbed): return={np.mean(returns):.2f}±{np.std(returns):.2f}, length={np.mean(lengths):.1f}±{np.std(lengths):.1f}")
print(f"  Survival rate: {sum(1 for l in lengths if l >= 800)/len(lengths)*100:.0f}%")
print(f"  Min length: {min(lengths)}")
