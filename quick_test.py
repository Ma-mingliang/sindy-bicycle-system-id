import sys
sys.stdout.reconfigure(line_buffering=True)
print('Importing...')
from path_tracking_env import PathTrackingEnv
print('Creating env...')
env = PathTrackingEnv(max_episode_steps=500)
print('Testing Stanley-only...')
returns = []
for ep in range(10):
    obs, _ = env.reset(seed=ep)
    total_r = 0
    for step in range(500):
        action = env.action_space.sample() * 0  # zero action
        obs, r, t, tr, info = env.step(action)
        total_r += r
        if t or tr:
            break
    returns.append(total_r)
    print(f'  Ep {ep}: return={total_r:.2f}, steps={step+1}')
import numpy as np
print(f'\nStanley-only: {np.mean(returns):.2f} +/- {np.std(returns):.2f}')
env.close()
print('Done')
