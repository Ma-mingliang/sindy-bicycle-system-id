"""Debug trajectory to understand oscillation."""
import numpy as np
from path_tracking_env import PathTrackingEnv

env = PathTrackingEnv(max_episode_steps=200)
obs, info = env.reset(seed=0)
print(f'Start: x={env._x:.2f}, y={env._y:.2f}')

for step in range(100):
    action = np.array([0.0])
    obs, r, t, tr, info = env.step(action)
    path = info['path_info']
    print(f'Step {step:3d}: x={env._x:.2f} y={env._y:.3f} hdg={env._heading:.3f} '
          f'delta={env._delta:.4f} | lat={path["lateral_error"]:.3f} course={path["course_error_angle"]:.3f}')
    if t or tr:
        print(f'STOPPED at step {step+1}')
        break
env.close()
