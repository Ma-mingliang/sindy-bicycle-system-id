"""Diagnostic script for path tracking environment."""
import numpy as np
from path_tracking_env import PathTrackingEnv

env = PathTrackingEnv(stage1_agent=None, max_episode_steps=200)
obs, info = env.reset(seed=0)
print(f'Start: x={env._x:.2f}, y={env._y:.2f}, heading={env._heading:.3f}')
print(f'Init obs: {obs}')
print(f'Init path_info: {info["path_info"]}')

for step in range(50):
    action = np.array([0.0])
    obs, r, t, tr, info = env.step(action)
    path = info['path_info']
    raw = info['raw_state']
    if step < 20 or step % 10 == 0:
        print(f'Step {step:3d}: x={env._x:.2f} y={env._y:.2f} hdg={env._heading:.3f} '
              f'| lat_err={path["lateral_error"]:.3f} course_err={path["course_error_angle"]:.3f} '
              f'| theta={raw[3]:.4f} theta_dot={raw[4]:.4f} delta={raw[6]:.4f} '
              f'| target_roll={info["target_roll"]:.4f} R={r:.3f} seg={path["segment"]}')
    if t or tr:
        term_reason = "roll_limit" if abs(raw[3]) > 1.05 else "lateral_limit" if abs(path["lateral_error"]) >= 3 else "heading_limit" if abs(path["course_error_angle"]) > 1.05 else "other"
        print(f'TERMINATED at step {step+1}: {term_reason}')
        break
env.close()
