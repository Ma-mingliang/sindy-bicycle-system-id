"""Debug: check what happens with different seeds."""
import numpy as np
from path_tracking_env import PathTrackingEnv

env = PathTrackingEnv(stage1_agent=None, max_episode_steps=500)

for seed in range(10):
    obs, info = env.reset(seed=seed)
    y0 = env._y
    lat0 = info['path_info']['lateral_error']
    total_r = 0
    term_reason = "completed"
    for step in range(500):
        action = np.array([0.0])
        obs, r, t, tr, info = env.step(action)
        total_r += r
        if t:
            path = info['path_info']
            raw = info['raw_state']
            if abs(raw[3]) > 1.05:
                term_reason = f"roll({raw[3]:.3f})"
            elif abs(path['lateral_error']) >= 3:
                term_reason = f"lat({path['lateral_error']:.3f})"
            elif abs(path['course_error_angle']) > 1.05:
                term_reason = f"heading({path['course_error_angle']:.3f})"
            break
        if tr:
            term_reason = "truncated"
            break
    print(f"Seed {seed}: y0={y0:.2f}, lat0={lat0:.2f}, steps={step+1}, "
          f"return={total_r:.1f}, reason={term_reason}")

env.close()
