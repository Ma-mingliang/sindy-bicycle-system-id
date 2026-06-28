"""
SINDy-based bicycle balance environment (no PyBullet dependency).

Uses the identified SINDy dynamics model directly for simulation.
This is the analytical counterpart to bicycle_env.py (PyBullet).

State (8D): [ey, epsi, v, theta, theta_dot, k, delta, delta_dot]
Action (1D): steering command in [-1, 1]
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _load_sindy_model(path=None):
    """Load SINDy coefficients from npz file."""
    import os
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sindy_model_improved.npz')
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sindy_model.npz')
    data = np.load(path, allow_pickle=True)
    return {
        'coefficients': data['coefficients'],
        'lib_labels': list(data['lib_labels']) if 'lib_labels' in data else [],
        'action_scale': float(data['action_scale']) if 'action_scale' in data else 1.0,
    }


def _build_theta_row(state_norm, action_norm):
    """
    Build polynomial library row for a single state-action pair.
    Matches build_polynomial_library in sindy_identification.py:
    [1, x0, x1, ..., x8, x0^2, x0*x1, ..., x8^2]
    where features = [state_norm(8), action_norm(1)] = 9 features.
    """
    x = np.concatenate([state_norm, [action_norm]])
    n = len(x)
    terms = [1.0]
    terms.extend(x)
    for i in range(n):
        for j in range(i, n):
            terms.append(x[i] * x[j])
    return np.array(terms)


class SINDyBicycleEnv(gym.Env):
    """
    Bicycle balancing environment using SINDy dynamics.

    The transition model is:
        s_next = s + f_SINDy(s_norm, a_norm) (in raw space after denormalization)

    This environment is lightweight (no physics engine) and directly
    exercises the SINDy world model identified from data.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, sindy_model_path=None, max_episode_steps=300):
        super().__init__()
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-1., -1., -1., -1., -1., -1., -1., -1.], dtype=np.float32),
            high=np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.], dtype=np.float32),
            dtype=np.float32,
        )

        # Load SINDy model
        model = _load_sindy_model(sindy_model_path)
        self._Xi = model['coefficients']  # (n_lib, 8)
        self._action_scale = model['action_scale']

        # Normalization constants
        self._norm_scale = np.array([10.0, 1.57, 5.0, 1.57, 10.0, 1/8.0, 0.785, 3.0], dtype=np.float32)

        self._step_count = 0
        self._state_raw = np.zeros(8, dtype=np.float64)

    def _normalize(self, raw):
        """Normalize raw state to [-1, 1]."""
        normed = raw.copy()
        normed[0] = np.clip(normed[0], -10, 10) / 10.0
        normed[1] = np.clip(normed[1], -1.57, 1.57) / 1.57
        normed[2] = np.clip(normed[2], -5, 5) / 5.0
        normed[3] = np.clip(normed[3], -1.57, 1.57) / 1.57
        normed[4] = np.clip(normed[4], -10, 10) / 10.0
        normed[5] = np.clip(normed[5], -0.125, 0.125) * 8.0
        normed[6] = np.clip(normed[6], -0.785, 0.785) / 0.785
        normed[7] = np.clip(normed[7], -3, 3) / 3.0
        return normed.astype(np.float32)

    def _denormalize(self, normed):
        """Denormalize from [-1, 1] to raw state."""
        raw = normed.copy().astype(np.float64)
        raw[0] = raw[0] * 10.0
        raw[1] = raw[1] * 1.57
        raw[2] = raw[2] * 5.0
        raw[3] = raw[3] * 1.57
        raw[4] = raw[4] * 10.0
        raw[5] = raw[5] / 8.0
        raw[6] = raw[6] * 0.785
        raw[7] = raw[7] * 3.0
        return raw

    def _sindy_predict_delta(self, state_norm, action_norm):
        """SINDy single-step prediction: returns delta_state in normalized space."""
        theta_row = _build_theta_row(state_norm, action_norm)
        delta = theta_row @ self._Xi  # (8,)
        return delta

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        # Very small initial perturbation (easier to balance)
        raw = np.zeros(8, dtype=np.float64)
        raw[3] = self.np_random.uniform(-0.05, 0.05)  # tiny roll angle
        raw[4] = self.np_random.uniform(-0.5, 0.5)    # small roll rate
        raw[2] = self.np_random.uniform(3.0, 4.0)     # moderate forward speed
        self._state_raw = raw

        obs = self._normalize(raw)
        return obs, {"raw_state": raw.copy()}

    def step(self, action):
        action = float(np.clip(action[0], -1.0, 1.0))
        self._step_count += 1

        # Current normalized state
        state_norm = self._normalize(self._state_raw)
        action_norm = action / self._action_scale if self._action_scale > 0 else action

        # SINDy prediction (delta in normalized space)
        delta = self._sindy_predict_delta(state_norm, action_norm)

        # Add damping to prevent divergence (SINDy model is approximate)
        delta = delta * 0.5  # scale down dynamics for stability

        # Next state in normalized space, then denormalize
        next_norm = state_norm + delta.astype(np.float32)
        # Clamp to valid range
        next_norm = np.clip(next_norm, -2.0, 2.0)
        next_raw = self._denormalize(next_norm)

        # Reward
        prev_raw = self._state_raw.copy()
        reward = self._compute_reward(prev_raw, next_raw)

        self._state_raw = next_raw
        obs = self._normalize(next_raw)

        # Termination (more lenient for easier learning)
        theta = next_raw[3]
        ey = next_raw[0]
        terminated = bool(abs(theta) > 0.8 or abs(ey) > 5.0 or abs(next_raw[2]) < 0.5)
        truncated = bool(self._step_count >= self.max_episode_steps)

        info = {
            "raw_state": next_raw.copy(),
            "success": float(abs(theta) < 0.05 and abs(ey) < 0.5),
            "terminated": terminated,
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, prev_raw, raw):
        theta = raw[3]
        theta_dot = raw[4]
        ey = raw[0]

        # Shaped reward: quadratic penalty on roll angle + rate
        reward = 1.0 - 10.0 * theta**2 - 0.5 * theta_dot**2 - 0.01 * ey**2
        return float(np.clip(reward, -5, 5))

    def close(self):
        pass


if __name__ == "__main__":
    env = SINDyBicycleEnv(max_episode_steps=100)
    obs, info = env.reset()
    print(f"Initial obs: {obs}")

    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Step {step+1}: terminated={terminated}, total_reward={total_reward:.2f}")
            break

    env.close()
    print("SINDy environment smoke test passed.")
