"""
Analytical bicycle balance environment with LQR baseline.

Physics: Linearized Whipple model (2-DOF roll-steering).
The LQR controller provides baseline stability.
RL learns a residual correction on top.

State (8D, normalized): [ey, epsi, v, theta, theta_dot, k, delta, delta_dot]
Action (1D): residual steering correction [-1, 1] added to LQR output
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import scipy.linalg as la


def _compute_lqr_gains(g=9.81, h=0.8, v=3.0, L=1.0, wn=5.0, zeta=0.5, trail=0.15):
    """Compute LQR gains for linearized bicycle dynamics."""
    A = np.array([
        [0, 1, 0, 0],
        [g/h, -2.0, -v**2/(h*L), 0],
        [0, 0, 0, 1],
        [wn**2*trail*v/L, 0, -wn**2, -2*zeta*wn]
    ])
    B = np.array([[0], [0], [0], [1.0]])
    Q = np.diag([100, 10, 10, 1])
    R = np.array([[1.0]])
    P = la.solve_continuous_are(A, B, Q, R)
    K = (np.linalg.inv(R) @ B.T @ P).flatten()
    return K  # [k_theta, k_theta_dot, k_delta, k_delta_dot]


class AnalyticalBicycleEnv(gym.Env):
    """
    Analytical bicycle balancing with LQR baseline + RL residual.

    The LQR controller stabilizes the linearized dynamics.
    The RL agent learns a residual steering correction.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_episode_steps=800, dt=1/30, omega_n=5.0, speed=3.0):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.dt = dt
        self._sub_steps = 5  # sub-stepping for integration accuracy

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-1., -1., -1., -1., -1., -1., -1., -1.], dtype=np.float32),
            high=np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.], dtype=np.float32),
            dtype=np.float32,
        )

        # Physical parameters
        self.g = 9.81
        self.h = 0.8
        self.wheelbase = 1.0
        self.omega_n = omega_n
        self.zeta = 0.5
        self.speed = speed
        self.trail = 0.15

        # LQR gains (pre-computed)
        self.K_lqr = _compute_lqr_gains(
            self.g, self.h, self.speed, self.wheelbase,
            self.omega_n, self.zeta, self.trail
        )

        self._step_count = 0
        self._state = np.zeros(8, dtype=np.float64)
        self.target_roll = 0.0  # Stage 2 can set this for roll tracking

    def _physics_step(self, state, u_total):
        """RK4 integration with sub-stepping for accuracy."""
        dt_sub = self.dt / self._sub_steps

        # Process noise: random torque disturbance on roll (simulates wind/gusts)
        disturb_torque = self.np_random.normal(0, 2.0)

        def derivatives(s, u):
            ey, epsi, v, theta, theta_dot, k_ref, delta, delta_dot = s
            trail_effect = self.trail * v * theta / self.wheelbase
            theta_ddot = (self.g / self.h) * theta \
                         - (v**2 / (self.h * self.wheelbase)) * delta \
                         - 2.0 * theta_dot \
                         + disturb_torque
            delta_ddot = -self.omega_n**2 * (delta - trail_effect) \
                         - 2 * self.zeta * self.omega_n * delta_dot \
                         + u
            return np.array([
                v * epsi,
                v * (k_ref - delta / self.wheelbase),
                -0.1 * (v - self.speed),
                theta_dot,
                theta_ddot,
                0.0,
                delta_dot,
                delta_ddot,
            ])

        s = state.copy()
        for _ in range(self._sub_steps):
            k1 = derivatives(s, u_total)
            k2 = derivatives(s + 0.5 * dt_sub * k1, u_total)
            k3 = derivatives(s + 0.5 * dt_sub * k2, u_total)
            k4 = derivatives(s + dt_sub * k3, u_total)
            s = s + (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return s

    def set_target_roll(self, target):
        """Set target roll angle for Stage 2 path tracking."""
        self.target_roll = target

    def _get_lqr_action(self, state):
        """Compute LQR baseline steering torque (tracks target_roll)."""
        theta, theta_dot, delta, delta_dot = state[3], state[4], state[6], state[7]
        x = np.array([theta - self.target_roll, theta_dot, delta, delta_dot])
        return float(-self.K_lqr @ x)

    def _normalize(self, raw):
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self.target_roll = 0.0

        raw = np.zeros(8, dtype=np.float64)
        raw[3] = self.np_random.uniform(-0.1, 0.1)   # theta
        raw[4] = self.np_random.uniform(-0.5, 0.5)    # theta_dot
        raw[2] = self.speed + self.np_random.uniform(-0.3, 0.3)  # v
        self._state = raw

        return self._normalize(raw), {"raw_state": raw.copy()}

    def step(self, action):
        action = float(np.clip(action[0], -1.0, 1.0))
        self._step_count += 1

        # LQR baseline + RL residual
        u_lqr = self._get_lqr_action(self._state)
        u_residual = action * 2.0  # HRRL-style: residual compensation (LQR does heavy lifting)
        u_total = u_lqr + u_residual

        prev_raw = self._state.copy()
        self._state = self._physics_step(self._state, u_total)

        obs = self._normalize(self._state)
        reward = self._compute_reward(prev_raw, self._state, action)

        theta = self._state[3]
        # Only terminate on excessive roll angle (balance task)
        terminated = bool(abs(theta) > 0.8)
        truncated = bool(self._step_count >= self.max_episode_steps)

        info = {
            "raw_state": self._state.copy(),
            "success": float(abs(theta) < 0.02 and abs(self._state[4]) < 0.5),
            "terminated": terminated,
            "u_lqr": u_lqr,
            "u_residual": u_residual,
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, prev_raw, raw, action):
        """HRRL-style reward: error reduction when far, smoothness when near."""
        prev_theta = prev_raw[3]
        theta = raw[3]
        theta_dot = raw[4]

        beta = 0.002  # ~0.1 degree threshold
        if abs(prev_theta) < beta:
            # Near upright: reward smoothness (penalize angular velocity)
            reward = 0.1 - abs(theta_dot)
        else:
            # Far from upright: reward error reduction
            reward = abs(prev_theta) - abs(theta)
        return float(reward)

    def close(self):
        pass


if __name__ == "__main__":
    env = AnalyticalBicycleEnv(max_episode_steps=300)

    # Test LQR-only (no residual)
    returns, lengths = [], []
    for _ in range(20):
        obs, _ = env.reset()
        total_r = 0
        for step in range(300):
            action = np.array([0.0])  # zero residual
            obs, r, t, tr, info = env.step(action)
            total_r += r
            if t or tr:
                break
        returns.append(total_r)
        lengths.append(step + 1)
    print(f'LQR-only: return={np.mean(returns):.2f}, length={np.mean(lengths):.1f}')

    # Test random residual
    returns2, lengths2 = [], []
    for _ in range(20):
        obs, _ = env.reset()
        total_r = 0
        for step in range(300):
            action = env.action_space.sample()
            obs, r, t, tr, _ = env.step(action)
            total_r += r
            if t or tr:
                break
        returns2.append(total_r)
        lengths2.append(step + 1)
    print(f'Random:   return={np.mean(returns2):.2f}, length={np.mean(lengths2):.1f}')

    # Test one episode detail
    obs, _ = env.reset()
    print(f'\nDetailed LQR episode:')
    print(f'Init: theta={obs[3]*1.57:.4f}')
    for step in range(300):
        action = np.array([0.0])
        obs, r, t, tr, info = env.step(action)
        if step % 30 == 0:
            raw = info['raw_state']
            print(f'  Step {step}: theta={raw[3]:.6f} td={raw[4]:.6f} R={r:.3f}')
        if t or tr:
            print(f'  Terminated at step {step+1}')
            break
    print(f'Final theta={info["raw_state"][3]:.6f}')
    env.close()
