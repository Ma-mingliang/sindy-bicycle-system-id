"""Stage 1: Straight-line attitude control environment.

Reference: HRRL Attitude_control class (env.py lines 39-344)
Architecture:
  1. Progressive disturbance: target_theta sampled every 100 steps
     sigma = min(0.3, 1.08^episode_count * 0.01)
  2. LQR controller tracks target_theta
  3. RL residual on steering torque

State (8D, normalized): [ey, epsi, v, theta, theta_dot, k=0, delta, delta_dot]
  - ey, epsi: straight-line deviation (simplified, no curved path)
  - k: always 0 (straight line)
Action (1D): RL residual [-1, 1], scaled to steering torque residual

Key differences from PathTrackingEnv:
  - No reference path (straight line only)
  - Progressive tilt disturbance (sigma grows with episode)
  - target_theta from disturbance, not from Stanley
  - Termination: |theta| > pi/3 (60 deg, matching HRRL)
  - Episode length: 1000 (matching HRRL)
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from path_tracking_env import _compute_lqr_gains
from reward_fn import compute_attitude_reward


class AttitudeControlEnv(gym.Env):
    """Straight-line attitude control with progressive disturbance (HRRL Stage 1).

    Control flow:
      target_theta = N(0, sigma)  # progressive disturbance, sampled every 100 steps
      u_lqr = -K @ [theta - target_theta, theta_dot, delta, delta_dot]
      u_total = u_lqr + alpha * action
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_episode_steps=1000, dt=1/30, speed=3.0,
                 lqr_Q=None, lqr_R=None):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.dt = dt
        self.speed = speed
        self.wheelbase = 1.0
        self.g = 9.81
        self.h = 0.8
        self.omega_n = 5.0
        self.zeta = 0.5
        self.trail = 0.15
        self._sub_steps = 5

        # LQR gains
        self.K_lqr = _compute_lqr_gains(
            self.g, self.h, self.speed, self.wheelbase,
            self.omega_n, self.zeta, self.trail,
            Q_diag=lqr_Q, R_val=lqr_R,
        )

        # Action: RL residual on steering torque [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 8D observation (same as PathTrackingEnv for SINDy compatibility)
        self.observation_space = spaces.Box(
            low=np.array([-1., -1., -1., -1., -1., -1., -1., -1.], dtype=np.float32),
            high=np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.], dtype=np.float32),
            dtype=np.float32,
        )

        # State
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0
        self._v = speed
        self._theta = 0.0
        self._theta_dot = 0.0
        self._delta = 0.0
        self._delta_dot = 0.0
        self._step_count = 0
        self._episode_count = 0

        # Progressive disturbance
        self._target_theta = 0.0

    def _normalize(self, ey, epsi, v, theta, theta_dot, k, delta, delta_dot):
        """Normalize 8D state to [-1, 1] range."""
        return np.array([
            np.clip(ey, -10, 10) / 10.0,
            np.clip(epsi, -1.57, 1.57) / 1.57,
            np.clip(v, -5, 5) / 5.0,
            np.clip(theta, -1.57, 1.57) / 1.57,
            np.clip(theta_dot, -10, 10) / 10.0,
            np.clip(k, -0.125, 0.125) * 8.0,
            np.clip(delta, -0.785, 0.785) / 0.785,
            np.clip(delta_dot, -3, 3) / 3.0,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._episode_count += 1

        # Reset state (straight line, small perturbations)
        self._x = 0.0
        self._y = self.np_random.uniform(-0.2, 0.2)
        self._heading = 0.0
        self._v = self.speed + self.np_random.uniform(-0.3, 0.3)
        self._theta = self.np_random.uniform(-0.05, 0.05)
        self._theta_dot = 0.0
        self._delta = 0.0
        self._delta_dot = 0.0

        # Initial target_theta
        self._target_theta = 0.0

        # Straight line: ey = y, epsi = heading, k = 0
        ey = self._y
        epsi = self._heading

        obs = self._normalize(
            ey, epsi, self._v, self._theta, self._theta_dot,
            0.0, self._delta, self._delta_dot,
        )
        return obs, {"target_theta": self._target_theta}

    def step(self, action):
        action = float(np.clip(action[0], -1.0, 1.0))
        self._step_count += 1

        # ============================================================
        # Progressive disturbance (reference: HRRL env.py line 249-253)
        # ============================================================
        if self._step_count % 100 == 1:  # Sample at step 1, 101, 201, ...
            sigma = min(0.3, (1.08 ** self._episode_count) * 0.01)
            self._target_theta = self.np_random.normal(0.0, sigma)
            # Clamp to [-pi/12, pi/12] (matching HRRL)
            self._target_theta = np.clip(self._target_theta, -math.pi / 12, math.pi / 12)

        # ============================================================
        # Control: LQR tracks target_theta + RL residual
        # ============================================================
        alpha = 0.15
        epsilon = action * alpha

        # LQR state: [theta - target_theta, theta_dot, delta, delta_dot]
        x_lqr = np.array([
            self._theta - self._target_theta,
            self._theta_dot,
            self._delta,
            self._delta_dot,
        ])
        u_lqr = float(-self.K_lqr @ x_lqr)
        u_total = u_lqr + epsilon

        # ============================================================
        # Current state (before dynamics update)
        # ============================================================
        ey = self._y
        epsi = self._heading
        prev_dis_angle = self._target_theta - self._theta

        # ============================================================
        # Dynamics (RK4 + sub-stepping)
        # ============================================================
        dt_sub = self.dt / self._sub_steps

        def derivatives(s, u):
            """Linearized Whipple dynamics."""
            x_s, heading_s, v_s, theta_s, theta_dot_s, k_s, delta_s, delta_dot_s = s
            trail_effect = self.trail * v_s * theta_s / self.wheelbase
            theta_ddot = (self.g / self.h) * theta_s \
                         - (v_s**2 / (self.h * self.wheelbase)) * delta_s \
                         - 2.0 * theta_dot_s
            delta_ddot = -self.omega_n**2 * (delta_s - trail_effect) \
                         - 2 * self.zeta * self.omega_n * delta_dot_s \
                         + u
            return np.array([
                v_s * heading_s,
                v_s * (k_s - delta_s / self.wheelbase),
                -0.1 * (v_s - self.speed),
                theta_dot_s,
                theta_ddot,
                0.0,
                delta_dot_s,
                delta_ddot,
            ])

        s = np.array([self._x, self._heading, self._v,
                       self._theta, self._theta_dot, 0.0,
                       self._delta, self._delta_dot])
        for _ in range(self._sub_steps):
            k1 = derivatives(s, u_total)
            k2 = derivatives(s + 0.5 * dt_sub * k1, u_total)
            k3 = derivatives(s + 0.5 * dt_sub * k2, u_total)
            k4 = derivatives(s + dt_sub * k3, u_total)
            s = s + (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Unpack
        self._x = s[0]
        self._heading = s[1]
        self._v = s[2]
        self._theta = np.clip(s[3], -1.57, 1.57)
        self._theta_dot = s[4]
        self._delta = np.clip(s[6], -0.785, 0.785)
        self._delta_dot = s[7]

        # Normalize heading
        while self._heading > math.pi:
            self._heading -= 2 * math.pi
        while self._heading < -math.pi:
            self._heading += 2 * math.pi

        # Update position (straight line)
        self._x += self._v * math.cos(self._heading) * self.dt
        self._y += self._v * math.sin(self._heading) * self.dt

        # ============================================================
        # New state
        # ============================================================
        new_ey = self._y
        new_epsi = self._heading
        new_dis_angle = self._target_theta - self._theta

        # ============================================================
        # Termination (reference: HRRL env.py line 312)
        # ============================================================
        terminated = False
        if abs(self._theta) > math.pi / 3:  # 60 degrees
            terminated = True

        # ============================================================
        # Reward (reference: HRRL env.py line 190-202)
        # ============================================================
        reward = compute_attitude_reward(prev_dis_angle, new_dis_angle,
                                         self._theta_dot, terminated)

        truncated = bool(self._step_count >= self.max_episode_steps)

        # ============================================================
        # Observation (8D, k=0 for straight line)
        # ============================================================
        obs = self._normalize(
            new_ey, new_epsi, self._v, self._theta, self._theta_dot,
            0.0, self._delta, self._delta_dot,
        )

        info = {
            "raw_state": np.array([new_ey, new_epsi, self._v, self._theta,
                                   self._theta_dot, 0.0, self._delta, self._delta_dot]),
            "u_total": u_total,
            "u_lqr": u_lqr,
            "target_theta": self._target_theta,
            "dis_angle": new_dis_angle,
            "epsilon": epsilon,
            "sigma": min(0.3, (1.08 ** self._episode_count) * 0.01),
            "episode_count": self._episode_count,
            "terminated": terminated,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        pass


if __name__ == "__main__":
    env = AttitudeControlEnv(max_episode_steps=1000)
    print(f"K_lqr: {env.K_lqr}")

    for ep in range(5):
        obs, info = env.reset(seed=ep)
        total_r = 0
        for step in range(1000):
            action = np.array([0.0])
            obs, r, t, tr, info = env.step(action)
            total_r += r
            if t or tr:
                break
        print(f"  Ep {ep}: return={total_r:.2f}, length={step+1}, "
              f"theta={info['raw_state'][3]:.4f}, "
              f"target_theta={info['target_theta']:.4f}, "
              f"sigma={info['sigma']:.4f}, ET={t}")
    env.close()
