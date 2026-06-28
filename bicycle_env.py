"""
Gymnasium-compatible bicycle balance environment using PyBullet physics.

Wraps HRRL's PyBullet URDF bicycle model with an 8D state space
matching the SINDy identification dimensions:
    [ey, eψ, v, θ, θ̇, k, δ, δ̇]

Designed for TD-MPC2 training.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import pybullet as p
import pybullet_data


def _find_urdf():
    """Locate the HRRL bicycle URDF file."""
    import os
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'HRRL', '3D', 'bike', 'urdf', 'bike.urdf'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'HRRL', '3D', 'bike', 'urdf', 'bike.urdf'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'HRRL', '3D', 'bike', 'urdf', 'bike.urdf'),
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "Cannot find HRRL bicycle URDF. Expected at: HRRL/3D/bike/urdf/bike.urdf"
    )


class BicycleBalanceEnv(gym.Env):
    """
    Bicycle balancing environment for TD-MPC2.

    State (8D, normalized to roughly [-1, 1]):
        [0] ey       - lateral error (m) / 10
        [1] epsi     - heading error (rad) / 1.57
        [2] v        - forward speed (m/s) / 5
        [3] theta    - roll angle (rad) / 1.57
        [4] theta_dot - roll rate (rad/s) / 10
        [5] k        - reference curvature (1/m) * 8
        [6] delta    - steering angle (rad) / 0.785
        [7] delta_dot - steering rate (rad/s) / 3

    Action (1D):
        Steering torque command in [-1, 1], mapped to handlebar position [-pi/4, pi/4].
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, max_episode_steps=300):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        # Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-1., -1., -1., -1., -1., -1., -1., -1.], dtype=np.float32),
            high=np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.], dtype=np.float32),
            dtype=np.float32,
        )

        # Normalization constants (matching SINDy pipeline)
        self._norm = np.array([10.0, 1.57, 5.0, 1.57, 10.0, 1/8.0, 0.785, 3.0], dtype=np.float32)

        # PyBullet setup
        use_gui = (render_mode == "human")
        self._cid = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._cid)
        p.setTimeStep(1.0 / 30.0, physicsClientId=self._cid)
        p.setGravity(0, 0, -9.8, physicsClientId=self._cid)

        # Load ground and bicycle
        p.loadURDF("plane.urdf", globalScaling=2, physicsClientId=self._cid)
        urdf_path = _find_urdf()
        self._bike = p.loadURDF(
            urdf_path, [0, 0, 1], p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self._cid,
        )

        # Wheel and handlebar dynamics
        for link_idx in [-1, 0, 1, 2]:
            p.changeDynamics(
                self._bike, link_idx,
                restitution=0.5, contactStiffness=1e8, contactDamping=1e5,
                physicsClientId=self._cid,
            )

        # State tracking
        self._step_count = 0
        self._theta_old = 0.0
        self._delta_old = 0.0
        self._cumulative_reward = 0.0

    def _get_raw_state(self):
        """Read raw (unnormalized) physics state from PyBullet."""
        # Roll angle from base orientation
        _, orn = p.getBasePositionAndOrientation(self._bike, physicsClientId=self._cid)
        euler = p.getEulerFromQuaternion(orn)
        theta = euler[0]  # roll

        # Roll rate via finite difference
        theta_dot = (theta - self._theta_old) / (1.0 / 30.0)

        # Forward speed from base velocity
        vel = p.getBaseVelocity(self._bike, physicsClientId=self._cid)[0]
        v = math.sqrt(vel[0] ** 2 + vel[1] ** 2)

        # Steering angle and rate from handlebar joint (joint 1)
        joint_state = p.getJointState(self._bike, 1, physicsClientId=self._cid)
        delta = joint_state[0]
        delta_dot = (delta - self._delta_old) / (1.0 / 30.0)

        # Lateral position (approximate ey) and heading (approximate epsi)
        # For a straight reference path: ey ~ y-position, epsi ~ yaw
        yaw = euler[2]
        pos, _ = p.getBasePositionAndOrientation(self._bike, physicsClientId=self._cid)
        ey = pos[1]  # lateral displacement from straight path
        epsi = yaw    # heading error from straight path

        # Reference curvature (straight path = 0)
        k_ref = 0.0

        return np.array([ey, epsi, v, theta, theta_dot, k_ref, delta, delta_dot], dtype=np.float64)

    def _normalize(self, raw):
        """Normalize raw state to [-1, 1] range."""
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
        self._cumulative_reward = 0.0

        # Reset bicycle to upright position with small random perturbation
        start_pos = [0, 0, 0.75]
        roll = self.np_random.uniform(-0.05, 0.05)
        start_orn = p.getQuaternionFromEuler([roll, 0, 0])
        p.resetBasePositionAndOrientation(self._bike, start_pos, start_orn, physicsClientId=self._cid)
        p.resetBaseVelocity(self._bike, [0, 0, 0], [0, 0, 0], physicsClientId=self._cid)

        # Reset joints
        p.resetJointState(self._bike, 0, 0, 0, physicsClientId=self._cid)
        p.resetJointState(self._bike, 1, 0, 0, physicsClientId=self._cid)
        p.resetJointState(self._bike, 2, 0, 0, physicsClientId=self._cid)

        # Drive wheels forward at constant speed
        p.setJointMotorControl2(self._bike, 0, p.VELOCITY_CONTROL, targetVelocity=-10, force=20, physicsClientId=self._cid)
        p.setJointMotorControl2(self._bike, 2, p.VELOCITY_CONTROL, targetVelocity=-10, force=20, physicsClientId=self._cid)

        p.stepSimulation(physicsClientId=self._cid)

        self._theta_old = 0.0
        self._delta_old = 0.0

        raw = self._get_raw_state()
        obs = self._normalize(raw)
        return obs, {"raw_state": raw}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._step_count += 1

        # Store previous state for reward
        raw_prev = self._get_raw_state()

        # Apply steering: action maps to handlebar position control
        target_delta = float(action[0]) * (math.pi / 4)
        p.setJointMotorControl2(
            self._bike, 1, p.POSITION_CONTROL,
            targetPosition=target_delta, force=200, positionGain=0.3,
            physicsClientId=self._cid,
        )

        # Keep wheels spinning
        p.setJointMotorControl2(self._bike, 0, p.VELOCITY_CONTROL, targetVelocity=-10, force=20, physicsClientId=self._cid)
        p.setJointMotorControl2(self._bike, 2, p.VELOCITY_CONTROL, targetVelocity=-10, force=20, physicsClientId=self._cid)

        p.stepSimulation(physicsClientId=self._cid)

        # Update state tracking
        _, orn = p.getBasePositionAndOrientation(self._bike, physicsClientId=self._cid)
        euler = p.getEulerFromQuaternion(orn)
        self._theta_old = euler[0]
        joint_state = p.getJointState(self._bike, 1, physicsClientId=self._cid)
        self._delta_old = joint_state[0]

        raw = self._get_raw_state()
        obs = self._normalize(raw)

        # Reward: balance-focused
        reward = self._compute_reward(raw_prev, raw)
        self._cumulative_reward += reward

        # Termination conditions
        theta = raw[3]
        ey = raw[0]
        terminated = bool(abs(theta) > 0.5 or abs(ey) > 3.0)
        truncated = bool(self._step_count >= self.max_episode_steps)

        info = {
            "raw_state": raw,
            "success": float(abs(theta) < 0.05 and abs(ey) < 0.5),
            "terminated": terminated,
            "cumulative_reward": self._cumulative_reward,
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, raw_prev, raw):
        """
        Reward for balancing:
        - Small roll angle and roll rate
        - Error reduction when large
        - Stability bonus when close to balanced
        """
        theta = raw[3]
        theta_dot = raw[4]
        ey = raw[0]

        # Stability reward when nearly balanced
        if abs(theta) < 0.02:
            reward = 0.1 - abs(theta_dot) * 0.5
        else:
            # Reward error reduction
            reward = abs(raw_prev[3]) - abs(theta)

        # Penalty for lateral drift
        reward -= 0.01 * abs(ey)

        return float(reward)

    def close(self):
        p.disconnect(self._cid)


# Quick smoke test
if __name__ == "__main__":
    env = BicycleBalanceEnv(max_episode_steps=100)
    obs, info = env.reset()
    print(f"Initial obs shape: {obs.shape}, values: {obs}")

    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode ended at step {step+1}, terminated={terminated}, total_reward={total_reward:.2f}")
            break

    env.close()
    print("Smoke test passed.")
