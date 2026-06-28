"""Shared reward function for path tracking.

Used by both path_tracking_env.py and residual_mppi.py to ensure
MPPI's internal reward matches the environment's reward exactly.

raw_state = [ey, epsi, v, theta, theta_dot, k, delta, delta_dot]
  indices:    0    1     2   3      4         5   6       7
"""

import numpy as np


def compute_tracking_reward(prev_raw, next_raw, u_total=0.0, u_prev=0.0,
                            terminated=False):
    """HRRL-style tracking reward.

    Args:
        prev_raw: raw state before step, shape (8,) — [ey, epsi, v, theta, theta_dot, k, delta, delta_dot]
        next_raw: raw state after step, shape (8,)
        u_total: total control input (unused in base HRRL, available for regularization)
        u_prev: previous control input (unused in base HRRL, available for smoothness)
        terminated: whether episode terminated early (penalize)

    Returns:
        reward: scalar
    """
    ey = abs(prev_raw[0])
    new_ey = abs(next_raw[0])
    new_epsi = abs(next_raw[1])
    new_theta = abs(next_raw[3])
    theta_dot = abs(next_raw[4])

    ey_threshold = 0.1

    if terminated:
        # Terminal penalty: -10 matching HRRL (was -1.0, too small)
        reward = -10.0 - 0.01 * theta_dot
    elif ey < ey_threshold:
        # Near path: reward smoothness and stability, penalize lean
        reward = 0.1 - 0.3 * new_epsi - 0.1 * new_theta - 0.01 * theta_dot
    else:
        # Far from path: reward error reduction, penalize lean
        reward = ey - new_ey - 0.1 * new_theta - 0.01 * theta_dot

    return reward


def compute_attitude_reward(prev_dis_angle, new_dis_angle, theta_dot,
                            terminated=False):
    """Stage 1 attitude control reward (reference: HRRL env.py line 190-202).

    Args:
        prev_dis_angle: target_theta - theta before step (scalar)
        new_dis_angle: target_theta - theta after step (scalar)
        theta_dot: roll angular velocity after step (scalar)
        terminated: whether episode terminated early

    Returns:
        reward: scalar
    """
    if terminated:
        return -1.0 - 0.01 * abs(theta_dot)
    elif abs(new_dis_angle) < 0.002:
        return 0.1 - abs(theta_dot)
    else:
        return abs(prev_dis_angle) - abs(new_dis_angle)


def compute_attitude_reward_torch(prev_dis_angle, new_dis_angle, theta_dot,
                                   terminated=None):
    """PyTorch batch version of compute_attitude_reward for MPPI planning.

    Args:
        prev_dis_angle: (batch,) tensor
        new_dis_angle: (batch,) tensor
        theta_dot: (batch,) tensor
        terminated: (batch,) bool tensor or None

    Returns:
        reward: (batch,) tensor
    """
    import torch

    near_target = new_dis_angle.abs() < 0.002
    reward_near = 0.1 - theta_dot.abs()
    reward_far = prev_dis_angle.abs() - new_dis_angle.abs()
    reward = torch.where(near_target, reward_near, reward_far)

    if terminated is not None:
        reward = torch.where(terminated,
                             -1.0 - 0.01 * theta_dot.abs(),
                             reward)

    return reward


def compute_tracking_reward_torch(prev_raw, next_raw, u_total=None, u_prev=None,
                                   terminated=None):
    """PyTorch version of compute_tracking_reward for batch computation.

    Args:
        prev_raw: (batch, 8) tensor
        next_raw: (batch, 8) tensor
        u_total: unused
        u_prev: unused
        terminated: (batch,) bool tensor or None

    Returns:
        reward: (batch,) tensor
    """
    import torch

    ey = prev_raw[:, 0].abs()
    new_ey = next_raw[:, 0].abs()
    new_epsi = next_raw[:, 1].abs()
    new_theta = next_raw[:, 3].abs()
    theta_dot = next_raw[:, 4].abs()

    ey_threshold = 0.1

    # Base reward (with theta penalty)
    near_path = ey < ey_threshold
    reward_near = 0.1 - 0.3 * new_epsi - 0.1 * new_theta - 0.01 * theta_dot
    reward_far = ey - new_ey - 0.1 * new_theta - 0.01 * theta_dot
    reward = torch.where(near_path, reward_near, reward_far)

    # Terminal penalty (-10 matching HRRL)
    if terminated is not None:
        reward = torch.where(terminated,
                             -10.0 - 0.01 * theta_dot,
                             reward)

    return reward
