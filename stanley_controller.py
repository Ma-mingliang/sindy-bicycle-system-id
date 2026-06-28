"""Stanley controller for bicycle path tracking.

Matches HRRL implementation:
  steering_angle = atan(k_e * lateral_error / v) + k_psi * course_error_angle
  target_roll = steady_state_calculation(steering_angle, v)
"""

import math
import numpy as np


def steady_state_calculation(steering_angle, v):
    """Convert steering angle to steady-state roll angle.

    From HRRL env.py - inverse problem: given steering angle and velocity,
    compute the required body roll angle for steady-state cornering.

    Args:
        steering_angle: front wheel steering angle (rad)
        v: velocity (m/s)

    Returns:
        target roll angle (rad)
    """
    if abs(v) < 0.1:
        return 0.0

    l = 1.489
    l1 = 0.7
    g = 9.8

    tan_delta = math.tan(steering_angle)
    numerator = tan_delta * (math.sqrt(l**2 + l1**2 * tan_delta**2) + 0.4407)
    denominator = (l / v)**2 * g

    if abs(denominator) < 1e-6:
        return 0.0

    theta = math.atan(numerator / denominator)
    return theta


def stanley_control(lateral_error, course_error_angle, v,
                    k_e=0.6, k_psi=0.4, max_roll=math.pi/6):
    """Stanley path tracking controller.

    Computes target roll angle from path tracking errors.

    Args:
        lateral_error: lateral deviation from path (m), positive = right of path
        course_error_angle: heading error (rad)
        v: velocity (m/s)
        k_e: lateral error gain (default 0.6, matching HRRL)
        k_psi: heading error gain (default 0.4, matching HRRL)
        max_roll: maximum target roll angle (rad, default pi/6 = 30 deg)

    Returns:
        target_roll: target roll angle (rad)
    """
    if abs(v) < 0.1:
        return 0.0

    # Stanley steering angle formula
    steering_angle = math.atan(k_e * lateral_error / v) + k_psi * course_error_angle

    # Convert steering angle to roll angle
    target_roll = steady_state_calculation(steering_angle, v)

    # Clamp to safe range
    target_roll = max(-max_roll, min(max_roll, target_roll))

    return target_roll
