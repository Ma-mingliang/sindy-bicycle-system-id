"""
Whipple bicycle dynamics model for data generation.

Implements the standard Whipple bicycle model with realistic parameters
from Jason Moore's benchmark bicycle (TU Delft).

State vector for path tracking (8D outer-loop):
    x = [ey, eψ, v, θ, θ̇, k, δ, δ̇]
where:
    ey   - lateral error (m)
    eψ   - heading error (rad)
    v    - longitudinal speed (m/s)
    θ    - roll angle (rad)
    θ̇    - roll angular velocity (rad/s)
    k    - reference path curvature (1/m)
    δ    - steering angle (rad)
    δ̇    - steering angular velocity (rad/s)

Control input:
    a    - steering torque command (N·m)
"""

import numpy as np
# from scipy.integrate import solve_ivp  # Not needed, using custom RK4


# Physical parameters (Moore benchmark bicycle, ~2013)
class BicycleParams:
    """Bicycle physical parameters."""
    def __init__(self):
        # Geometry
        self.wheelbase = 1.02        # m, wheelbase
        self.head_angle = 1.3963     # rad, head angle (~70 deg from horizontal)
        self.trail = 0.08            # m, trail (mechanical)
        self.steer_axis_tilt = np.pi/2 - self.head_angle  # from vertical

        # Mass and inertia
        self.m_rider = 70.0          # kg, rider mass
        self.m_bike = 25.0           # kg, bike mass
        self.m_total = self.m_rider + self.m_bike

        # Center of mass positions (from rear wheel contact)
        self.x_rider = 0.44          # m
        self.z_rider = 1.07          # m
        self.x_bike = 0.42           # m
        self.z_bike = 0.46           # m

        # Rear wheel
        self.r_rear = 0.3            # m, rear wheel radius
        self.m_rear = 2.0            # kg
        self.I_rear = 0.0603         # kg·m^2 (spin)
        self.I_rear_d = 0.0          # kg·m^2 (diametral)

        # Front wheel
        self.r_front = 0.35          # m, front wheel radius
        self.m_front = 3.0           # kg
        self.I_front = 0.1405        # kg·m^2 (spin)
        self.I_front_d = 0.0         # kg·m^2 (diametral)

        # Front assembly (fork + handlebar)
        self.m_front_assembly = 4.0  # kg
        self.I_front_assembly = 0.06 # kg·m^2

        # Effective parameters for simplified model
        self.g = 9.81                # m/s^2

        # Compute equivalent height and other derived params
        self._compute_derived()

    def _compute_derived(self):
        """Compute derived parameters for simplified dynamics."""
        # Equivalent COM height
        self.h_com = (self.m_rider * self.z_rider + self.m_bike * self.z_bike) / self.m_total

        # Equivalent COM horizontal position
        self.x_com = (self.m_rider * self.x_rider + self.m_bike * self.x_bike) / self.m_total

        # Distance from rear wheel to COM
        self.l1 = self.x_com
        # Distance from COM to front wheel
        self.l2 = self.wheelbase - self.x_com

        # Steering geometry leverage
        self.c = self.trail + self.r_rear * np.sin(self.steer_axis_tilt)


def whipple_eom(params, state, delta_torque):
    """
    Whipple bicycle equations of motion (simplified for path tracking).

    Returns the time derivatives of the full bicycle state.
    This is used for high-fidelity data generation.

    Parameters
    ----------
    params : BicycleParams
        Physical parameters
    state : ndarray, shape (10,)
        [x, y, psi, v, theta, theta_dot, delta, delta_dot, phi_f, omega_f]
        Note: wheel dynamics are simplified
    delta_torque : float
        Steering torque input

    Returns
    -------
    dstate : ndarray, shape (10,)
        Time derivatives
    """
    p = params
    x, y, psi, v, theta, theta_dot, delta, delta_dot, phi_f, omega_f = state

    # Simplified Whipple model (valid for moderate speeds and angles)
    # This captures the key roll-steering coupling

    # Gravity component on roll
    gravity_roll = p.m_total * p.g * p.h_com * np.sin(theta)

    # Centrifugal effect (simplified)
    # At speed v with steering delta, the front wheel generates lateral force
    centrifugal = p.m_total * v**2 * np.tan(delta) / p.wheelbase

    # Steering inertia coupling
    I_roll = p.m_total * p.h_com**2  # roll inertia (simplified)

    # Roll dynamics: coupling with steering
    # The key coupling: steering creates lateral force -> roll torque
    # And roll creates gravitational torque -> steering tendency
    roll_coupling = p.h_com * np.sin(p.steer_axis_tilt) / p.wheelbase

    # Roll angular acceleration
    theta_ddot = (
        gravity_roll / I_roll +
        centrifugal * p.h_com * np.cos(theta) / I_roll -
        p.c * delta * p.m_total * p.g / (I_roll * p.wheelbase) +
        0.1 * delta_dot * v  # gyroscopic coupling approximation
    )

    # Steering dynamics
    I_steer = p.m_front_assembly * p.c**2 + p.I_front  # steer inertia (simplified)

    # Steering angular acceleration
    delta_ddot = (
        delta_torque / I_steer -
        p.m_total * p.g * p.c * np.sin(theta) / (I_steer * p.wheelbase) +
        p.m_total * v**2 * p.c * delta / (I_steer * p.wheelbase**2) -
        2.0 * delta_dot  # damping
    )

    # Kinematic bicycle model for position
    beta = np.arctan(p.l2 * np.tan(delta) / p.wheelbase)  # slip angle (simplified)

    dx = v * np.cos(psi + beta)
    dy = v * np.sin(psi + beta)
    dpsi = v * np.tan(delta) * np.cos(beta) / p.wheelbase

    # Wheel rotation (simplified)
    dphi_f = omega_f
    domega_f = -0.1 * omega_f  # free rolling with slight friction

    return np.array([dx, dy, dpsi, 0, theta_dot, theta_ddot, delta_dot, delta_ddot, dphi_f, domega_f])


def simulate_bicycle(params, x0, t_span, dt, control_func):
    """
    Simulate the bicycle system.

    Parameters
    ----------
    params : BicycleParams
    x0 : ndarray, shape (10,)
        Initial state
    t_span : tuple (t_start, t_end)
    dt : float
        Time step
    control_func : callable
        control_func(t, state) -> delta_torque

    Returns
    -------
    t : ndarray
    x : ndarray, shape (n_steps, 10)
    """
    n_steps = int((t_span[1] - t_span[0]) / dt)
    t = np.linspace(t_span[0], t_span[1], n_steps)
    x = np.zeros((n_steps, 10))
    x[0] = x0

    for i in range(n_steps - 1):
        u = control_func(t[i], x[i])
        # Use RK4 for integration
        k1 = whipple_eom(params, x[i], u)
        k2 = whipple_eom(params, x[i] + 0.5*dt*k1, u)
        k3 = whipple_eom(params, x[i] + 0.5*dt*k2, u)
        k4 = whipple_eom(params, x[i] + dt*k3, u)
        x[i+1] = x[i] + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # Enforce reasonable bounds (prevent numerical explosion)
        x[i+1] = np.clip(x[i+1], -100, 100)

    return t, x


def full_state_to_path_tracking(x_full, ref_path_curvature=0.0):
    """
    Convert full bicycle state to 8D path-tracking state.

    Parameters
    ----------
    x_full : ndarray, shape (10,)
        [x, y, psi, v, theta, theta_dot, delta, delta_dot, phi_f, omega_f]
    ref_path_curvature : float
        Curvature of reference path (1/m)

    Returns
    -------
    s : ndarray, shape (8,)
        [ey, eψ, v, theta, theta_dot, k, delta, delta_dot]
    """
    x, y, psi, v, theta, theta_dot, delta, delta_dot, _, _ = x_full

    # For simplicity, assume reference path is along x-axis
    # lateral error = y position (perpendicular to path)
    ey = y
    # heading error = yaw angle deviation
    epsi = psi

    return np.array([ey, epsi, v, theta, theta_dot, ref_path_curvature, delta, delta_dot])
