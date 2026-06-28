"""Path tracking environment with HRRL hierarchical control.

Architecture (HRRL):
  1. Stanley controller → target steering angle (path tracking)
  2. steady_state_calculation → target roll angle
  3. LQR controller → tracks target roll angle (balance)
  4. RL residual → small correction on top of LQR

Control flow:
  ey, epsi, v → Stanley → δ_stanley → steady_state → θ_target
  θ, θ̇, δ, δ̇ → LQR(θ_target) → u_lqr
  action → u_residual = action × α
  u_total = u_lqr + u_residual

State (8D, normalized): [ey, epsi, v, theta, theta_dot, k, delta, delta_dot]
Action (1D): RL residual [-1, 1], scaled to steering torque residual

Stage 2 mode (stage1_mppi):
  - Outer loop: Stanley → target_roll_base, Stage 2 MPPI adds residual on target_roll
  - Inner loop: Stage 1 MPPI (frozen) computes steering torque to track target_roll
  - Matches HRRL: tar_attitude = stanley_roll + 0.3 * rl_roll
"""

import math
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import scipy.linalg as la
from reference_path import ReferencePath
from stanley_controller import steady_state_calculation
from reward_fn import compute_tracking_reward
from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix


def _build_sindy_library_numpy(state_norm, action_norm):
    """Build SINDy polynomial library row (numpy). Matches sindy_identification.py.

    Features: [1, x0..x8, x0*x0, x0*x1, ..., x8*x8]
    where x = [state_norm(8), action_norm(1)] = 9 features.
    Total: 1 + 9 + 9*10/2 = 55 terms.
    """
    x = np.concatenate([state_norm, [action_norm]])
    n = len(x)
    terms = [1.0]
    terms.extend(x)
    for i in range(n):
        for j in range(i, n):
            terms.append(x[i] * x[j])
    return np.array(terms, dtype=np.float64)


def _build_physics_informed_library_numpy(state_norm, action_norm):
    """Build physics-informed SINDy library row (numpy).

    Standard polynomial (55 terms) + 4 physics-informed terms:
      - theta * v^2 (centrifugal roll)
      - delta * v^2 (steering centrifugal)
      - delta * theta * v (roll-steering coupling)
      - theta^3 (nonlinear gravity)
    Total: 59 terms.
    """
    x = np.concatenate([state_norm, [action_norm]])
    n = len(x)
    terms = [1.0]
    terms.extend(x)
    for i in range(n):
        for j in range(i, n):
            terms.append(x[i] * x[j])

    # Physics-informed terms
    theta_norm = state_norm[3]
    v_norm = state_norm[2]
    delta_norm = state_norm[6]
    terms.append(theta_norm * v_norm**2)
    terms.append(delta_norm * v_norm**2)
    terms.append(delta_norm * theta_norm * v_norm)
    terms.append(theta_norm**3)

    return np.array(terms, dtype=np.float64)


def _compute_lqr_gains(g=9.81, h=0.8, v=3.0, L=1.0, wn=5.0, zeta=0.5,
                       trail=0.15, Q_diag=None, R_val=None):
    """Compute LQR gains for linearized bicycle dynamics.

    State: [theta - target_roll, theta_dot, delta, delta_dot]
    Input: steering torque u

    Dynamics (linearized Whipple):
      theta_ddot = (g/h)*theta - (v²/(h*L))*delta - 2*theta_dot + disturbance
      delta_ddot = -wn²*(delta - trail_effect) - 2*zeta*wn*delta_dot + u
    """
    A = np.array([
        [0, 1, 0, 0],
        [g/h, -2.0, -v**2/(h*L), 0],
        [0, 0, 0, 1],
        [wn**2*trail*v/L, 0, -wn**2, -2*zeta*wn]
    ])
    B = np.array([[0], [0], [0], [1.0]])
    Q = np.diag(Q_diag if Q_diag is not None else [100, 10, 10, 1])
    R = np.array([[R_val if R_val is not None else 1.0]])
    P = la.solve_continuous_are(A, B, Q, R)
    K = (np.linalg.inv(R) @ B.T @ P).flatten()
    return K  # [k_theta, k_theta_dot, k_delta, k_delta_dot]


class PathTrackingEnv(gym.Env):
    """Path tracking environment: Stanley + LQR + RL residual (HRRL).

    Hierarchical control:
      - Stanley: path tracking → target steering angle
      - steady_state_calculation: steering angle → target roll angle
      - LQR: balance control → tracks target roll angle
      - RL residual: small correction for optimization
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_episode_steps=1500, dt=1/30, speed=3.0,
                 residual_injection='final_action', residual_scale=0.15,
                 lqr_Q=None, lqr_R=None,
                 stage1_mppi_path=None, stage1_cfg=None,
                 dynamics_model='linearized',
                 disturbance_type='none', disturbance_params=None):
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.dt = dt
        self.speed = speed
        # Residual injection position: 'final_action', 'stanley_ref', 'theta_target', 'stage1_mppi'
        self.residual_injection = residual_injection
        self.residual_scale = float(residual_scale)
        self.wheelbase = 1.0
        self.g = 9.81
        self.h = 0.8
        self.omega_n = 5.0
        self.zeta = 0.5
        self.trail = 0.15
        self._sub_steps = 5  # RK4 sub-stepping (matching bicycle_env_analytical.py)

        # Disturbance configuration
        self.disturbance_type = disturbance_type  # 'none', 'wind', 'impulse', 'mixed'
        self._dist_params = disturbance_params or {}
        self._dist_rng = None
        self._current_wind = 0.0
        self._impulse_until = 0

        # Dynamics model: 'linearized' (Whipple), 'sindy' (data-driven), or 'physics_sindy'
        self.dynamics_model = dynamics_model
        self._sindy_Xi = None
        self._sindy_action_scale = 1.0
        self._norm_scale = np.array(
            [10.0, 1.57, 5.0, 1.57, 10.0, 1/8.0, 0.785, 3.0], dtype=np.float64)
        if dynamics_model == 'sindy':
            sindy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'sindy_model_improved.npz')
            if not os.path.exists(sindy_path):
                sindy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'sindy_model.npz')
            sindy_data = np.load(sindy_path, allow_pickle=True)
            self._sindy_Xi = sindy_data['coefficients']  # (55, 8)
            self._sindy_action_scale = float(sindy_data['action_scale'])
        elif dynamics_model == 'physics_sindy':
            sindy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'runs', 'physics_informed_dynamics',
                                      'physics_informed_sindy.npz')
            sindy_data = np.load(sindy_path, allow_pickle=True)
            self._sindy_Xi = sindy_data['coefficients']  # (59, 8)
            self._sindy_action_scale = float(sindy_data['action_scale'])

        # Meijaard 2007 benchmark bicycle (exact Whipple model)
        self._meijaard_M = None
        self._meijaard_C1 = None
        self._meijaard_K0 = None
        self._meijaard_K2 = None
        self._meijaard_params = None
        if dynamics_model == 'meijaard':
            self._meijaard_params = {
                'IBxx': 11.3557360401, 'IBxz': -1.96756380745,
                'IByy': 12.2177848012, 'IBzz': 3.12354397008,
                'IFxx': 0.0904106601579, 'IFyy': 0.149389340425,
                'IHxx': 0.253379594731, 'IHxz': -0.0720452391817,
                'IHyy': 0.246138810935, 'IHzz': 0.0955770796289,
                'IRxx': 0.0883819364527, 'IRyy': 0.152467620286,
                'c': 0.0685808540382, 'g': 9.81,
                'lam': 0.399680398707,
                'mB': 81.86, 'mF': 2.02, 'mH': 3.22, 'mR': 3.11,
                'rF': 0.34352982332, 'rR': 0.340958858855,
                'w': 1.121,
                'xB': 0.289099434117, 'xH': 0.866949640247,
                'zB': -1.04029228321, 'zH': -0.748236400835,
            }
            self._meijaard_M, self._meijaard_C1, self._meijaard_K0, self._meijaard_K2 = \
                benchmark_par_to_canonical(self._meijaard_params)
            # Update physical params from Meijaard
            self.wheelbase = self._meijaard_params['w']
            self.h = abs((-self._meijaard_params['rR'] * self._meijaard_params['mR'] +
                          self._meijaard_params['zB'] * self._meijaard_params['mB'] +
                          self._meijaard_params['zH'] * self._meijaard_params['mH'] -
                          self._meijaard_params['rF'] * self._meijaard_params['mF']) /
                         (self._meijaard_params['mR'] + self._meijaard_params['mB'] +
                          self._meijaard_params['mH'] + self._meijaard_params['mF']))
            self.trail = self._meijaard_params['c']

        # LQR gains (pre-computed, matching bicycle_env_analytical.py)
        if dynamics_model == 'meijaard' and self._meijaard_M is not None:
            # Use exact Meijaard A, B matrices for LQR
            # Meijaard state: [phi, delta, phi_dot, delta_dot]
            # Our state:      [phi, phi_dot, delta, delta_dot]
            # Need to reorder!
            A_m, B_m = ab_matrix(self._meijaard_M, self._meijaard_C1,
                                 self._meijaard_K0, self._meijaard_K2,
                                 self.speed, self.g)
            B_steer_meijaard = B_m[:, 1:2]  # steer torque column in Meijaard ordering

            # Reorder from [phi, delta, phi_dot, delta_dot] to [phi, phi_dot, delta, delta_dot]
            reorder = [0, 2, 1, 3]  # Meijaard idx -> our idx
            A_reordered = A_m[np.ix_(reorder, reorder)]
            B_reordered = B_steer_meijaard[reorder]

            Q = np.diag(lqr_Q if lqr_Q is not None else [1000, 100, 10, 1])
            R = np.array([[lqr_R if lqr_R is not None else 0.2]])
            P = la.solve_continuous_are(A_reordered, B_reordered, Q, R)
            self.K_lqr = (np.linalg.inv(R) @ B_reordered.T @ P).flatten()
            self._meijaard_A = A_m
            self._meijaard_B_steer = B_steer_meijaard
        else:
            self.K_lqr = _compute_lqr_gains(
                self.g, self.h, self.speed, self.wheelbase,
                self.omega_n, self.zeta, self.trail,
                Q_diag=lqr_Q, R_val=lqr_R,
            )

        # Stage 1 MPPI model (frozen inner-loop controller for Stage 2)
        self.stage1_agent = None
        if stage1_mppi_path is not None:
            from residual_mppi import ConservativeEnsembleResidualMPPI
            import json

            # Load config if not provided
            if stage1_cfg is None:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                cfg_path = os.path.join(base_dir, 'configs', 'stage1_best.json')
                with open(cfg_path) as f:
                    stage1_cfg = json.load(f)

            # Load SINDy coefficients
            sindy_data = np.load(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'sindy_model_improved.npz'))
            sindy_Xi = sindy_data['coefficients']
            action_scale = float(sindy_data['action_scale'])

            # Create and load Stage 1 agent
            cfg = {
                'obs_dim': 8, 'action_dim': 1, 'mlp_dim': 256,
                'ensemble_size': 3,
                'horizon': 6, 'num_samples': 64, 'num_elites': 8,
                'iterations': 4, 'temperature': 0.5,
                'epsilon_max': 0.1,
            }
            self.stage1_agent = ConservativeEnsembleResidualMPPI(
                cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
            self.stage1_agent.load(stage1_mppi_path)
            print(f"  Loaded Stage 1 model from {stage1_mppi_path}")

        # Action: RL residual on steering torque [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 8D observation
        self.observation_space = spaces.Box(
            low=np.array([-1., -1., -1., -1., -1., -1., -1., -1.], dtype=np.float32),
            high=np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.], dtype=np.float32),
            dtype=np.float32,
        )

        self.path = ReferencePath()

        # Position
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0
        self._v = speed

        # Roll dynamics
        self._theta = 0.0
        self._theta_dot = 0.0

        # Steering dynamics
        self._delta = 0.0
        self._delta_dot = 0.0

        self._step_count = 0

    def _get_disturbance(self):
        """Generate disturbance force for current step.

        Returns: disturbance force applied to theta_dot (roll dynamics).
          - 'wind': continuous sinusoidal wind + random walk
          - 'impulse': random lateral impulse pushes
          - 'mixed': alternating wind and impulse phases
        """
        if self.disturbance_type == 'none':
            return 0.0

        step = self._step_count
        dt = self.dt
        d = self._dist_params

        if self.disturbance_type == 'wind':
            # Sinusoidal wind + random walk
            wind_amp = d.get('wind_amp', 3.0)  # N·m
            wind_freq = d.get('wind_freq', 0.5)  # Hz
            noise_std = d.get('noise_std', 0.5)
            wind = wind_amp * math.sin(2 * math.pi * wind_freq * step * dt)
            self._current_wind += self._dist_rng.randn() * noise_std * dt
            self._current_wind *= 0.99  # decay
            return wind + self._current_wind

        elif self.disturbance_type == 'impulse':
            # Random impulse pushes
            impulse_prob = d.get('impulse_prob', 0.02)  # 2% chance per step
            impulse_mag = d.get('impulse_mag', 5.0)  # N·m
            if step < self._impulse_until:
                return self._impulse_force
            if self._dist_rng.random() < impulse_prob:
                self._impulse_force = self._dist_rng.uniform(-impulse_mag, impulse_mag)
                self._impulse_until = step + self._dist_rng.randint(3, 10)
                return self._impulse_force
            return 0.0

        elif self.disturbance_type == 'mixed':
            # Phase 1 (first half): wind, Phase 2 (second half): impulse
            max_steps = self.max_episode_steps
            if step < max_steps // 2:
                self.disturbance_type = 'wind'
                f = self._get_disturbance()
                self.disturbance_type = 'mixed'
                return f
            else:
                self.disturbance_type = 'impulse'
                f = self._get_disturbance()
                self.disturbance_type = 'mixed'
                return f

        return 0.0

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

        # Reset Stage 1 MPPI planning state
        if self.stage1_agent is not None:
            self.stage1_agent.reset_planning()

        # Reset disturbance state
        self._dist_rng = np.random.RandomState(seed if seed is not None else 0)
        self._current_wind = 0.0
        self._impulse_until = 0

        # Reset position
        y_offset = self.np_random.uniform(-0.5, 0.5)
        self._x = 0.0
        self._y = y_offset
        self._heading = 0.0
        self._v = self.speed + self.np_random.uniform(-0.3, 0.3)

        # Reset dynamics
        self._theta = self.np_random.uniform(-0.05, 0.05)
        self._theta_dot = 0.0
        self._delta = 0.0
        self._delta_dot = 0.0

        # Get initial path info
        fx = self._x + self.wheelbase * math.cos(self._heading)
        fy = self._y + self.wheelbase * math.sin(self._heading)
        path_info = self.path.get_closest_point(fx, fy, self._heading)

        obs = self._normalize(
            path_info['lateral_error'], path_info['course_error_angle'],
            self._v, self._theta, self._theta_dot,
            path_info['curvature'], self._delta, self._delta_dot,
        )
        return obs, {"path_info": path_info}

    def step(self, action):
        action_raw = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        action = float(np.clip(action_raw, -1.0, 1.0))
        action_was_clipped = abs(action_raw - action) > 1e-12
        self._step_count += 1

        # Current path errors
        fx = self._x + self.wheelbase * math.cos(self._heading)
        fy = self._y + self.wheelbase * math.sin(self._heading)
        path_info = self.path.get_closest_point(fx, fy, self._heading)
        ey = path_info['lateral_error']
        epsi = path_info['course_error_angle']
        k = path_info['curvature']

        # ============================================================
        # HRRL Hierarchical Control
        # ============================================================

        epsilon = action * self.residual_scale  # scaled residual
        delta_stanley_base = None
        delta_stanley = None
        target_roll_base = None
        target_roll = None

        if self.residual_injection == 'final_action':
            # --- A: Final-action residual (current) ---
            delta_stanley_base = -(math.atan(0.6 * ey / max(self._v, 0.5)) + 0.4 * epsi)
            delta_stanley = delta_stanley_base
            target_roll_base = steady_state_calculation(delta_stanley, self._v)
            target_roll = target_roll_base
            x_lqr = np.array([
                self._theta - target_roll,
                self._theta_dot,
                self._delta,
                self._delta_dot,
            ])
            u_lqr = float(-self.K_lqr @ x_lqr)
            u_residual = epsilon
            u_total = u_lqr + u_residual

        elif self.residual_injection == 'stanley_ref':
            # --- B: Stanley reference residual ---
            delta_stanley_base = -(math.atan(0.6 * ey / max(self._v, 0.5)) + 0.4 * epsi)
            delta_stanley = delta_stanley_base + epsilon  # residual on Stanley ref
            target_roll_base = steady_state_calculation(float(delta_stanley_base), self._v)
            target_roll = steady_state_calculation(float(delta_stanley), self._v)
            x_lqr = np.array([
                self._theta - target_roll,
                self._theta_dot,
                self._delta,
                self._delta_dot,
            ])
            u_lqr = float(-self.K_lqr @ x_lqr)
            u_residual = 0.0  # residual absorbed into reference
            u_total = u_lqr

        elif self.residual_injection == 'theta_target':
            # --- C: theta_target residual ---
            delta_stanley_base = -(math.atan(0.6 * ey / max(self._v, 0.5)) + 0.4 * epsi)
            delta_stanley = delta_stanley_base
            target_roll_base = steady_state_calculation(delta_stanley, self._v)
            target_roll = target_roll_base + epsilon  # residual on target roll
            x_lqr = np.array([
                self._theta - target_roll,
                self._theta_dot,
                self._delta,
                self._delta_dot,
            ])
            u_lqr = float(-self.K_lqr @ x_lqr)
            u_residual = 0.0  # residual absorbed into target
            u_total = u_lqr

        elif self.residual_injection == 'stage1_mppi':
            # --- D: Stage 2 dual-layer (HRRL architecture) ---
            # Outer loop: Stanley → target_roll_base, Stage 2 MPPI adds residual
            # Inner loop: Stage 1 MPPI (frozen) computes steering torque
            delta_stanley_base = -(math.atan(0.6 * ey / max(self._v, 0.5)) + 0.4 * epsi)
            delta_stanley = delta_stanley_base
            target_roll_base = steady_state_calculation(delta_stanley, self._v)
            # Stage 2 residual on target_roll (matching HRRL: tar_attitude = stanley + 0.3 * rl)
            target_roll = target_roll_base + 0.3 * epsilon

            # Inner loop: Stage 1 MPPI computes steering torque to track target_roll
            if self.stage1_agent is not None:
                # Build 8D observation for Stage 1 (same format as attitude_control_env)
                obs_stage1 = np.array([
                    self._y / 10.0,  # ey normalized
                    self._heading / 1.57,  # epsi normalized
                    self._v / 5.0,  # v normalized
                    self._theta / 1.57,  # theta normalized
                    self._theta_dot / 10.0,  # theta_dot normalized
                    0.0,  # k = 0 for straight line (Stage 1 doesn't track path)
                    self._delta / 0.785,  # delta normalized
                    self._delta_dot / 3.0,  # delta_dot normalized
                ], dtype=np.float32)

                # Stage 1 MPPI computes residual on steering torque
                eps_stage1, _, _, _ = self.stage1_agent.act(obs_stage1, eval_mode=True)
                alpha_stage1 = 0.15
                epsilon_stage1 = float(eps_stage1[0]) * alpha_stage1

                # LQR tracks target_roll, Stage 1 MPPI adds residual
                x_lqr = np.array([
                    self._theta - target_roll,
                    self._theta_dot,
                    self._delta,
                    self._delta_dot,
                ])
                u_lqr = float(-self.K_lqr @ x_lqr)
                u_total = u_lqr + epsilon_stage1
                u_residual = epsilon_stage1
            else:
                # Fallback: use LQR only
                x_lqr = np.array([
                    self._theta - target_roll,
                    self._theta_dot,
                    self._delta,
                    self._delta_dot,
                ])
                u_lqr = float(-self.K_lqr @ x_lqr)
                u_total = u_lqr
                u_residual = 0.0

        else:
            raise ValueError(f"Unknown residual_injection: {self.residual_injection}")

        # ============================================================
        # Dynamics
        # ============================================================
        dist_force = self._get_disturbance()

        if self.dynamics_model in ('sindy', 'physics_sindy') and self._sindy_Xi is not None:
            # --- SINDy dynamics: discrete-step polynomial model ---
            state_norm = np.array([
                np.clip(ey, -10, 10) / 10.0,
                np.clip(epsi, -1.57, 1.57) / 1.57,
                np.clip(self._v, -5, 5) / 5.0,
                np.clip(self._theta, -1.57, 1.57) / 1.57,
                np.clip(self._theta_dot, -10, 10) / 10.0,
                np.clip(0.0, -0.125, 0.125) * 8.0,
                np.clip(self._delta, -0.785, 0.785) / 0.785,
                np.clip(self._delta_dot, -3, 3) / 3.0,
            ], dtype=np.float64)
            a_norm = float(np.clip(u_total / (self._sindy_action_scale + 1e-8), -1.0, 1.0))
            if self.dynamics_model == 'physics_sindy':
                lib_row = _build_physics_informed_library_numpy(state_norm, a_norm)
            else:
                lib_row = _build_sindy_library_numpy(state_norm, a_norm)
            delta_norm = lib_row @ self._sindy_Xi
            delta_raw = delta_norm * self._norm_scale

            theta_new = self._theta + float(delta_raw[3])
            # Apply disturbance to theta_dot (convert force to acceleration: a = F / m, m ~ h/g)
            theta_dot_new = self._theta_dot + float(delta_raw[4]) + dist_force * self.dt
            delta_new = float(np.clip(self._delta + delta_raw[6], -0.785, 0.785))
            delta_dot_new = self._delta_dot + float(delta_raw[7])
            v_new = self._v - 0.1 * (self._v - self.speed) * self.dt
            heading_new = self._heading + self._v * (-self._delta / self.wheelbase) * self.dt
            x_new = self._x + self._v * math.cos(self._heading) * self.dt
            self._y += self._v * math.sin(self._heading) * self.dt

            s = np.array([x_new, heading_new, v_new,
                          theta_new, theta_dot_new, 0.0,
                          delta_new, delta_dot_new])
        elif self.dynamics_model == 'meijaard' and self._meijaard_M is not None:
            # --- Meijaard 2007 exact Whipple dynamics (RK4 + sub-stepping) ---
            # State-space: x_dot = A(v)*x + B_steer*u
            # x = [phi, delta, phi_dot, delta_dot]
            # Also includes kinematics for position/heading
            dt_sub = self.dt / self._sub_steps

            def derivatives_meijaard(s, u, dist_f):
                """Meijaard Whipple dynamics with disturbance."""
                ey_s, epsi_s, v_s, theta_s, theta_dot_s, k_s, delta_s, delta_dot_s = s

                # Compute A matrix at current speed
                A_m, B_m = ab_matrix(self._meijaard_M, self._meijaard_C1,
                                     self._meijaard_K0, self._meijaard_K2,
                                     max(v_s, 0.5), self.g)
                B_steer = B_m[:, 1:2]

                # State vector for Meijaard: [phi, delta, phi_dot, delta_dot]
                x_state = np.array([[theta_s], [delta_s],
                                    [theta_dot_s], [delta_dot_s]])

                # x_dot = A*x + B*u (+ disturbance on phi_dot)
                x_dot = A_m @ x_state + B_steer * u
                # Add disturbance to lean acceleration
                x_dot[2, 0] += dist_f

                return np.array([
                    v_s * epsi_s,                    # x_dot (position)
                    v_s * (k_s - delta_s / self.wheelbase),  # heading_dot
                    -0.1 * (v_s - self.speed),       # v_dot (speed recovery)
                    x_dot[0, 0],                     # theta_dot
                    x_dot[2, 0],                     # theta_ddot
                    0.0,                             # k_dot
                    x_dot[1, 0],                     # delta_dot
                    x_dot[3, 0],                     # delta_ddot
                ])

            s = np.array([self._x, self._heading, self._v,
                           self._theta, self._theta_dot, 0.0,
                           self._delta, self._delta_dot])
            for _ in range(self._sub_steps):
                k1 = derivatives_meijaard(s, u_total, dist_force)
                k2 = derivatives_meijaard(s + 0.5 * dt_sub * k1, u_total, dist_force)
                k3 = derivatives_meijaard(s + 0.5 * dt_sub * k2, u_total, dist_force)
                k4 = derivatives_meijaard(s + dt_sub * k3, u_total, dist_force)
                s = s + (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        else:
            # --- Linearized Whipple dynamics (RK4 + sub-stepping) ---
            dt_sub = self.dt / self._sub_steps

            def derivatives(s, u, dist_f):
                """Linearized Whipple dynamics with disturbance."""
                ey_s, epsi_s, v_s, theta_s, theta_dot_s, k_s, delta_s, delta_dot_s = s
                trail_effect = self.trail * v_s * theta_s / self.wheelbase
                theta_ddot = (self.g / self.h) * theta_s \
                             - (v_s**2 / (self.h * self.wheelbase)) * delta_s \
                             - 2.0 * theta_dot_s \
                             + dist_f  # disturbance force
                delta_ddot = -self.omega_n**2 * (delta_s - trail_effect) \
                             - 2 * self.zeta * self.omega_n * delta_dot_s \
                             + u
                return np.array([
                    v_s * epsi_s,
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
                k1 = derivatives(s, u_total, dist_force)
                k2 = derivatives(s + 0.5 * dt_sub * k1, u_total, dist_force)
                k3 = derivatives(s + 0.5 * dt_sub * k2, u_total, dist_force)
                k4 = derivatives(s + dt_sub * k3, u_total, dist_force)
                s = s + (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Unpack updated state
        if self.dynamics_model in ('sindy', 'physics_sindy'):
            # SINDy: s already contains updated values including kinematics
            self._x = s[0]
            self._heading = s[1]
            self._v = s[2]
            self._theta = float(s[3])
            self._theta_dot = s[4]
            self._delta = np.clip(s[6], -0.785, 0.785)
            self._delta_dot = s[7]
            # Normalize heading
            while self._heading > math.pi:
                self._heading -= 2 * math.pi
            while self._heading < -math.pi:
                self._heading += 2 * math.pi
        else:
            # Linearized: unpack from RK4 integration
            self._x = s[0]
            self._heading = s[1]
            self._v = s[2]
            self._theta = float(s[3])
            self._theta_dot = s[4]
            self._delta = np.clip(s[6], -0.785, 0.785)
            self._delta_dot = s[7]

            # Normalize heading
            while self._heading > math.pi:
                self._heading -= 2 * math.pi
            while self._heading < -math.pi:
                self._heading += 2 * math.pi

            # Update position (kinematics)
            self._x += self._v * math.cos(self._heading) * self.dt
            self._y += self._v * math.sin(self._heading) * self.dt

        theta_raw = self._theta
        theta_clipped = float(np.clip(theta_raw, -1.57, 1.57))
        theta_was_clipped = abs(theta_raw - theta_clipped) > 1e-12
        theta_raw_abs_exceeds_fall_limit = abs(theta_raw) > math.pi / 2
        # CRITICAL FIX: store raw theta for termination check BEFORE clipping
        self._theta_raw_for_termination = theta_raw
        self._theta = theta_clipped

        # ============================================================
        # Path errors (position updated by RK4 dynamics above)
        # ============================================================
        fx = self._x + self.wheelbase * math.cos(self._heading)
        fy = self._y + self.wheelbase * math.sin(self._heading)
        new_path_info = self.path.get_closest_point(fx, fy, self._heading)
        new_ey = new_path_info['lateral_error']
        new_epsi = new_path_info['course_error_angle']
        new_k = new_path_info['curvature']

        # ============================================================
        # Termination (check before reward so terminated flag is correct)
        # ============================================================
        terminated = False
        terminated_by = []
        # FIX: use raw theta for fall detection, not clipped theta
        # NOTE: pi/4 (45 deg) threshold for linearized model — the g/h=12.2625
        # destabilization makes pi/2 unreachable for LQR within meaningful episodes.
        if abs(self._theta_raw_for_termination) > math.pi / 4:
            terminated = True
            terminated_by.append("theta_raw")

        # ============================================================
        # Reward (shared HRRL-style: error reduction + smoothness)
        # ============================================================
        prev_raw = np.array([ey, epsi, self._v, self._theta,
                             self._theta_dot, k, self._delta, self._delta_dot])
        next_raw = np.array([new_ey, new_epsi, self._v, self._theta,
                             self._theta_dot, new_k, self._delta, self._delta_dot])
        reward = compute_tracking_reward(prev_raw, next_raw, u_total, 0.0, terminated)

        truncated = bool(self._step_count >= self.max_episode_steps)

        # ============================================================
        # Observation (8D)
        # ============================================================
        obs = self._normalize(
            new_ey, new_epsi, self._v, self._theta, self._theta_dot,
            new_k, self._delta, self._delta_dot,
        )

        info = {
            "path_info": new_path_info,
            "raw_state": np.array([new_ey, new_epsi, self._v, self._theta,
                                   self._theta_dot, new_k, self._delta, self._delta_dot]),
            "u_total": u_total,
            "u_lqr": u_lqr if self.residual_injection in ['final_action', 'stage1_mppi'] else None,
            "u_stanley": delta_stanley,
            "target_roll": target_roll,
            "target_roll_base": target_roll_base,
            "theta_target": target_roll,
            "theta_target_base": target_roll_base,
            "theta_target_delta": float(target_roll - target_roll_base) if target_roll_base is not None else 0.0,
            "delta_stanley_base": delta_stanley_base,
            "delta_ref": delta_stanley,
            "delta_ref_delta": float(delta_stanley - delta_stanley_base) if delta_stanley_base is not None else 0.0,
            "u_prior": u_lqr if self.residual_injection in ['final_action', 'stage1_mppi'] else u_total,
            "u_residual": u_residual,
            "epsilon": epsilon,
            "residual_scale": self.residual_scale,
            "action_raw": action_raw,
            "action_clipped": action,
            "action_was_clipped": action_was_clipped,
            "theta_raw": theta_raw,
            "theta_clipped": self._theta,
            "theta_was_clipped": theta_was_clipped,
            "theta_raw_abs_exceeds_fall_limit": theta_raw_abs_exceeds_fall_limit,
            "injection_mode": self.residual_injection,
            "success": float(abs(new_ey) < 0.05 and abs(self._theta) < 0.1),
            "terminated": terminated,
            "terminated_by": terminated_by,
            "fallback": abs(action) < 1e-8,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        pass


if __name__ == "__main__":
    import sys
    dynamics = sys.argv[1] if len(sys.argv) > 1 else 'linearized'
    print(f"=== Testing dynamics_model='{dynamics}' ===")
    env = PathTrackingEnv(max_episode_steps=1500, dynamics_model=dynamics)
    returns, lengths, ets = [], [], []
    for ep in range(10):
        obs, _ = env.reset(seed=ep)
        total_r = 0
        for step in range(1500):
            action = np.array([0.0])
            obs, r, t, tr, info = env.step(action)
            total_r += r
            if t or tr:
                break
        returns.append(total_r)
        lengths.append(step + 1)
        ets.append(t)
        path_info = info.get('path_info', {})
        print(f"  Ep {ep}: return={total_r:.2f}, length={step+1}, ET={t}, "
              f"ey={path_info.get('lateral_error', 0):.3f}, "
              f"theta_raw={info.get('theta_raw', 0):.4f}")
    print(f"\nZero-residual baseline ({dynamics}):")
    print(f"  return={np.mean(returns):.2f}+/-{np.std(returns):.2f}")
    print(f"  length={np.mean(lengths):.1f}+/-{np.std(lengths):.1f}")
    print(f"  ET rate={np.mean(ets):.0%}")
    env.close()
