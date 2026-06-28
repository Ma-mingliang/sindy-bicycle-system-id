"""Meijaard bicycle dynamics - canonical 4D implementation."""
import numpy as np
import math
from .config import DynamicsConfig


class BicycleDynamics:
    """Meijaard 2007 benchmark bicycle dynamics with RK4 integration."""

    def __init__(self, config: DynamicsConfig = None):
        self.config = config or DynamicsConfig()
        self._setup_matrices()

    def _setup_matrices(self):
        from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix
        p = {
            'IBxx': 11.3557360401, 'IBxz': -1.96756380745,
            'IByy': 12.2177848012, 'IBzz': 3.12354397008,
            'IFxx': 0.0904106601579, 'IFyy': 0.149389340425,
            'IHxx': 0.253379594731, 'IHxz': -0.0720452391817,
            'IHyy': 0.246138810935, 'IHzz': 0.0955770796289,
            'IRxx': 0.0883819364527, 'IRyy': 0.152467620286,
            'c': 0.0685808540382, 'g': 9.81, 'lam': 0.399680398707,
            'mB': 81.86, 'mF': 2.02, 'mH': 3.22, 'mR': 3.11,
            'rF': 0.34352982332, 'rR': 0.340958858855, 'w': 1.121,
            'xB': 0.289099434117, 'xH': 0.866949640247,
            'zB': -1.04029228321, 'zH': -0.748236400835,
        }
        from scipy.linalg import solve_continuous_are
        M, C1, K0, K2 = benchmark_par_to_canonical(p)
        self.M = M
        self.C1 = C1
        self.K0 = K0
        self.K2 = K2
        self.invM = np.linalg.inv(M)

        A, B = ab_matrix(M, C1, K0, K2, self.config.v0, self.config.g)
        reorder = [0, 2, 1, 3]
        self.A_r = A[np.ix_(reorder, reorder)]
        self.B_r = B[reorder, 1:2]
        Q = np.diag([1000.0, 100.0, 10.0, 1.0])
        R = np.array([[0.2]])
        P = solve_continuous_are(self.A_r, self.B_r, Q, R)
        self.K_lqr = (np.linalg.inv(R) @ self.B_r.T @ P).flatten()

    def _dynamics(self, s, tau):
        phi, delta, phi_dot, delta_dot = s
        q = np.array([[phi], [delta]])
        q_dot = np.array([[phi_dot], [delta_dot]])
        if abs(phi) < 1e-8:
            sin_ratio = 1.0 - phi**2 / 6.0
        else:
            sin_ratio = math.sin(phi) / phi
        K0_eff = self.K0.copy()
        K0_eff[0, 0] = self.K0[0, 0] * sin_ratio
        F = np.array([[0.0], [tau]])
        rhs = -self.C1 * self.config.v0 @ q_dot - (
            self.config.g * K0_eff + self.config.v0**2 * self.K2
        ) @ q + F
        q_dd = self.invM @ rhs
        return np.array([phi_dot, delta_dot, q_dd[0, 0], q_dd[1, 0]])

    def step(self, s: np.ndarray, tau: float) -> np.ndarray:
        dt_sub = self.config.dt / self.config.dt_sub
        s_cur = s.copy()
        for _ in range(self.config.dt_sub):
            k1 = self._dynamics(s_cur, tau)
            k2 = self._dynamics(s_cur + 0.5 * dt_sub * k1, tau)
            k3 = self._dynamics(s_cur + 0.5 * dt_sub * k2, tau)
            k4 = self._dynamics(s_cur + dt_sub * k3, tau)
            s_cur = s_cur + (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return s_cur

    def is_diverged(self, s: np.ndarray) -> bool:
        return abs(s[0]) > self.config.phi_limit or np.any(np.abs(s) > self.config.state_limit)

    def generate_lqr_tau(self, seg_idx: int, n_steps: int = 500,
                         target: float = None) -> callable:
        rng = np.random.RandomState(42 + seg_idx)
        disturbances = rng.uniform(-0.5, 0.5, n_steps)
        if target is None:
            target = np.clip(rng.normal(0, 0.15), -math.pi / 12, math.pi / 12)
        K = self.K_lqr

        def tau_func(step_i: int, s: np.ndarray) -> float:
            x_lqr = np.array([s[0] - target, s[2], s[1], s[3]])
            return float(-K @ x_lqr) + disturbances[step_i % n_steps]
        return tau_func
