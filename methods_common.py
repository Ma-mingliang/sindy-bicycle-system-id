"""公共基础设施：Meijaard参数、动力学函数、LQR控制器、训练数据生成。

所有12种方法共享此模块的常量和函数。
"""

import numpy as np
import math
from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix
from scipy.linalg import solve_continuous_are

# ============================================================
# Meijaard 2007 参数
# ============================================================
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
v0 = 3.5
g = 9.81
dt = 1 / 30

M, C1, K0, K2 = benchmark_par_to_canonical(p)
invM = np.linalg.inv(M)

# LQR控制器
A_lqr, B_lqr = ab_matrix(M, C1, K0, K2, v0, g)
reorder = [0, 2, 1, 3]
A_lqr_r = A_lqr[np.ix_(reorder, reorder)]
B_lqr_r = B_lqr[reorder, 1:2]
Q_lqr = np.diag([1000.0, 100.0, 10.0, 1.0])
R_lqr = np.array([[0.2]])
P_are = solve_continuous_are(A_lqr_r, B_lqr_r, Q_lqr, R_lqr)
K_lqr = (np.linalg.inv(R_lqr) @ B_lqr_r.T @ P_are).flatten()


# ============================================================
# 真实动力学（Meijaard非线性 + RK4，5子步）
# ============================================================
def nonlinear_dynamics(s: np.ndarray, tau: float, v: float = 3.5) -> np.ndarray:
    phi, delta, phi_dot, delta_dot = s
    q = np.array([[phi], [delta]])
    q_dot = np.array([[phi_dot], [delta_dot]])
    if abs(phi) < 1e-8:
        sin_ratio = 1.0 - phi**2 / 6.0
    else:
        sin_ratio = math.sin(phi) / phi
    K0_eff = K0.copy()
    K0_eff[0, 0] = K0[0, 0] * sin_ratio
    F = np.array([[0.0], [tau]])
    rhs = -C1 * v @ q_dot - (g * K0_eff + v**2 * K2) @ q + F
    q_dd = invM @ rhs
    return np.array([phi_dot, delta_dot, q_dd[0, 0], q_dd[1, 0]])


def nonlinear_step_rk4(s: np.ndarray, tau: float, dt_sub: float, v: float = 3.5) -> np.ndarray:
    k1 = nonlinear_dynamics(s, tau, v)
    k2 = nonlinear_dynamics(s + 0.5 * dt_sub * k1, tau, v)
    k3 = nonlinear_dynamics(s + 0.5 * dt_sub * k2, tau, v)
    k4 = nonlinear_dynamics(s + dt_sub * k3, tau, v)
    return s + (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def real_step(s: np.ndarray, tau: float) -> np.ndarray:
    """真实模型：5子步RK4积分。"""
    dt_sub = dt / 5
    s_cur = s.copy()
    for _ in range(5):
        s_cur = nonlinear_step_rk4(s_cur, tau, dt_sub)
    return s_cur


# ============================================================
# 训练数据生成
# ============================================================
def generate_training_data(n_samples: int = 30000, seed: int = 42):
    """生成随机(s, tau) → delta_s训练数据。

    Returns:
        states: (n, 4) [phi, delta, phi_dot, delta_dot]
        actions: (n,) tau
        deltas: (n, 4) s_next - s
        state_std: (4,)
        action_std: scalar
        delta_std: (4,)
    """
    rng = np.random.RandomState(seed)
    phis = rng.uniform(-0.5, 0.5, n_samples)
    deltas_angle = rng.uniform(-0.3, 0.3, n_samples)
    phi_dots = rng.uniform(-2.0, 2.0, n_samples)
    delta_dots = rng.uniform(-1.0, 1.0, n_samples)
    taus = rng.uniform(-50, 50, n_samples)

    states = np.column_stack([phis, deltas_angle, phi_dots, delta_dots])
    deltas = np.empty_like(states)
    for i in range(n_samples):
        s_next = real_step(states[i], taus[i])
        deltas[i] = s_next - states[i]

    state_std = np.std(states, axis=0)
    action_std = float(np.std(taus))
    delta_std = np.std(deltas, axis=0)

    return states, taus, deltas, state_std, action_std, delta_std
