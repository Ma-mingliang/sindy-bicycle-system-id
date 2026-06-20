"""测试SINDy加入三角函数库后能否拟合非线性动力学。

测试：
1. 标准多项式库 vs 多项式+sin/cos库的拟合精度
2. 不同角度下的单步预测精度
3. 多步rollout精度对比
"""

import numpy as np
import math
from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix
from sklearn.linear_model import orthogonal_mp_gram


# ============================================================
# Meijaard 参数
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


# ============================================================
# 动力学函数
# ============================================================
def nonlinear_dynamics(s, tau, v):
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


def nonlinear_step_rk4(s, tau, dt_sub, v=3.5):
    k1 = nonlinear_dynamics(s, tau, v)
    k2 = nonlinear_dynamics(s + 0.5 * dt_sub * k1, tau, v)
    k3 = nonlinear_dynamics(s + 0.5 * dt_sub * k2, tau, v)
    k4 = nonlinear_dynamics(s + dt_sub * k3, tau, v)
    return s + (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def real_step(s, tau):
    dt_sub = dt / 5
    s_cur = s.copy()
    for _ in range(5):
        s_cur = nonlinear_step_rk4(s_cur, tau, dt_sub)
    return s_cur


# ============================================================
# SINDy库函数
# ============================================================
def build_poly_library(states, actions):
    """标准多项式库：常数+线性+二次交叉项。"""
    n = len(states)
    x = np.column_stack([states, actions.reshape(-1, 1)])  # (n, 5)
    features = [np.ones(n)]  # 常数
    for i in range(5):
        features.append(x[:, i])
    for i in range(5):
        for j in range(i, 5):
            features.append(x[:, i] * x[:, j])
    return np.column_stack(features)  # (n, 21)


def build_trig_library(states, actions):
    """多项式+三角函数库：标准多项式 + sin(phi) + cos(phi) + sin(phi)*其他项。"""
    n = len(states)
    phi = states[:, 0]
    x = np.column_stack([states, actions.reshape(-1, 1)])  # (n, 5)
    features = [np.ones(n)]  # 常数

    # 线性项
    for i in range(5):
        features.append(x[:, i])

    # 二次交叉项
    for i in range(5):
        for j in range(i, 5):
            features.append(x[:, i] * x[:, j])

    # 三角函数项（避免高阶多项式溢出）
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    features.append(sin_phi)                    # sin(phi)
    features.append(cos_phi)                    # cos(phi)
    features.append(sin_phi * phi)              # sin(phi)*phi
    features.append(sin_phi * x[:, 1])          # sin(phi)*delta
    features.append(sin_phi * x[:, 2])          # sin(phi)*theta_dot
    features.append(sin_phi * x[:, 3])          # sin(phi)*delta_dot
    features.append(sin_phi * x[:, 4])          # sin(phi)*tau
    features.append(cos_phi * phi)              # cos(phi)*phi
    features.append(cos_phi * x[:, 2])          # cos(phi)*theta_dot
    features.append(phi * phi * phi)            # phi^3 (用乘法避免溢出)

    return np.column_stack(features)  # (n, 32)


# ============================================================
# 生成训练数据
# ============================================================
print("生成训练数据...")
np.random.seed(42)
n_samples = 30000

phis = np.random.uniform(-0.5, 0.5, n_samples)
deltas = np.random.uniform(-0.3, 0.3, n_samples)
theta_dots = np.random.uniform(-2.0, 2.0, n_samples)
delta_dots = np.random.uniform(-1.0, 1.0, n_samples)
taus = np.random.uniform(-50, 50, n_samples)

states = np.column_stack([phis, deltas, theta_dots, delta_dots])

# 计算真实delta（5子步 - 初始状态）
deltas_real = []
for i in range(n_samples):
    s = states[i]
    tau = taus[i]
    s_next = real_step(s, tau)
    deltas_real.append(s_next - s)
deltas_real = np.array(deltas_real)

print(f"  样本数: {n_samples}")
print(f"  delta范围: {np.min(deltas_real, axis=0)} ~ {np.max(deltas_real, axis=0)}")

# 归一化
state_std = np.std(states, axis=0)
action_std = np.std(taus)
delta_std = np.std(deltas_real, axis=0)

states_norm = states / state_std
actions_norm = taus / action_std
deltas_norm = deltas_real / delta_std


# ============================================================
# STLSQ稀疏回归
# ============================================================
def stlsq(Theta, dXdt, threshold=0.05, max_iter=20):
    """Sequential Thresholded Least Squares。"""
    n_features = Theta.shape[1]
    n_states = dXdt.shape[1]
    Xi = np.zeros((n_features, n_states))

    for col in range(n_states):
        y = dXdt[:, col]
        coef = np.linalg.lstsq(Theta, y, rcond=None)[0]
        for _ in range(max_iter):
            small = np.abs(coef) < threshold
            coef[small] = 0
            big = ~small
            if not np.any(big):
                break
            coef[big] = np.linalg.lstsq(Theta[:, big], y, rcond=None)[0]
        Xi[:, col] = coef

    return Xi


# ============================================================
# 拟合标准多项式库
# ============================================================
print(f"\n{'='*80}")
print("标准多项式库（21个特征）")
print("=" * 80)

Theta_poly = build_poly_library(states_norm, actions_norm)
Xi_poly = stlsq(Theta_poly, deltas_norm, threshold=0.05)

feature_names_poly = ['1']
for i in range(5):
    feature_names_poly.append(f'x{i}')
for i in range(5):
    for j in range(i, 5):
        feature_names_poly.append(f'x{i}*x{j}')

print(f"\n非零系数:")
state_names = ['theta', 'delta', 'theta_dot', 'delta_dot']
for col, sname in enumerate(state_names):
    nonzero = np.where(np.abs(Xi_poly[:, col]) > 1e-6)[0]
    if len(nonzero) > 0:
        terms = [f"{Xi_poly[i, col]:+.4f}*{feature_names_poly[i]}" for i in nonzero]
        print(f"  {sname}: {' '.join(terms)}")
    else:
        print(f"  {sname}: (全零)")

# 预测误差
pred_poly = Theta_poly @ Xi_poly
rmse_poly = np.sqrt(np.mean((pred_poly - deltas_norm) ** 2, axis=0))
print(f"\n归一化RMSE: {rmse_poly}")
rmse_poly_phys = rmse_poly * delta_std
print(f"物理RMSE: {rmse_poly_phys}")


# ============================================================
# 拟合多项式+三角函数库
# ============================================================
print(f"\n{'='*80}")
print("多项式+三角函数库（33个特征）")
print("=" * 80)

Theta_trig = build_trig_library(states_norm, actions_norm)
Xi_trig = stlsq(Theta_trig, deltas_norm, threshold=0.05)

feature_names_trig = ['1']
for i in range(5):
    feature_names_trig.append(f'x{i}')
for i in range(5):
    for j in range(i, 5):
        feature_names_trig.append(f'x{i}*x{j}')
feature_names_trig.extend([
    'sin(phi)', 'cos(phi)', 'sin(phi)*phi',
    'sin(phi)*delta', 'sin(phi)*w', 'sin(phi)*wd', 'sin(phi)*tau',
    'cos(phi)*phi', 'cos(phi)*w',
    'phi^3'
])

print(f"\n非零系数:")
for col, sname in enumerate(state_names):
    nonzero = np.where(np.abs(Xi_trig[:, col]) > 1e-6)[0]
    if len(nonzero) > 0:
        terms = [f"{Xi_trig[i, col]:+.4f}*{feature_names_trig[i]}" for i in nonzero]
        print(f"  {sname}: {' '.join(terms)}")
    else:
        print(f"  {sname}: (全零)")

# 预测误差
pred_trig = Theta_trig @ Xi_trig
rmse_trig = np.sqrt(np.mean((pred_trig - deltas_norm) ** 2, axis=0))
print(f"\n归一化RMSE: {rmse_trig}")
rmse_trig_phys = rmse_trig * delta_std
print(f"物理RMSE: {rmse_trig_phys}")


# ============================================================
# 多步rollout对比
# ============================================================
print(f"\n{'='*80}")
print("多步rollout对比（5段×500步）")
print("=" * 80)

from scipy.linalg import solve_continuous_are

A_lqr, B_lqr = ab_matrix(M, C1, K0, K2, v0, g)
reorder = [0, 2, 1, 3]
A_lqr_r = A_lqr[np.ix_(reorder, reorder)]
B_lqr_r = B_lqr[reorder, 1:2]
Q = np.diag([1000.0, 100.0, 10.0, 1.0])
R_mat = np.array([[0.2]])
P = solve_continuous_are(A_lqr_r, B_lqr_r, Q, R_mat)
K_lqr = (np.linalg.inv(R_mat) @ B_lqr_r.T @ P).flatten()


def sindy_poly_step(s, tau):
    s_norm = s / state_std
    a_norm = tau / action_std
    lib = build_poly_library(s_norm.reshape(1, -1), np.array([a_norm]))
    delta_norm = (lib @ Xi_poly)[0]
    return s + delta_norm * delta_std


def sindy_trig_step(s, tau):
    s_norm = s / state_std
    a_norm = tau / action_std
    lib = build_trig_library(s_norm.reshape(1, -1), np.array([a_norm]))
    delta_norm = (lib @ Xi_trig)[0]
    return s + delta_norm * delta_std


def run_trajectory(s0, n_steps, tau_func):
    s_real = s0.copy()
    s_poly = s0.copy()
    s_trig = s0.copy()

    traj_real = [s_real[0]]
    traj_poly = [s_poly[0]]
    traj_trig = [s_trig[0]]

    for i in range(n_steps):
        tau = tau_func(i, s_real)
        s_real = real_step(s_real, tau)
        s_poly = sindy_poly_step(s_poly, tau)
        s_trig = sindy_trig_step(s_trig, tau)

        if abs(s_real[0]) > math.pi / 3:
            traj_real.extend([float('nan')] * (n_steps - i))
            traj_poly.extend([float('nan')] * (n_steps - i))
            traj_trig.extend([float('nan')] * (n_steps - i))
            break

        traj_real.append(s_real[0])
        traj_poly.append(s_poly[0])
        traj_trig.append(s_trig[0])

    return np.array(traj_real), np.array(traj_poly), np.array(traj_trig)


np.random.seed(42)
n_segments = 5
n_steps = 500
eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]

def make_tau_func(seg_idx):
    np.random.seed(42 + seg_idx)
    disturbances = np.random.uniform(-0.5, 0.5, n_steps)
    targets = np.clip(np.random.normal(0, 0.15), -math.pi/12, math.pi/12)
    def tau_func(step_i, s):
        x_lqr = np.array([s[0] - targets, s[2], s[1], s[3]])
        return float(-K_lqr @ x_lqr) + disturbances[step_i]
    return tau_func

all_trajs = []
for seg_i in range(n_segments):
    phi_init = np.random.uniform(-0.25, 0.25)
    s0 = np.array([phi_init, 0.0, 0.0, 0.0])
    tau_func = make_tau_func(seg_i)
    traj = run_trajectory(s0, n_steps, tau_func)
    all_trajs.append(traj)

# 计算RMSE
print(f"\n  ┌──────┬──────────┬──────────┬──────────┐")
print(f"  │ 步数 │   线性   │ SINDy多项式 │ SINDy+三角 │")
print(f"  ├──────┼──────────┼──────────┼──────────┤")

A_lin, B_lin = ab_matrix(M, C1, K0, K2, v0, g)
B_s_lin = B_lin[:, 1:2]

for step in eval_steps:
    sq_poly = []
    sq_trig = []
    for traj_real, traj_poly, traj_trig in all_trajs:
        if step >= len(traj_real):
            continue
        if math.isnan(traj_real[step]):
            continue
        sq_poly.append((traj_poly[step] - traj_real[step]) ** 2)
        sq_trig.append((traj_trig[step] - traj_real[step]) ** 2)

    rmse_poly = math.sqrt(np.mean(sq_poly)) if sq_poly else float('nan')
    rmse_trig = math.sqrt(np.mean(sq_trig)) if sq_trig else float('nan')

    def fmt(v):
        if math.isnan(v): return "  N/A "
        if v >= 1e6: return f"{v:.1e}"
        if v >= 1: return f"{v:.3f}"
        return f"{v:.4f}"

    print(f"  │ {step:4d} │ {fmt(rmse_poly):>8} │ {fmt(rmse_trig):>8} │")

print(f"  └──────┴──────────┴──────────┘")

print(f"\n结论:")
print(f"  多项式库RMSE at 10步: {rmse_poly:.4f} rad")
print(f"  三角函数库RMSE at 10步: {rmse_trig:.4f} rad")
if rmse_trig < rmse_poly:
    print(f"  三角函数库更好: {rmse_poly/rmse_trig:.1f}x")
else:
    print(f"  多项式库更好: {rmse_trig/rmse_poly:.1f}x")
