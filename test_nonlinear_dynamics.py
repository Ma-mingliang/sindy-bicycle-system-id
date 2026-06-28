"""对比线性化 vs 非线性Meijaard动力学的长时间多步精度。

测试场景：
1. 不同角度下单步误差
2. 10/20/50/100步rollout轨迹对比
3. 带LQR控制的实际场景rollout
4. 带随机扰动的Monte Carlo测试
"""

import numpy as np
import math
from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix
from scipy.linalg import solve_continuous_are


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

# LQR
A_lqr, B_lqr = ab_matrix(M, C1, K0, K2, v0, g)
reorder = [0, 2, 1, 3]
A_lqr_r = A_lqr[np.ix_(reorder, reorder)]
B_lqr_r = B_lqr[reorder, 1:2]
Q = np.diag([1000.0, 100.0, 10.0, 1.0])
R_mat = np.array([[0.2]])
P = solve_continuous_are(A_lqr_r, B_lqr_r, Q, R_mat)
K_lqr = (np.linalg.inv(R_mat) @ B_lqr_r.T @ P).flatten()


# ============================================================
# 动力学函数
# ============================================================
A_lin, B_lin = ab_matrix(M, C1, K0, K2, v0, g)
B_s_lin = B_lin[:, 1:2]


def linear_step(s, tau, dt_sub):
    x = s.reshape(4, 1)
    xd = A_lin @ x + B_s_lin * tau
    return s + dt_sub * xd.flatten()


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


def multi_substep(s, tau, dt_total, step_fn, n_sub=5):
    """多子步积分（模拟GraduatedEnv的sub_steps=5）。"""
    dt_sub = dt_total / n_sub
    s_cur = s.copy()
    for _ in range(n_sub):
        s_cur = step_fn(s_cur, tau, dt_sub)
    return s_cur


# ============================================================
# 测试1: 单步误差（不同角度）
# ============================================================
print("=" * 70)
print("测试1: 单步预测对比（不同倾斜角，tau=5.0 Nm）")
print("=" * 70)

tau_test = 5.0
angles_deg = [1, 2, 5, 10, 15, 17, 20, 30, 45, 60]
print(f"\n{'角度':>6} | {'线性化phi_next':>14} | {'非线性phi_next':>14} | {'差异(rad)':>10} | {'差异(°)':>8} | {'相对误差':>8}")
print("-" * 75)

for deg in angles_deg:
    phi = math.radians(deg)
    s = np.array([phi, 0.0, 0.0, 0.0])
    s_lin = multi_substep(s, tau_test, dt, linear_step, 5)
    s_nonlin = multi_substep(s, tau_test, dt, nonlinear_step_rk4, 5)
    diff = s_nonlin[0] - s_lin[0]
    rel_err = abs(diff) / max(abs(s_nonlin[0]), 1e-10) * 100
    print(f"{deg:5d}° | {s_lin[0]:14.6f} | {s_nonlin[0]:14.6f} | {diff:+10.6f} | {math.degrees(diff):+8.3f}° | {rel_err:7.2f}%")


# ============================================================
# 测试2: 长时间rollout（无控制，纯开环）
# ============================================================
print(f"\n{'='*70}")
print("测试2: 长时间开环rollout（phi=17°, tau=5.0 Nm, 无LQR）")
print("=" * 70)

for n_steps in [10, 20, 50, 100]:
    s0 = np.array([math.radians(17), 0.0, 0.0, 0.0])
    s_lin = s0.copy()
    s_nonlin = s0.copy()
    traj_lin = [math.degrees(s_lin[0])]
    traj_nonlin = [math.degrees(s_nonlin[0])]

    for _ in range(n_steps):
        s_lin = multi_substep(s_lin, tau_test, dt, linear_step, 5)
        s_nonlin = multi_substep(s_nonlin, tau_test, dt, nonlinear_step_rk4, 5)
        traj_lin.append(math.degrees(s_lin[0]))
        traj_nonlin.append(math.degrees(s_nonlin[0]))

    final_diff = traj_nonlin[-1] - traj_lin[-1]
    max_diff = max(abs(traj_nonlin[i] - traj_lin[i]) for i in range(len(traj_lin)))

    print(f"\n  {n_steps}步:")
    print(f"    线性化最终: {traj_lin[-1]:+.2f}°")
    print(f"    非线性最终: {traj_nonlin[-1]:+.2f}°")
    print(f"    最终差异:   {final_diff:+.2f}°")
    print(f"    最大差异:   {max_diff:.2f}°")

    # 打印关键时间点
    key_steps = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps]
    print(f"    {'步':>5} | {'线性化(°)':>10} | {'非线性(°)':>10} | {'差异(°)':>10}")
    for k in key_steps:
        if k < len(traj_lin):
            d = traj_nonlin[k] - traj_lin[k]
            print(f"    {k:5d} | {traj_lin[k]:+10.2f} | {traj_nonlin[k]:+10.2f} | {d:+10.2f}")


# ============================================================
# 测试3: 带LQR控制的长时间rollout（最接近实际场景）
# ============================================================
print(f"\n{'='*70}")
print("测试3: 带LQR控制的rollout（phi_init=17°, target=0°）")
print("=" * 70)

for n_steps in [10, 20, 50, 100]:
    s0 = np.array([math.radians(17), 0.0, 0.0, 0.0])
    target = 0.0

    s_lin = s0.copy()
    s_nonlin = s0.copy()
    traj_lin = [math.degrees(s_lin[0])]
    traj_nonlin = [math.degrees(s_nonlin[0])]

    for step_i in range(n_steps):
        # LQR控制（与GraduatedEnv一致）
        x_lqr_lin = np.array([s_lin[0] - target, s_lin[2], s_lin[1], s_lin[3]])
        u_lqr_lin = float(-K_lqr @ x_lqr_lin)

        x_lqr_nonlin = np.array([s_nonlin[0] - target, s_nonlin[2], s_nonlin[1], s_nonlin[3]])
        u_lqr_nonlin = float(-K_lqr @ x_lqr_nonlin)

        s_lin = multi_substep(s_lin, u_lqr_lin, dt, linear_step, 5)
        s_nonlin = multi_substep(s_nonlin, u_lqr_nonlin, dt, nonlinear_step_rk4, 5)

        traj_lin.append(math.degrees(s_lin[0]))
        traj_nonlin.append(math.degrees(s_nonlin[0]))

    final_diff = traj_nonlin[-1] - traj_lin[-1]
    max_diff = max(abs(traj_nonlin[i] - traj_lin[i]) for i in range(len(traj_lin)))

    print(f"\n  {n_steps}步:")
    print(f"    线性化最终: {traj_lin[-1]:+.2f}°")
    print(f"    非线性最终: {traj_nonlin[-1]:+.2f}°")
    print(f"    最终差异:   {final_diff:+.2f}°")
    print(f"    最大差异:   {max_diff:.2f}°")

    key_steps = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps]
    print(f"    {'步':>5} | {'线性化(°)':>10} | {'非线性(°)':>10} | {'差异(°)':>10}")
    for k in key_steps:
        if k < len(traj_lin):
            d = traj_nonlin[k] - traj_lin[k]
            print(f"    {k:5d} | {traj_lin[k]:+10.2f} | {traj_nonlin[k]:+10.2f} | {d:+10.2f}")


# ============================================================
# 测试4: 带LQR+随机残差的长时间rollout（模拟RL训练）
# ============================================================
print(f"\n{'='*70}")
print("测试4: 带LQR+随机残差的rollout（模拟RL虚拟rollout场景）")
print("=" * 70)

np.random.seed(42)
n_episodes = 5
n_steps = 50

for ep in range(n_episodes):
    phi_init = np.random.uniform(-0.3, 0.3)  # 随机初始倾斜角
    s0 = np.array([phi_init, 0.0, 0.0, 0.0])
    target = np.clip(np.random.normal(0, 0.1), -math.pi/12, math.pi/12)

    s_lin = s0.copy()
    s_nonlin = s0.copy()

    for step_i in range(n_steps):
        # LQR + 随机残差
        x_lqr_lin = np.array([s_lin[0] - target, s_lin[2], s_lin[1], s_lin[3]])
        u_lqr_lin = float(-K_lqr @ x_lqr_lin)
        residual = np.random.uniform(-0.1, 0.1)  # 随机残差
        tau_lin = u_lqr_lin + residual

        x_lqr_nonlin = np.array([s_nonlin[0] - target, s_nonlin[2], s_nonlin[1], s_nonlin[3]])
        u_lqr_nonlin = float(-K_lqr @ x_lqr_nonlin)
        tau_nonlin = u_lqr_nonlin + residual

        s_lin = multi_substep(s_lin, tau_lin, dt, linear_step, 5)
        s_nonlin = multi_substep(s_nonlin, tau_nonlin, dt, nonlinear_step_rk4, 5)

        # 如果倾倒就停止
        if abs(s_lin[0]) > math.pi/3 or abs(s_nonlin[0]) > math.pi/3:
            break

    diff_final = math.degrees(s_nonlin[0] - s_lin[0])
    print(f"  Ep{ep+1}: phi_init={math.degrees(phi_init):+.1f}°, target={math.degrees(target):+.1f}°, "
          f"steps={step_i+1}, "
          f"lin={math.degrees(s_lin[0]):+.2f}°, nonlin={math.degrees(s_nonlin[0]):+.2f}°, "
          f"diff={diff_final:+.2f}°")


# ============================================================
# 测试5: sin(phi)/phi 修正效果量化
# ============================================================
print(f"\n{'='*70}")
print("测试5: sin(phi)/phi 修正项效果")
print("=" * 70)

print(f"\n{'角度':>6} | {'sin(phi)/phi':>12} | {'误差(1-近似)':>12} | {'说明':>10}")
print("-" * 50)
for deg in [1, 5, 10, 15, 17, 20, 30, 45, 60]:
    phi = math.radians(deg)
    ratio = math.sin(phi) / phi if abs(phi) > 1e-8 else 1.0
    err = abs(1.0 - ratio) * 100
    note = "可忽略" if err < 1 else ("轻微" if err < 5 else ("显著" if err < 15 else "严重"))
    print(f"{deg:5d}° | {ratio:12.6f} | {err:11.2f}% | {note}")


# ============================================================
# 测试6: 两种模型的稳定性对比（长时间无控制）
# ============================================================
print(f"\n{'='*70}")
print("测试6: 稳定性对比（无控制，观察自然衰减/发散）")
print("=" * 70)

for deg_init in [5, 10, 17, 30]:
    s0 = np.array([math.radians(deg_init), 0.0, 0.0, 0.0])
    s_lin = s0.copy()
    s_nonlin = s0.copy()

    print(f"\n  初始角度={deg_init}°, tau=0（无控制）:")
    print(f"    {'步':>5} | {'线性化(°)':>10} | {'非线性(°)':>10}")
    print(f"    " + "-" * 35)

    for step_i in range(100):
        s_lin = multi_substep(s_lin, 0.0, dt, linear_step, 5)
        s_nonlin = multi_substep(s_nonlin, 0.0, dt, nonlinear_step_rk4, 5)

        if step_i in [0, 9, 19, 49, 99]:
            print(f"    {step_i+1:5d} | {math.degrees(s_lin[0]):+10.2f} | {math.degrees(s_nonlin[0]):+10.2f}")


print(f"\n{'='*70}")
print("总结")
print("=" * 70)
print("""
关键发现:
1. 单步误差在17°时很小（<0.1°），但多步累积显著
2. 带LQR控制时，两种模型的轨迹差异比开环更小（LQR纠正偏差）
3. 非线性模型的sin(phi)/phi修正在17°时仅1.46%，单步影响小
4. 多步累积效应是主要差异来源（非线性恢复力更强）
5. 在RL虚拟rollout场景中，50步后差异可达数度
6. 建议: 虚拟rollout使用非线性动力学+RK4，提升大角度精度
""")
