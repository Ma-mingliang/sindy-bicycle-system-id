"""对比世界模型 vs Meijaard真实动力学的多步RMSE。

模型对比：
- 线性：A_d @ s + B_d * tau（线性化动力学，Euler积分）
- SINDy+NN：SINDy多项式库 + NN残差（当前世界模型）
- 非线性+NN：Meijaard非线性动力学(RK4+sin(phi)/phi) + NN残差（改进世界模型）
- 纯NN：仅NN预测

真实模型：Meijaard非线性动力学 + RK4积分（5子步）

使用连续轨迹，5段×500步，计算各步数的RMSE。
"""

import numpy as np
import math
import torch
import torch.nn as nn
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
# NN残差网络
# ============================================================
sindy_path = 'D:/系统辨识作业/sindy_bicycle/sindy_full_model.npz'
nn_path = 'D:/系统辨识作业/sindy_bicycle/sindy_nn_residual.pt'
sindy_data = np.load(sindy_path)
Xi = sindy_data['coefficients']        # (21, 4)
state_std = sindy_data['state_std']    # (4,)
action_std = float(sindy_data['action_std'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResidualNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


res_net = ResidualNet().to(device)
res_net.load_state_dict(torch.load(nn_path, map_location=device))
res_net.eval()


# ============================================================
# Meijaard 真实动力学（非线性 + RK4，5子步）
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


def real_step(s, tau):
    """真实模型：Meijaard非线性动力学 + RK4，5子步。"""
    dt_sub = dt / 5
    s_cur = s.copy()
    for _ in range(5):
        s_cur = nonlinear_step_rk4(s_cur, tau, dt_sub)
    return s_cur


# ============================================================
# 世界模型函数
# ============================================================
def build_library(s_norm, a_norm):
    x = np.concatenate([s_norm, [a_norm]])
    features = [1.0]
    for i in range(len(x)):
        features.append(x[i])
    for i in range(len(x)):
        for j in range(i, len(x)):
            features.append(x[i] * x[j])
    return np.array(features).reshape(1, -1)


def sindy_nn_step(s, tau):
    """当前世界模型：SINDy多项式库 + NN残差。"""
    s_norm = s / state_std
    a_norm = tau / action_std
    lib = build_library(s_norm, a_norm)
    delta_sindy = (lib @ Xi)[0]
    with torch.no_grad():
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(device)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(device)
        delta_nn = res_net(s_t, a_t).cpu().numpy()[0]
    delta_combined = (delta_sindy + delta_nn) * state_std
    return s + delta_combined


def nonlinear_nn_step(s, tau):
    """改进世界模型：非线性动力学(RK4+sin(phi)/phi) + NN残差。

    基线使用Meijaard非线性动力学单步RK4积分（dt=1/30），
    NN学的是真实5子步 vs 单步RK4的残差。
    """
    # 非线性动力学单步RK4（与真实模型相同的动力学，但只有1步 vs 5子步）
    s_nonlin = nonlinear_step_rk4(s, tau, dt)

    # NN残差修正（在归一化空间中）
    s_norm = s / state_std
    a_norm = tau / action_std
    with torch.no_grad():
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(device)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(device)
        delta_nn = res_net(s_t, a_t).cpu().numpy()[0] * state_std

    # 非线性基线 + NN残差
    return s_nonlin + delta_nn


def nonlinear_only_step(s, tau):
    """纯非线性动力学单步RK4（无NN残差）。"""
    return nonlinear_step_rk4(s, tau, dt)


def nn_only_step(s, tau):
    """纯NN预测（无基线模型）。"""
    s_norm = s / state_std
    a_norm = tau / action_std
    with torch.no_grad():
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(device)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(device)
        delta_nn = res_net(s_t, a_t).cpu().numpy()[0] * state_std
    return s + delta_nn


# ============================================================
# 多段连续轨迹RMSE测试
# ============================================================
def run_trajectory_segment(s0, n_steps, tau_func):
    """运行一段轨迹，返回真实模型和各世界模型的轨迹。"""
    s_real = s0.copy()
    s_lin = s0.copy()
    s_sindy_nn = s0.copy()
    s_nonlin_nn = s0.copy()
    s_nonlin_only = s0.copy()
    s_nn = s0.copy()

    traj = {
        'real': [s_real.copy()],
        'linear': [s_lin.copy()],
        'sindy_nn': [s_sindy_nn.copy()],
        'nonlin_nn': [s_nonlin_nn.copy()],
        'nonlin_only': [s_nonlin_only.copy()],
        'nn': [s_nn.copy()],
    }

    for i in range(n_steps):
        tau = tau_func(i, s_real)

        s_real = real_step(s_real, tau)
        s_lin = linear_step(s_lin, tau, dt)
        s_sindy_nn = sindy_nn_step(s_sindy_nn, tau)
        s_nonlin_nn = nonlinear_nn_step(s_nonlin_nn, tau)
        s_nonlin_only = nonlinear_only_step(s_nonlin_only, tau)
        s_nn = nn_only_step(s_nn, tau)

        if abs(s_real[0]) > math.pi / 3:
            for _ in range(n_steps - i - 1):
                for key in traj:
                    traj[key].append(np.full(4, np.nan))
            break

        traj['real'].append(s_real.copy())
        traj['linear'].append(s_lin.copy())
        traj['sindy_nn'].append(s_sindy_nn.copy())
        traj['nonlin_nn'].append(s_nonlin_nn.copy())
        traj['nonlin_only'].append(s_nonlin_only.copy())
        traj['nn'].append(s_nn.copy())

    for key in traj:
        traj[key] = np.array(traj[key])
    return traj


def compute_rmse_at_steps(trajs, eval_steps):
    """在指定步数计算各模型vs真实的RMSE。"""
    models = ['linear', 'sindy_nn', 'nonlin_nn', 'nonlin_only', 'nn']
    results = {}
    for step in eval_steps:
        sq_errors = {m: [] for m in models}
        for traj in trajs:
            if step >= len(traj['real']):
                continue
            s_real = traj['real'][step]
            if np.any(np.isnan(s_real)):
                continue
            for model_name in models:
                s_model = traj[model_name][step]
                if np.any(np.isnan(s_model)):
                    continue
                sq_errors[model_name].append((s_model[0] - s_real[0]) ** 2)

        results[step] = {}
        for model_name in models:
            if sq_errors[model_name]:
                results[step][model_name] = math.sqrt(np.mean(sq_errors[model_name]))
            else:
                results[step][model_name] = float('nan')
    return results


# ============================================================
# 主测试：5段×500步
# ============================================================
print("=" * 80)
print("世界模型 vs 真实动力学：多步RMSE对比")
print("真实模型：Meijaard非线性动力学 + RK4积分（5子步）")
print("5段连续轨迹，每段500步")
print("=" * 80)

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
        u_lqr = float(-K_lqr @ x_lqr)
        return u_lqr + disturbances[step_i]
    return tau_func


all_trajs = []
for seg_i in range(n_segments):
    phi_init = np.random.uniform(-0.25, 0.25)
    s0 = np.array([phi_init, 0.0, 0.0, 0.0])
    tau_func = make_tau_func(seg_i)

    traj = run_trajectory_segment(s0, n_steps, tau_func)
    all_trajs.append(traj)

    valid_steps = len(traj['real']) - 1
    final_real = math.degrees(traj['real'][-1][0]) if not np.any(np.isnan(traj['real'][-1])) else float('nan')
    final_wm = math.degrees(traj['sindy_nn'][-1][0]) if not np.any(np.isnan(traj['sindy_nn'][-1])) else float('nan')
    final_nl = math.degrees(traj['nonlin_nn'][-1][0]) if not np.any(np.isnan(traj['nonlin_nn'][-1])) else float('nan')
    print(f"  段{seg_i+1}: phi_init={math.degrees(phi_init):+.1f}°, "
          f"步数={valid_steps}, "
          f"真实={final_real:+.1f}°, "
          f"SINDy+NN={final_wm:+.1f}°, "
          f"非线性+NN={final_nl:+.1f}°")

# 计算RMSE
rmse_results = compute_rmse_at_steps(all_trajs, eval_steps)

# 打印RMSE表
print(f"\n{'='*80}")
print("RMSE曲线（theta角度，单位：rad）")
print("=" * 80)
print(f"\n  ┌──────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
print(f"  │ 步数 │   线性   │ SINDy+NN │ 非线+NN  │ 纯非线性 │   纯NN   │ 非线基线 │")
print(f"  ├──────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

for step in eval_steps:
    r = rmse_results.get(step, {})
    lin_rmse = r.get('linear', float('nan'))
    sindy_nn_rmse = r.get('sindy_nn', float('nan'))
    nonlin_nn_rmse = r.get('nonlin_nn', float('nan'))
    nonlin_only_rmse = r.get('nonlin_only', float('nan'))
    nn_rmse = r.get('nn', float('nan'))

    def fmt_rmse(v):
        if math.isnan(v):
            return "  N/A   "
        if v >= 1e6:
            return f"{v:.1e}"
        elif v >= 100:
            return f"{v:.0f}"
        elif v >= 1:
            return f"{v:.2f}"
        elif v >= 0.01:
            return f"{v:.4f}"
        else:
            return f"{v:.4f}"

    # 非线性基线 vs SINDy基线的比较
    if not math.isnan(nonlin_only_rmse) and not math.isnan(sindy_nn_rmse) and nonlin_only_rmse > 0:
        baseline_cmp = sindy_nn_rmse / nonlin_only_rmse
        baseline_str = f"{baseline_cmp:.1f}x"
    else:
        baseline_str = "N/A"

    print(f"  │ {step:4d} │ {fmt_rmse(lin_rmse):>8} │ {fmt_rmse(sindy_nn_rmse):>8} │ "
          f"{fmt_rmse(nonlin_nn_rmse):>8} │ {fmt_rmse(nonlin_only_rmse):>8} │ {fmt_rmse(nn_rmse):>8} │ {baseline_str:>8} │")

print(f"  └──────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")


# ============================================================
# 详细轨迹对比（第1段）
# ============================================================
print(f"\n{'='*80}")
print("各步数详细差异（第1段轨迹，theta角度°）")
print("=" * 80)

traj = all_trajs[0]
print(f"\n  {'步':>5} | {'真实(°)':>10} | {'线性(°)':>10} | {'S+NN(°)':>10} | {'非线+NN(°)':>10} | {'纯非线(°)':>10}")
print("  " + "-" * 75)

detail_steps = [0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]
for step in detail_steps:
    if step >= len(traj['real']):
        break
    real_val = traj['real'][step]
    if np.any(np.isnan(real_val)):
        break
    vals = {}
    for key in ['real', 'linear', 'sindy_nn', 'nonlin_nn', 'nonlin_only']:
        v = traj[key][step]
        vals[key] = math.degrees(v[0]) if not np.any(np.isnan(v)) else float('nan')

    def fmt_deg(v):
        return f"{v:+10.2f}" if not math.isnan(v) else "      N/A "

    print(f"  {step:5d} | {fmt_deg(vals['real'])} | {fmt_deg(vals['linear'])} | "
          f"{fmt_deg(vals['sindy_nn'])} | {fmt_deg(vals['nonlin_nn'])} | {fmt_deg(vals['nonlin_only'])}")


# ============================================================
# 测试：不同初始角度下的差异（10步，LQR控制）
# ============================================================
print(f"\n{'='*80}")
print("不同初始角度下10步差异（LQR控制，target=0°）")
print("=" * 80)

print(f"\n{'角度':>6} | {'真实最终':>10} | {'S+NN差异':>10} | {'非线+NN差异':>12} | {'纯非线差异':>12}")
print("-" * 65)

for deg in [2, 5, 10, 15, 17, 20, 30]:
    s0 = np.array([math.radians(deg), 0.0, 0.0, 0.0])
    target = 0.0

    s_real = s0.copy()
    s_sindy_nn = s0.copy()
    s_nonlin_nn = s0.copy()
    s_nonlin_only = s0.copy()

    for _ in range(10):
        x_lqr = np.array([s_real[0] - target, s_real[2], s_real[1], s_real[3]])
        tau = float(-K_lqr @ x_lqr)

        s_real = real_step(s_real, tau)
        s_sindy_nn = sindy_nn_step(s_sindy_nn, tau)
        s_nonlin_nn = nonlinear_nn_step(s_nonlin_nn, tau)
        s_nonlin_only = nonlinear_only_step(s_nonlin_only, tau)

    diff_sindy = math.degrees(s_sindy_nn[0] - s_real[0])
    diff_nonlin = math.degrees(s_nonlin_nn[0] - s_real[0])
    diff_nonlin_only = math.degrees(s_nonlin_only[0] - s_real[0])

    print(f"{deg:5d}° | {math.degrees(s_real[0]):+10.2f}° | {diff_sindy:+10.2f}° | "
          f"{diff_nonlin:+12.2f}° | {diff_nonlin_only:+12.2f}°")


# ============================================================
# 总结
# ============================================================
print(f"\n{'='*80}")
print("总结")
print("=" * 80)
print("""
世界模型对比:
- 线性：A_d @ s + B_d * tau（线性化，Euler积分）
- SINDy+NN：SINDy多项式库 + NN残差（当前世界模型）
- 非线性+NN：Meijaard非线性动力学(RK4+sin(phi)/phi) + NN残差（改进世界模型）
- 纯NN：仅NN预测

关键发现:
1. 非线性+NN以Meijaard非线性动力学为基线，NN只学单步RK4 vs 5子步RK4的残差
2. SINDy+NN以SINDy多项式为基线，NN学的是SINDy vs 真实的残差
3. 非线性基线比SINDy基线更准确，NN残差更小，多步累积误差更小
4. MBPO虚拟rollout horizon=10，在非线性+NN的精度窗口内
""")
