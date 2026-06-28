"""训练非线性动力学基线的NN残差模型。

基线：Meijaard非线性动力学单步RK4（dt=1/30）
目标：5子步RK4（真实模型）
残差 = 真实 - 基线（在物理空间中，然后归一化）

训练数据：从开环仿真数据中采样(s, tau)对，计算残差。
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix


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


def baseline_step(s, tau):
    """基线：非线性动力学单步RK4（dt=1/30）。"""
    return nonlinear_step_rk4(s, tau, dt)


def real_step(s, tau):
    """真实：5子步RK4。"""
    dt_sub = dt / 5
    s_cur = s.copy()
    for _ in range(5):
        s_cur = nonlinear_step_rk4(s_cur, tau, dt_sub)
    return s_cur


# ============================================================
# 生成训练数据
# ============================================================
print("生成训练数据...")

np.random.seed(42)
n_samples = 50000

# 随机采样状态和动作
phis = np.random.uniform(-0.5, 0.5, n_samples)        # ±28.6°
deltas = np.random.uniform(-0.3, 0.3, n_samples)       # ±17.2°
theta_dots = np.random.uniform(-2.0, 2.0, n_samples)   # 角速度
delta_dots = np.random.uniform(-1.0, 1.0, n_samples)   # 转向角速度
taus = np.random.uniform(-50, 50, n_samples)            # 力矩

states = np.column_stack([phis, deltas, theta_dots, delta_dots])

# 计算残差
residuals = []
for i in range(n_samples):
    s = states[i]
    tau = taus[i]
    s_baseline = baseline_step(s, tau)
    s_real = real_step(s, tau)
    residuals.append(s_real - s_baseline)

residuals = np.array(residuals)

# 归一化
state_std = np.std(states, axis=0)
action_std = np.std(taus)
residual_std = np.std(residuals, axis=0)

print(f"  样本数: {n_samples}")
print(f"  state_std: {state_std}")
print(f"  action_std: {action_std}")
print(f"  residual_std: {residual_std}")
print(f"  残差均值: {np.mean(residuals, axis=0)}")
print(f"  残差范围: {np.min(residuals, axis=0)} ~ {np.max(residuals, axis=0)}")

# 归一化输入和输出
states_norm = states / state_std
actions_norm = taus / action_std
residuals_norm = residuals / residual_std

# 划分训练/验证
n_train = int(0.9 * n_samples)
train_states = torch.FloatTensor(states_norm[:n_train])
train_actions = torch.FloatTensor(actions_norm[:n_train]).unsqueeze(1)
train_residuals = torch.FloatTensor(residuals_norm[:n_train])

val_states = torch.FloatTensor(states_norm[n_train:])
val_actions = torch.FloatTensor(actions_norm[n_train:]).unsqueeze(1)
val_residuals = torch.FloatTensor(residuals_norm[n_train:])

train_dataset = TensorDataset(train_states, train_actions, train_residuals)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


# ============================================================
# NN残差网络
# ============================================================
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResidualNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print(f"\n训练NN残差模型...")
print(f"  设备: {device}")
print(f"  参数量: {sum(p.numel() for p in model.parameters())}")

# 训练
n_epochs = 200
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    n_batches = 0
    for s_batch, a_batch, r_batch in train_loader:
        s_batch = s_batch.to(device)
        a_batch = a_batch.to(device)
        r_batch = r_batch.to(device)

        pred = model(s_batch, a_batch)
        loss = criterion(pred, r_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    train_loss = epoch_loss / n_batches
    train_losses.append(train_loss)

    # 验证
    model.eval()
    with torch.no_grad():
        val_pred = model(val_states.to(device), val_actions.to(device))
        val_loss = criterion(val_pred, val_residuals.to(device)).item()
    val_losses.append(val_loss)

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

# 保存模型
save_path = 'D:/系统辨识作业/sindy_bicycle/nonlinear_residual_nn.pt'
torch.save(model.state_dict(), save_path)
print(f"\n模型已保存: {save_path}")

# 保存归一化参数
np.savez('D:/系统辨识作业/sindy_bicycle/nonlinear_residual_params.npz',
         state_std=state_std, action_std=action_std, residual_std=residual_std)
print(f"归一化参数已保存: nonlinear_residual_params.npz")


# ============================================================
# 验证：单步预测精度
# ============================================================
print(f"\n{'='*80}")
print("验证：单步预测精度")
print("=" * 80)

model.eval()
test_angles = [5, 10, 15, 17, 20, 30]
test_tau = 10.0

print(f"\n  tau = {test_tau} Nm")
print(f"  {'角度':>6} | {'真实(5子步)':>12} | {'基线(单步RK4)':>14} | {'非线+NN':>12} | {'基线误差':>10} | {'非线+NN误差':>12}")
print("  " + "-" * 75)

for deg in test_angles:
    phi = math.radians(deg)
    s = np.array([phi, 0.0, 0.0, 0.0])

    s_real = real_step(s, test_tau)
    s_baseline = baseline_step(s, test_tau)

    # NN预测
    s_norm = s / state_std
    a_norm = test_tau / action_std
    with torch.no_grad():
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(device)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(device)
        delta_nn_norm = model(s_t, a_t).cpu().numpy()[0]
    delta_nn = delta_nn_norm * residual_std
    s_nonlin_nn = s_baseline + delta_nn

    err_baseline = math.degrees(s_baseline[0] - s_real[0])
    err_nonlin_nn = math.degrees(s_nonlin_nn[0] - s_real[0])

    print(f"  {deg:5d}° | {math.degrees(s_real[0]):12.4f}° | {math.degrees(s_baseline[0]):14.4f}° | "
          f"{math.degrees(s_nonlin_nn[0]):12.4f}° | {err_baseline:+10.4f}° | {err_nonlin_nn:+12.4f}°")


print(f"\n训练完成。")
