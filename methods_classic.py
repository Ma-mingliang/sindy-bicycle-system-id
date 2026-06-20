"""经典方法7-9：高斯过程、PINN、参数辨识。

统一接口：
    train(states, actions, deltas, state_std, action_std, delta_std)
    predict(s, tau) -> s_next
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# 方法7：高斯过程（GP）
# ============================================================
class GPMethod:
    name = '7_GP'

    def __init__(self, max_samples: int = 5000):
        self.max_samples = max_samples
        self._gps = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        # 下采样
        n = len(states)
        if n > self.max_samples:
            idx = np.random.choice(n, self.max_samples, replace=False)
        else:
            idx = np.arange(n)
        X = np.column_stack([states[idx] / state_std, actions[idx].reshape(-1, 1) / action_std])
        Y = deltas[idx] / delta_std

        self._gps = []
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        for col in range(4):
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
            gp.fit(X, Y[:, col])
            self._gps.append(gp)
            print(f"    GP输出{col} 训练完成")

    def predict(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        delta_norm = np.array([gp.predict(x)[0] for gp in self._gps])
        return s + delta_norm * self._delta_std


# ============================================================
# 方法8：PINN（物理信息神经网络）
# ============================================================
class PINNMethod:
    name = '8_PINN'

    def __init__(self, n_epochs: int = 200, hidden: int = 128, phys_weight: float = 0.1):
        self.n_epochs = n_epochs
        self._hidden = hidden
        self._phys_weight = phys_weight
        self._nn = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        from methods_common import M, C1, K0, K2, g, invM, dt, v0
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        states_norm = states / state_std
        actions_norm = actions / action_std
        deltas_norm = deltas / delta_std

        train_x = torch.FloatTensor(np.column_stack([states_norm, actions_norm.reshape(-1, 1)])).to(DEVICE)
        train_y = torch.FloatTensor(deltas_norm).to(DEVICE)

        self._nn = nn.Sequential(
            nn.Linear(5, self._hidden), nn.SiLU(),
            nn.Linear(self._hidden, self._hidden), nn.SiLU(),
            nn.Linear(self._hidden, 4),
        ).to(DEVICE)

        ds = TensorDataset(train_x, train_y)
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        opt = optim.Adam(self._nn.parameters(), lr=1e-3)
        mse = nn.MSELoss()

        # 预计算物理常量（tensor）
        M_t = torch.FloatTensor(M).to(DEVICE)
        C1_t = torch.FloatTensor(C1).to(DEVICE)
        K0_t = torch.FloatTensor(K0).to(DEVICE)
        K2_t = torch.FloatTensor(K2).to(DEVICE)
        invM_t = torch.FloatTensor(invM).to(DEVICE)

        self._nn.train()
        for epoch in range(self.n_epochs):
            total_data = 0.0
            total_phys = 0.0
            for xb, yb in loader:
                pred = self._nn(xb)
                loss_data = mse(pred, yb)

                # 物理约束：预测的delta_s应满足 M·q_dd ≈ -C1·v·q_dot - (g·K0+v^2·K2)·q + F
                # 反推q_dd ≈ (delta_q_dot) / dt
                s_phys = xb[:, :4] * torch.FloatTensor(state_std).to(DEVICE)
                delta_phys = pred * torch.FloatTensor(delta_std).to(DEVICE)
                phi = s_phys[:, 0]
                delta = s_phys[:, 1]
                q = torch.stack([phi, delta], dim=1)  # (B, 2)
                q_dot = torch.stack([s_phys[:, 2], s_phys[:, 3]], dim=1)  # (B, 2)
                q_dd_pred = delta_phys[:, 2:4] / dt  # (B, 2)
                # 物理残差
                tau_phys = xb[:, 4:5] * action_std
                F = torch.zeros(q.shape[0], 2, device=DEVICE)
                F[:, 1] = tau_phys.squeeze()
                # rhs = -v*C1*q_dot - (g*K0+v^2*K2)*q + F  对batch做einsum
                stiff = g * K0_t + v0**2 * K2_t  # (2,2)
                rhs_phys = -v0 * torch.einsum('ij,bj->bi', C1_t, q_dot) \
                           - torch.einsum('ij,bj->bi', stiff, q) + F  # (B,2)
                q_dd_phys = torch.einsum('ij,bj->bi', invM_t, rhs_phys)  # (B,2)
                loss_phys = mse(q_dd_pred, q_dd_phys)

                loss = loss_data + self._phys_weight * loss_phys
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_data += loss_data.item()
                total_phys += loss_phys.item()

        self._nn.eval()

    def predict(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        with torch.no_grad():
            inp = torch.FloatTensor(np.concatenate([s_norm, [a_norm]])).unsqueeze(0).to(DEVICE)
            delta_norm = self._nn(inp).cpu().numpy()[0]
        return s + delta_norm * self._delta_std


# ============================================================
# 方法9：参数辨识（假设Meijaard结构）
# ============================================================
class ParamIDMethod:
    name = '9_ParamID'

    def __init__(self):
        self._M_id = None
        self._C1_id = None
        self._K0_id = None
        self._K2_id = None
        self._invM_id = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        from methods_common import dt, v0, g
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        # 用最小二乘拟合 M, C1, K0, K2
        # 动力学：M·q_dd = -C1·v·q_dot - (g·K0 + v^2·K2)·q + F
        # q_dd ≈ delta_q_dot / dt
        n = len(states)
        q_dd = deltas[:, 2:4] / dt  # (n, 2)
        q = states[:, :2]           # (n, 2)
        q_dot = states[:, 2:4]      # (n, 2)
        F = np.zeros((n, 2))
        F[:, 1] = actions

        # 直接拟合 delta_s ≈ f(s, tau) 的线性参数
        X = states  # (n, 4)
        A_lin = np.column_stack([X, actions])  # (n, 5)
        # 拟合 delta_phi_dot 和 delta_delta_dot
        coef_phi_dot, _, _, _ = np.linalg.lstsq(A_lin, deltas[:, 2], rcond=None)
        coef_delta_dot, _, _, _ = np.linalg.lstsq(A_lin, deltas[:, 3], rcond=None)
        # 拟合 delta_phi 和 delta_delta（位置部分）
        coef_phi, _, _, _ = np.linalg.lstsq(A_lin, deltas[:, 0], rcond=None)
        coef_delta, _, _, _ = np.linalg.lstsq(A_lin, deltas[:, 1], rcond=None)

        self._coef = np.array([coef_phi, coef_delta, coef_phi_dot, coef_delta_dot]).T  # (5, 4)

        # 验证
        pred = A_lin @ self._coef
        rmse = np.sqrt(np.mean((pred - deltas) ** 2, axis=0))
        return rmse

    def predict(self, s, tau):
        x = np.concatenate([s, [tau]])
        delta = self._coef.T @ x  # (4,)
        return s + delta

