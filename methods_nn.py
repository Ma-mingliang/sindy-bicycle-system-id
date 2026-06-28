"""NN方法4-6：SINDy+NN残差、端到端NN、Neural ODE。

统一接口：
    train(states, actions, deltas, state_std, action_std, delta_std)
    predict(s, tau) -> s_next
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from methods_sindy import SINDyPoly, SINDyTrig, SINDyTrigExp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# NN残差网络（方法4, 10, 11, 12共用）
# ============================================================
class ResidualNet(nn.Module):
    def __init__(self, state_dim: int = 4, action_dim: int = 1, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, s, a=None):
        if a is not None:
            return self.net(torch.cat([s, a], dim=-1))
        return self.net(s)  # 单输入模式（已拼接）


def _train_nn(model: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor,
              n_epochs: int = 200, lr: float = 1e-3, batch_size: int = 256) -> None:
    """通用NN训练循环。"""
    ds = TensorDataset(train_x, train_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = crit(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()


# ============================================================
# 方法4：SINDy最佳 + NN残差
# ============================================================
class SINDyBestNN:
    name = '4_SINDyBestNN'

    def __init__(self, n_epochs: int = 200):
        self.n_epochs = n_epochs
        self._sindy = None
        self._nn = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        # 选最佳SINDy
        sindy_methods = [SINDyPoly(), SINDyTrig(), SINDyTrigExp()]
        best_rmse, best_sindy = 1e9, None
        for sm in sindy_methods:
            rmse = sm.train(states, actions, deltas, state_std, action_std, delta_std)
            total = np.mean(rmse)
            if total < best_rmse:
                best_rmse, best_sindy = total, sm
        self._sindy = best_sindy
        print(f"    最佳SINDy: {best_sindy.name} (avg RMSE={best_rmse:.6f})")

        # 计算残差 = delta_real - delta_sindy，在归一化空间
        states_norm = states / state_std
        actions_norm = actions / action_std
        deltas_norm = deltas / delta_std
        sindy_pred = np.empty_like(deltas_norm)
        for i in range(len(states)):
            s_next = best_sindy.predict(states[i], actions[i])
            sindy_pred[i] = (s_next - states[i]) / delta_std
        residuals = deltas_norm - sindy_pred

        # 训练NN
        train_x = torch.FloatTensor(np.column_stack([states_norm, actions_norm.reshape(-1, 1)])).to(DEVICE)
        train_y = torch.FloatTensor(residuals).to(DEVICE)
        self._nn = ResidualNet().to(DEVICE)
        _train_nn(self._nn, train_x, train_y, self.n_epochs)

    def predict(self, s, tau):
        s_next_sindy = self._sindy.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        with torch.no_grad():
            s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
            a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
            delta_nn = self._nn(s_t, a_t).cpu().numpy()[0] * self._delta_std
        return s_next_sindy + delta_nn


# ============================================================
# 方法5：端到端NN
# ============================================================
class NNE2E:
    name = '5_NNE2E'

    def __init__(self, n_epochs: int = 200, hidden: int = 128):
        self.n_epochs = n_epochs
        self._hidden = hidden
        self._nn = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        states_norm = states / state_std
        actions_norm = actions / action_std
        deltas_norm = deltas / delta_std
        train_x = torch.FloatTensor(np.column_stack([states_norm, actions_norm.reshape(-1, 1)])).to(DEVICE)
        train_y = torch.FloatTensor(deltas_norm).to(DEVICE)
        self._nn = ResidualNet(hidden=self._hidden).to(DEVICE)
        _train_nn(self._nn, train_x, train_y, self.n_epochs)

    def predict(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        with torch.no_grad():
            s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
            a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
            delta_norm = self._nn(s_t, a_t).cpu().numpy()[0]
        return s + delta_norm * self._delta_std


# ============================================================
# 方法6：Neural ODE
# ============================================================
class NeuralODEMethod:
    name = '6_NeuralODE'

    def __init__(self, n_epochs: int = 100, hidden: int = 64):
        self.n_epochs = n_epochs
        self._hidden = hidden
        self._nn = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        # ds/dt ≈ delta_s / dt，用NN拟合f(s,a)
        from methods_common import dt
        states_norm = states / state_std
        actions_norm = actions / action_std
        derivs = deltas / (delta_std * dt)  # 归一化空间的导数

        train_x = torch.FloatTensor(np.column_stack([states_norm, actions_norm.reshape(-1, 1)])).to(DEVICE)
        train_y = torch.FloatTensor(derivs).to(DEVICE)
        self._nn = nn.Sequential(
            nn.Linear(5, self._hidden), nn.Tanh(),
            nn.Linear(self._hidden, self._hidden), nn.Tanh(),
            nn.Linear(self._hidden, 4),
        ).to(DEVICE)
        _train_nn(self._nn, train_x, train_y, self.n_epochs)

    def predict(self, s, tau):
        from methods_common import dt
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        with torch.no_grad():
            inp = torch.FloatTensor(np.concatenate([s_norm, [a_norm]])).unsqueeze(0).to(DEVICE)
            dsdt_norm = self._nn(inp).cpu().numpy()[0]
        # Euler积分一步
        ds_norm = dsdt_norm * dt
        return s + ds_norm * self._delta_std
