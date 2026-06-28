"""混合方法10-12：基线模型 + NN残差。

统一接口：
    train(states, actions, deltas, state_std, action_std, delta_std)
    predict(s, tau) -> s_next
"""

import numpy as np
import torch
from methods_nn import ResidualNet, _train_nn, DEVICE


class _HybridBase:
    """混合方法通用逻辑：基线 + NN残差。"""
    name: str = 'HybridBase'
    _baseline_cls = None

    def __init__(self, n_epochs: int = 200, **baseline_kwargs):
        self.n_epochs = n_epochs
        self._baseline_kwargs = baseline_kwargs
        self._baseline = None
        self._nn = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        # 训练基线
        self._baseline = self._baseline_cls(**self._baseline_kwargs)
        self._baseline.train(states, actions, deltas, state_std, action_std, delta_std)
        print(f"    基线 {self._baseline.name} 训练完成")

        # 计算残差 = delta_real - delta_baseline（归一化空间）
        states_norm = states / state_std
        actions_norm = actions / action_std
        deltas_norm = deltas / delta_std
        baseline_pred = np.empty_like(deltas_norm)
        for i in range(len(states)):
            s_next = self._baseline.predict(states[i], actions[i])
            baseline_pred[i] = (s_next - states[i]) / delta_std
        residuals = deltas_norm - baseline_pred

        # 训练NN残差
        train_x = torch.FloatTensor(np.column_stack([states_norm, actions_norm.reshape(-1, 1)])).to(DEVICE)
        train_y = torch.FloatTensor(residuals).to(DEVICE)
        self._nn = ResidualNet().to(DEVICE)
        _train_nn(self._nn, train_x, train_y, self.n_epochs)

    def predict(self, s, tau):
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        with torch.no_grad():
            s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
            a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
            delta_nn = self._nn(s_t, a_t).cpu().numpy()[0] * self._delta_std
        return s_next_base + delta_nn


class NeuralODE_NN(_HybridBase):
    name = '10_NeuralODE_NN'

    def __init__(self, n_epochs: int = 200):
        from methods_nn import NeuralODEMethod
        super().__init__(n_epochs=n_epochs)
        self._baseline_cls = NeuralODEMethod


class GP_NN(_HybridBase):
    name = '11_GP_NN'

    def __init__(self, n_epochs: int = 200):
        from methods_classic import GPMethod
        super().__init__(n_epochs=n_epochs)
        self._baseline_cls = GPMethod


class ParamID_NN(_HybridBase):
    name = '12_ParamID_NN'

    def __init__(self, n_epochs: int = 200):
        from methods_classic import ParamIDMethod
        super().__init__(n_epochs=n_epochs)
        self._baseline_cls = ParamIDMethod
