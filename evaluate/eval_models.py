"""模型定义：6种世界模型的统一接口。"""

import numpy as np
import math
import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple

# 延迟导入以避免循环依赖
_methods_common = None
_methods_nn = None


def _get_methods_common():
    global _methods_common
    if _methods_common is None:
        import methods_common
        _methods_common = methods_common
    return _methods_common


def _get_methods_nn():
    global _methods_nn
    if _methods_nn is None:
        import methods_nn
        _methods_nn = methods_nn
    return _methods_nn


class BaseModel(ABC):
    """模型基类。"""

    @abstractmethod
    def predict(self, s: np.ndarray, tau: float) -> np.ndarray:
        """单步预测: 给定当前状态和力矩，返回下一状态。"""
        pass

    @abstractmethod
    def name(self) -> str:
        """模型名称。"""
        pass

    def predict_with_uncertainty(self, s: np.ndarray, tau: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """带不确定性的预测。默认无不确定性。"""
        return self.predict(s, tau), None


class RealDynamics(BaseModel):
    """真实非线性动力学 (参考基准)。"""

    def __init__(self):
        mc = _get_methods_common()
        self._real_step = mc.real_step

    def predict(self, s: np.ndarray, tau: float) -> np.ndarray:
        return self._real_step(s, tau)

    def name(self) -> str:
        return 'real_dynamics'


class LinearizedModel(BaseModel):
    """线性化物理模型: s_next = s + (A*s + B*tau)*dt。"""

    def __init__(self):
        mc = _get_methods_common()
        from meijaard_dynamics import ab_matrix
        M, C1, K0, K2 = mc.M, mc.C1, mc.K0, mc.K2
        self.A, self.B = ab_matrix(M, C1, K0, K2, mc.v0, mc.g)
        self.dt = mc.dt
        # 状态重排: [phi, delta, phi_dot, delta_dot] -> [phi, phi_dot, delta, delta_dot]
        self.reorder = [0, 2, 1, 3]
        self.inv_reorder = [0, 2, 1, 3]

    def predict(self, s: np.ndarray, tau: float) -> np.ndarray:
        # 重排到线性化坐标
        s_r = s[self.reorder]
        ds_r = self.A @ s_r + self.B[:, 1] * tau
        s_next_r = s_r + ds_r * self.dt
        return s_next_r[self.inv_reorder]

    def name(self) -> str:
        return 'linearized_model'


class GPStandard(BaseModel):
    """标准GP模型。"""

    def __init__(self, max_samples: int = 5000, n_restarts: int = 2):
        self.max_samples = max_samples
        self.n_restarts = n_restarts
        self._gps = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states: np.ndarray, actions: np.ndarray, deltas: np.ndarray,
              state_std: np.ndarray, action_std: float, delta_std: np.ndarray):
        """训练GP模型。"""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        n = len(states)
        if n > self.max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, self.max_samples, replace=False)
        else:
            idx = np.arange(n)

        X = np.column_stack([states[idx] / state_std, actions[idx].reshape(-1, 1) / action_std])
        Y = deltas[idx] / delta_std

        self._gps = []
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        for col in range(4):
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self.n_restarts,
                alpha=1e-6
            )
            gp.fit(X, Y[:, col])
            self._gps.append(gp)

    def predict(self, s: np.ndarray, tau: float) -> np.ndarray:
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        delta_norm = np.array([gp.predict(x)[0] for gp in self._gps])
        return s + delta_norm * self._delta_std

    def name(self) -> str:
        return 'gp_standard'


class GPB4Sparse(BaseModel):
    """GP-B4 稀疏GP (Nystroem近似)。"""

    def __init__(self, n_components: int = 500, max_samples: int = 5000):
        self.n_components = n_components
        self.max_samples = max_samples
        self._gps = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None
        self._transformers = None

    def train(self, states: np.ndarray, actions: np.ndarray, deltas: np.ndarray,
              state_std: np.ndarray, action_std: float, delta_std: np.ndarray):
        """训练稀疏GP模型。"""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        from sklearn.kernel_approximation import Nystroem

        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        # 下采样以避免内存问题
        n = len(states)
        if n > self.max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, self.max_samples, replace=False)
        else:
            idx = np.arange(n)

        X = np.column_stack([states[idx] / state_std, actions[idx].reshape(-1, 1) / action_std])
        Y = deltas[idx] / delta_std

        n_comp = min(self.n_components, len(X))
        self._transformers = []
        self._gps = []

        for col in range(4):
            transformer = Nystroem(kernel='rbf', n_components=n_comp, random_state=42)
            X_transformed = transformer.fit_transform(X)

            gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=1e-6)
            gp.fit(X_transformed, Y[:, col])

            self._transformers.append(transformer)
            self._gps.append(gp)

    def predict(self, s: np.ndarray, tau: float) -> np.ndarray:
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        delta_norm = np.array([
            gp.predict(t.transform(x))[0]
            for gp, t in zip(self._gps, self._transformers)
        ])
        return s + delta_norm * self._delta_std

    def name(self) -> str:
        return 'gp_b4_sparse'


class SINDy4D(BaseModel):
    """SINDy 4D 模型: 多项式回归。"""

    def __init__(self, model_file: str):
        self._model_file = model_file
        self._coefficients = None
        self._state_std = None
        self._delta_std = None

    def train(self, states: np.ndarray, actions: np.ndarray, deltas: np.ndarray,
              state_std: np.ndarray, action_std: float, delta_std: np.ndarray):
        """加载SINDy系数或从数据拟合。"""
        import os
        mc = _get_methods_common()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fpath = os.path.join(root, self._model_file)

        if os.path.exists(fpath):
            data = np.load(fpath)
            self._coefficients = data['coefficients']
        else:
            # 从数据拟合
            self._coefficients = self._fit_sindy(states, actions, deltas)

        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

    def _fit_sindy(self, states, actions, deltas):
        """用多项式库拟合SINDy。"""
        X = np.column_stack([states, actions.reshape(-1, 1)])
        Theta = self._build_library(X)
        coefficients = np.linalg.lstsq(Theta, deltas, rcond=None)[0]
        return coefficients

    def _build_library(self, X):
        """构建多项式库 (与原SINDy一致)。"""
        n = X.shape[0]
        lib = [np.ones(n)]
        for i in range(X.shape[1]):
            lib.append(X[:, i])
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                lib.append(X[:, i] * X[:, j])
        return np.column_stack(lib)

    def predict(self, s: np.ndarray, tau: float) -> np.ndarray:
        x = np.concatenate([s, [tau]])
        Theta = self._build_library(x.reshape(1, -1))
        delta = (Theta @ self._coefficients).flatten()
        return s + delta

    def name(self) -> str:
        return 'sindy_4d'


class GPEEnsemble(BaseModel):
    """GP + NN集成 + DAgger 模型。"""

    def __init__(self, n_models: int = 5, dagger_rounds: int = 3,
                 residual_scale: float = 0.3, n_epochs: int = 100,
                 use_scheduler: bool = True):
        self.n_models = n_models
        self.dagger_rounds = dagger_rounds
        self.residual_scale = residual_scale
        self.n_epochs = n_epochs
        self.use_scheduler = use_scheduler
        self._baseline = None
        self._ensemble = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states: np.ndarray, actions: np.ndarray, deltas: np.ndarray,
              state_std: np.ndarray, action_std: float, delta_std: np.ndarray):
        """训练GP基线 + NN集成 + DAgger。"""
        import methods_common as mc
        import methods_evaluate as me

        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        # 1. 训练GP基线
        self._baseline = GPStandard(max_samples=5000)
        self._baseline.train(states, actions, deltas, state_std, action_std, delta_std)

        # 2. 计算残差
        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_base = self._baseline.predict(states[i], actions[i])
            residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std

        # 3. 训练NN集成 + DAgger
        nn_mod = _get_methods_nn()
        ResidualNet = nn_mod.ResidualNet
        DEVICE = nn_mod.DEVICE

        train_inputs_base = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
        all_train_inputs = [train_inputs_base.copy()]
        all_train_residuals = [residuals.copy()]

        self._ensemble_models = []

        for round_i in range(self.dagger_rounds):
            train_inputs = np.vstack(all_train_inputs)
            train_residuals = np.vstack(all_train_residuals)

            # 训练集成
            models = []
            for i in range(self.n_models):
                torch.manual_seed(i * 42)
                np.random.seed(i * 42)

                model = ResidualNet().to(DEVICE)
                train_x = torch.FloatTensor(train_inputs).to(DEVICE)
                train_y = torch.FloatTensor(train_residuals).to(DEVICE)
                ds = torch.utils.data.TensorDataset(train_x, train_y)
                loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)

                if self.use_scheduler:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.n_epochs)
                else:
                    scheduler = None

                crit = torch.nn.MSELoss()
                model.train()
                for epoch in range(self.n_epochs):
                    for xb, yb in loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        loss = crit(model(xb), yb)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    if scheduler:
                        scheduler.step()
                model.eval()
                models.append(model)

            self._ensemble_models = models

            # DAgger: 用模型收集新数据
            if round_i < self.dagger_rounds - 1:
                new_inputs, new_residuals = self._collect_dagger_data(
                    models, state_std, action_std, delta_std, me.make_tau_func, mc.real_step
                )
                if new_inputs:
                    all_train_inputs.append(np.array(new_inputs))
                    all_train_residuals.append(np.array(new_residuals))

    def _collect_dagger_data(self, models, state_std, action_std, delta_std,
                             make_tau_func, real_step):
        """收集DAgger数据。"""
        new_inputs = []
        new_residuals = []
        for seg_i in range(3):
            tau_func = make_tau_func(100 + seg_i, 500)
            phi_init = np.random.uniform(-0.25, 0.25)
            s0 = np.array([phi_init, 0.0, 0.0, 0.0])
            s_base = s0.copy()
            s_real = s0.copy()
            for step in range(500):
                tau = tau_func(step, s_real)
                s_real = real_step(s_real, tau)
                s_norm = s_base / state_std
                a_norm = tau / action_std
                delta_nn = self._predict_ensemble_mean(models, s_norm, a_norm)
                s_base = self._baseline.predict(s_base, tau) + delta_nn * delta_std * self.residual_scale
                if abs(s_real[0]) > math.pi / 3:
                    break
                new_inputs.append(np.concatenate([s_norm, [a_norm]]))
                s_next_base = self._baseline.predict(
                    s_base - delta_nn * delta_std * self.residual_scale, tau
                )
                base_delta = s_next_base - (s_base - delta_nn * delta_std * self.residual_scale)
                real_delta = s_real - (s_base - delta_nn * delta_std * self.residual_scale)
                new_residuals.append((real_delta - base_delta) / delta_std)
        return new_inputs, new_residuals

    def _predict_ensemble_mean(self, models, s_norm, a_norm):
        """集成均值预测。"""
        nn_mod = _get_methods_nn()
        DEVICE = nn_mod.DEVICE
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
        preds = []
        for model in models:
            with torch.no_grad():
                pred = model(s_t, a_t).cpu().numpy()[0]
            preds.append(pred)
        return np.mean(preds, axis=0)

    def _predict_ensemble_with_std(self, models, s_norm, a_norm):
        """集成均值+标准差预测。"""
        nn_mod = _get_methods_nn()
        DEVICE = nn_mod.DEVICE
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
        preds = []
        for model in models:
            with torch.no_grad():
                pred = model(s_t, a_t).cpu().numpy()[0]
            preds.append(pred)
        preds = np.array(preds)
        return np.mean(preds, axis=0), np.std(preds, axis=0)

    def predict(self, s: np.ndarray, tau: float) -> np.ndarray:
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        delta_nn = self._predict_ensemble_mean(self._ensemble_models, s_norm, a_norm)
        return s_next_base + delta_nn * self._delta_std * self.residual_scale

    def predict_with_uncertainty(self, s: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        mean_nn, std_nn = self._predict_ensemble_with_std(
            self._ensemble_models, s_norm, a_norm
        )
        s_next = s_next_base + mean_nn * self._delta_std * self.residual_scale
        uncertainty = std_nn * self._delta_std * self.residual_scale
        return s_next, uncertainty

    def name(self) -> str:
        return f'gp_ensemble_{self.n_models}_{self.dagger_rounds}'


def build_model(model_config: dict, config: dict) -> BaseModel:
    """根据配置构建模型实例。"""
    model_type = model_config['type']

    if model_type == 'reference':
        return RealDynamics()
    elif model_type == 'linearized':
        return LinearizedModel()
    elif model_type == 'gp':
        return GPStandard(
            max_samples=config['gp']['max_samples'],
            n_restarts=config['gp']['n_restarts_optimizer']
        )
    elif model_type == 'gp_sparse':
        return GPB4Sparse(
            n_components=config['gp']['sparse']['n_components']
        )
    elif model_type == 'sindy':
        return SINDy4D(model_file=model_config['model_file'])
    elif model_type == 'ensemble':
        return GPEEnsemble(
            n_models=model_config.get('n_models', 5),
            dagger_rounds=model_config.get('dagger_rounds', 3),
            residual_scale=config['ensemble']['standard']['residual_scale'],
            n_epochs=config['ensemble']['training']['n_epochs'],
            use_scheduler=config['ensemble']['training']['use_scheduler']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
