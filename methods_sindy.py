"""SINDy方法1-3：多项式库、三角函数库、三角+指数库。

统一接口：
    train(states, actions, deltas, state_std, action_std, delta_std)
    predict(s, tau) -> s_next
"""

import numpy as np
from typing import Tuple


# ============================================================
# STLSQ稀疏回归
# ============================================================
def stlsq(Theta: np.ndarray, dXdt: np.ndarray, threshold: float = 0.05, max_iter: int = 20) -> np.ndarray:
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
# 库函数
# ============================================================
def build_poly_library(states_norm: np.ndarray, actions_norm: np.ndarray) -> Tuple[np.ndarray, list]:
    """标准多项式库：常数+线性+二次交叉项（21特征）。"""
    n = len(states_norm)
    x = np.column_stack([states_norm, actions_norm.reshape(-1, 1)])
    features = [np.ones(n)]
    names = ['1']
    for i in range(5):
        features.append(x[:, i])
        names.append(f'x{i}')
    for i in range(5):
        for j in range(i, 5):
            features.append(x[:, i] * x[:, j])
            names.append(f'x{i}*x{j}')
    return np.column_stack(features), names


def build_trig_library(states_norm: np.ndarray, actions_norm: np.ndarray) -> Tuple[np.ndarray, list]:
    """多项式+三角函数库（32特征）。"""
    n = len(states_norm)
    phi = states_norm[:, 0]
    x = np.column_stack([states_norm, actions_norm.reshape(-1, 1)])
    features = [np.ones(n)]
    names = ['1']
    for i in range(5):
        features.append(x[:, i])
        names.append(f'x{i}')
    for i in range(5):
        for j in range(i, 5):
            features.append(x[:, i] * x[:, j])
            names.append(f'x{i}*x{j}')
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    trig_items = [
        (sin_phi, 'sin(phi)'), (cos_phi, 'cos(phi)'),
        (sin_phi * phi, 'sin(phi)*phi'), (sin_phi * x[:, 1], 'sin(phi)*delta'),
        (sin_phi * x[:, 2], 'sin(phi)*wd'), (sin_phi * x[:, 3], 'sin(phi)*wdd'),
        (sin_phi * x[:, 4], 'sin(phi)*tau'),
        (cos_phi * phi, 'cos(phi)*phi'), (cos_phi * x[:, 2], 'cos(phi)*wd'),
        (phi * phi * phi, 'phi^3'),
    ]
    for val, name in trig_items:
        features.append(val)
        names.append(name)
    return np.column_stack(features), names


def build_trig_exp_library(states_norm: np.ndarray, actions_norm: np.ndarray) -> Tuple[np.ndarray, list]:
    """多项式+三角+指数库（36特征）。"""
    Theta_trig, names = build_trig_library(states_norm, actions_norm)
    phi = states_norm[:, 0]
    delta = states_norm[:, 1]
    exp_phi = np.exp(-phi**2)
    exp_delta = np.exp(-delta**2)
    extra = [
        (exp_phi, 'exp(-phi^2)'), (exp_delta, 'exp(-delta^2)'),
        (exp_phi * states_norm[:, 2], 'exp(-phi^2)*wd'),
        (exp_phi * states_norm[:, 3], 'exp(-phi^2)*wdd'),
    ]
    cols = [Theta_trig]
    for val, name in extra:
        cols.append(val.reshape(-1, 1))
        names.append(name)
    return np.hstack(cols), names


# ============================================================
# SINDy方法基类
# ============================================================
class _SINDyBase:
    """SINDy方法通用逻辑。"""
    name: str = 'SINDyBase'
    _build_fn = None

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.Xi = None
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
        Theta, self._names = self._build_fn(states_norm, actions_norm)
        self.Xi = stlsq(Theta, deltas_norm, threshold=self.threshold)
        pred = Theta @ self.Xi
        rmse = np.sqrt(np.mean((pred - deltas_norm) ** 2, axis=0))
        return rmse * delta_std

    def predict(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        Theta, _ = self._build_fn(s_norm.reshape(1, -1), np.array([a_norm]))
        delta_norm = (Theta @ self.Xi)[0]
        return s + delta_norm * self._delta_std


class SINDyPoly(_SINDyBase):
    name = '1_SINDyPoly'
    _build_fn = staticmethod(build_poly_library)


class SINDyTrig(_SINDyBase):
    name = '2_SINDyTrig'
    _build_fn = staticmethod(build_trig_library)


class SINDyTrigExp(_SINDyBase):
    name = '3_SINDyTrigExp'
    _build_fn = staticmethod(build_trig_exp_library)
