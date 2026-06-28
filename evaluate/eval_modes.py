"""评估模式：Mode A/B/C/D 的统一实现。"""

import numpy as np
import math
from typing import Callable, Optional, Tuple


def run_mode_a(model, s0: np.ndarray, tau_func: Callable,
               n_steps: int, real_dynamics) -> dict:
    """Mode A: Teacher Forcing — 每步用真实状态预测。

    模型接收真实状态和真实力矩，检查单步拟合能力。
    """
    states_model = [s0.copy()]
    states_real = [s0.copy()]
    s_real = s0.copy()
    uncertainties = []

    for i in range(n_steps):
        tau = tau_func(i, s_real)
        s_real_next = real_dynamics(s_real, tau)

        if abs(s_real_next[0]) > math.pi / 3 or np.any(np.abs(s_real_next) > 100):
            # 发散，填充NaN
            remaining = n_steps - i
            states_model.extend([np.full(4, np.nan)] * remaining)
            states_real.extend([s_real_next.copy()] + [np.full(4, np.nan)] * (remaining - 1))
            return _pack_result(states_model, states_real, None, None, i + 1, True)

        # 模型从真实状态预测
        s_next, unc = model.predict_with_uncertainty(s_real, tau)
        states_model.append(s_next.copy())
        states_real.append(s_real_next.copy())
        if unc is not None:
            uncertainties.append(unc.copy())

        s_real = s_real_next

    result = _pack_result(states_model, states_real, None, None, n_steps, False)
    if uncertainties:
        result['uncertainties'] = np.array(uncertainties)
    return result


def run_mode_b(model, s0: np.ndarray, tau_func: Callable,
               n_steps: int, real_dynamics) -> dict:
    """Mode B: 固定动作序列开环 — 从真实系统生成动作，模型自滚动。

    先从真实系统收集完整动作序列，然后模型用相同动作序列自滚动。
    """
    # 第一遍: 从真实系统收集动作
    actions = []
    s_real = s0.copy()
    for i in range(n_steps):
        tau = tau_func(i, s_real)
        actions.append(tau)
        s_real = real_dynamics(s_real, tau)
        if abs(s_real[0]) > math.pi / 3 or np.any(np.abs(s_real) > 100):
            break

    # 第二遍: 模型用固定动作自滚动
    states_model = [s0.copy()]
    states_real = [s0.copy()]
    s_model = s0.copy()
    s_real = s0.copy()
    uncertainties = []

    for i in range(n_steps):
        tau = actions[i] if i < len(actions) else actions[-1]
        s_real = real_dynamics(s_real, tau)

        if abs(s_real[0]) > math.pi / 3 or np.any(np.abs(s_real) > 100):
            remaining = n_steps - i
            states_model.extend([np.full(4, np.nan)] * remaining)
            states_real.extend([s_real.copy()] + [np.full(4, np.nan)] * (remaining - 1))
            return _pack_result(states_model, states_real, np.array(actions), None, i + 1, True)

        s_next, unc = model.predict_with_uncertainty(s_model, tau)
        s_model = s_next
        states_model.append(s_model.copy())
        states_real.append(s_real.copy())
        if unc is not None:
            uncertainties.append(unc.copy())

    result = _pack_result(states_model, states_real, np.array(actions), None, n_steps, False)
    if uncertainties:
        result['uncertainties'] = np.array(uncertainties)
    return result


def run_mode_c(model, s0: np.ndarray, tau_func: Callable,
               n_steps: int, real_dynamics) -> dict:
    """Mode C: 混合模式 — tau来自真实状态，模型自滚动。

    当前最常用的评估模式。控制器用真实状态计算力矩，但模型从自身预测滚动。
    """
    states_model = [s0.copy()]
    states_real = [s0.copy()]
    actions = []
    s_model = s0.copy()
    s_real = s0.copy()
    uncertainties = []

    for i in range(n_steps):
        # tau 从真实状态计算
        tau = tau_func(i, s_real)
        actions.append(tau)

        # 真实系统前进一步
        s_real = real_dynamics(s_real, tau)

        if abs(s_real[0]) > math.pi / 3 or np.any(np.abs(s_real) > 100):
            remaining = n_steps - i
            states_model.extend([np.full(4, np.nan)] * remaining)
            states_real.extend([s_real.copy()] + [np.full(4, np.nan)] * (remaining - 1))
            return _pack_result(states_model, states_real, np.array(actions), None, i + 1, True)

        # 模型从自身状态预测
        s_next, unc = model.predict_with_uncertainty(s_model, tau)
        s_model = s_next
        states_model.append(s_model.copy())
        states_real.append(s_real.copy())
        if unc is not None:
            uncertainties.append(unc.copy())

    result = _pack_result(states_model, states_real, np.array(actions), None, n_steps, False)
    if uncertainties:
        result['uncertainties'] = np.array(uncertainties)
    return result


def run_mode_d(model, s0: np.ndarray, n_steps: int, real_dynamics,
               K_lqr: np.ndarray, rng: np.random.RandomState,
               target: float = 0.0, disturbance_range: Tuple[float, float] = (-0.5, 0.5)) -> dict:
    """Mode D: 完整闭环 — 控制器完全使用模型预测状态。

    真实系统和模型各自独立运行LQR控制器。
    """
    states_model = [s0.copy()]
    states_real = [s0.copy()]
    actions_real = []
    actions_model = []
    s_model = s0.copy()
    s_real = s0.copy()
    uncertainties = []

    disturbances = rng.uniform(disturbance_range[0], disturbance_range[1], n_steps)

    for i in range(n_steps):
        # 真实系统LQR
        x_real = np.array([s_real[0] - target, s_real[2], s_real[1], s_real[3]])
        tau_real = float(-K_lqr @ x_real) + disturbances[i]

        # 模型LQR (使用模型自身状态)
        x_model = np.array([s_model[0] - target, s_model[2], s_model[1], s_model[3]])
        tau_model = float(-K_lqr @ x_model) + disturbances[i]

        actions_real.append(tau_real)
        actions_model.append(tau_model)

        # 真实系统前进一步
        s_real = real_dynamics(s_real, tau_real)

        if abs(s_real[0]) > math.pi / 3 or np.any(np.abs(s_real) > 100):
            remaining = n_steps - i
            states_model.extend([np.full(4, np.nan)] * remaining)
            states_real.extend([s_real.copy()] + [np.full(4, np.nan)] * (remaining - 1))
            return _pack_result(
                states_model, states_real,
                np.array(actions_model), np.array(actions_real),
                i + 1, True
            )

        # 模型前进一步
        s_next, unc = model.predict_with_uncertainty(s_model, tau_model)
        s_model = s_next
        states_model.append(s_model.copy())
        states_real.append(s_real.copy())
        if unc is not None:
            uncertainties.append(unc.copy())

    result = _pack_result(
        states_model, states_real,
        np.array(actions_model), np.array(actions_real),
        n_steps, False
    )
    if uncertainties:
        result['uncertainties'] = np.array(uncertainties)
    return result


def _pack_result(states_model, states_real, actions, actions_real,
                 survival_steps, instability_detected) -> dict:
    """打包评估结果。"""
    return {
        'states_model': np.array(states_model),
        'states_real': np.array(states_real),
        'actions': actions,
        'actions_real': actions_real,
        'survival_steps': survival_steps,
        'instability_detected': instability_detected,
    }
