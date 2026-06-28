"""评估模式 v2: 修复 Mode B 实现。

Mode B 修复要点:
1. 先从真实系统生成参考轨迹和固定动作序列
2. 保存参考轨迹作为 ground truth
3. 模型使用固定动作序列自滚动
4. 真实系统也使用相同固定动作序列 (从相同初始状态)
5. 比较模型预测与真实轨迹
"""

import numpy as np
import math
from typing import Callable, Optional, Tuple


def generate_reference_trajectory(s0: np.ndarray, tau_func: Callable,
                                   n_steps: int, real_dynamics) -> dict:
    """从真实系统生成参考轨迹和固定动作序列。

    返回:
        actions: (n_steps,) 固定动作序列
        states_ref: (n_steps+1, 4) 参考状态轨迹
    """
    states_ref = [s0.copy()]
    actions = []
    s = s0.copy()

    for i in range(n_steps):
        tau = tau_func(i, s)
        actions.append(tau)
        s = real_dynamics(s, tau)
        states_ref.append(s.copy())

        if abs(s[0]) > math.pi / 3 or np.any(np.abs(s) > 100):
            break

    return {
        'actions': np.array(actions),
        'states_ref': np.array(states_ref),
        'n_valid_steps': len(actions),
    }


def run_mode_b_fixed(model, s0: np.ndarray, reference: dict,
                      n_steps: int, real_dynamics) -> dict:
    """Mode B (修复版): 固定动作序列开环。

    使用预生成的参考轨迹中的固定动作序列。
    模型和真实系统都从相同初始状态出发，使用相同固定动作。

    这是真正的开环测试: 模型误差不会通过动作反馈放大。
    """
    actions = reference['actions']
    n_valid = min(n_steps, len(actions))

    # 真实系统: 从 s0 出发，使用固定动作
    states_real = [s0.copy()]
    s_real = s0.copy()
    for i in range(n_valid):
        s_real = real_dynamics(s_real, actions[i])
        states_real.append(s_real.copy())
        if abs(s_real[0]) > math.pi / 3 or np.any(np.abs(s_real) > 100):
            break

    # 模型: 从 s0 出发，使用固定动作
    states_model = [s0.copy()]
    s_model = s0.copy()
    uncertainties = []
    for i in range(n_valid):
        tau = actions[i]
        s_next, unc = model.predict_with_uncertainty(s_model, tau)
        s_model = s_next
        states_model.append(s_model.copy())
        if unc is not None:
            uncertainties.append(unc.copy())

        # 检查模型发散
        if np.any(np.isnan(s_model)) or np.any(np.abs(s_model) > 100):
            remaining = n_valid - i - 1
            states_model.extend([np.full(4, np.nan)] * remaining)
            break

    result = {
        'states_model': np.array(states_model),
        'states_real': np.array(states_real),
        'actions': actions[:n_valid],
        'survival_steps': n_valid,
        'instability_detected': False,
    }
    if uncertainties:
        result['uncertainties'] = np.array(uncertainties)
    return result


def run_mode_a(model, s0: np.ndarray, tau_func: Callable,
               n_steps: int, real_dynamics) -> dict:
    """Mode A: Teacher Forcing — 每步用真实状态预测。"""
    states_model = [s0.copy()]
    states_real = [s0.copy()]
    s_real = s0.copy()
    uncertainties = []

    for i in range(n_steps):
        tau = tau_func(i, s_real)
        s_real_next = real_dynamics(s_real, tau)

        if abs(s_real_next[0]) > math.pi / 3 or np.any(np.abs(s_real_next) > 100):
            remaining = n_steps - i
            states_model.extend([np.full(4, np.nan)] * remaining)
            states_real.extend([s_real_next.copy()] + [np.full(4, np.nan)] * (remaining - 1))
            return _pack_result(states_model, states_real, None, None, i + 1, True)

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


def run_mode_c(model, s0: np.ndarray, tau_func: Callable,
               n_steps: int, real_dynamics) -> dict:
    """Mode C: 混合模式 — tau来自真实状态，模型自滚动。"""
    states_model = [s0.copy()]
    states_real = [s0.copy()]
    actions = []
    s_model = s0.copy()
    s_real = s0.copy()
    uncertainties = []

    for i in range(n_steps):
        tau = tau_func(i, s_real)
        actions.append(tau)
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


def run_mode_d(model, s0: np.ndarray, n_steps: int, real_dynamics,
               K_lqr: np.ndarray, rng: np.random.RandomState,
               target: float = 0.0, disturbance_range: Tuple[float, float] = (-0.5, 0.5)) -> dict:
    """Mode D: 完整闭环 — 控制器完全使用模型预测状态。"""
    states_model = [s0.copy()]
    states_real = [s0.copy()]
    actions_real = []
    actions_model = []
    s_model = s0.copy()
    s_real = s0.copy()
    uncertainties = []

    disturbances = rng.uniform(disturbance_range[0], disturbance_range[1], n_steps)

    for i in range(n_steps):
        x_real = np.array([s_real[0] - target, s_real[2], s_real[1], s_real[3]])
        tau_real = float(-K_lqr @ x_real) + disturbances[i]

        x_model = np.array([s_model[0] - target, s_model[2], s_model[1], s_model[3]])
        tau_model = float(-K_lqr @ x_model) + disturbances[i]

        actions_real.append(tau_real)
        actions_model.append(tau_model)

        s_real = real_dynamics(s_real, tau_real)

        if abs(s_real[0]) > math.pi / 3 or np.any(np.abs(s_real) > 100):
            remaining = n_steps - i
            states_model.extend([np.full(4, np.nan)] * remaining)
            states_real.extend([s_real.copy()] + [np.full(4, np.nan)] * (remaining - 1))
            return _pack_result(states_model, states_real, np.array(actions_model),
                              np.array(actions_real), i + 1, True)

        s_next, unc = model.predict_with_uncertainty(s_model, tau_model)
        s_model = s_next
        states_model.append(s_model.copy())
        states_real.append(s_real.copy())
        if unc is not None:
            uncertainties.append(unc.copy())

    result = _pack_result(states_model, states_real, np.array(actions_model),
                          np.array(actions_real), n_steps, False)
    if uncertainties:
        result['uncertainties'] = np.array(uncertainties)
    return result


def _pack_result(states_model, states_real, actions, actions_real,
                 survival_steps, instability_detected) -> dict:
    return {
        'states_model': np.array(states_model),
        'states_real': np.array(states_real),
        'actions': actions,
        'actions_real': actions_real,
        'survival_steps': survival_steps,
        'instability_detected': instability_detected,
    }
