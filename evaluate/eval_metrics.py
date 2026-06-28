"""指标计算：统一的评估指标集合。"""

import numpy as np
import math
from typing import Optional


def compute_all_metrics(states_model: np.ndarray, states_real: np.ndarray,
                        actions_model: Optional[np.ndarray],
                        actions_real: Optional[np.ndarray],
                        state_std: np.ndarray,
                        max_steps: int) -> dict:
    """计算所有评估指标。

    Args:
        states_model: (n+1, 4) 模型轨迹
        states_real: (n+1, 4) 真实轨迹
        actions_model: (n,) 模型控制力矩 (Mode D)
        actions_real: (n,) 真实控制力矩 (Mode D)
        state_std: (4,) 训练数据标准差
        max_steps: 最大步数

    Returns:
        dict: 所有指标
    """
    # 逐状态误差
    errors = states_model - states_real  # (n+1, 4)
    abs_errors = np.abs(errors)

    result = {}

    # 全局指标
    result['overall'] = _compute_overall_metrics(abs_errors, errors, state_std)

    # 逐状态指标
    state_names = ['phi', 'delta', 'phi_dot', 'delta_dot']
    for i, name in enumerate(state_names):
        result[name] = _compute_per_state_metrics(
            abs_errors[:, i], errors[:, i], state_std[i], name
        )

    # 稳定性指标
    n_steps = len(states_model) - 1
    survival_steps = _compute_survival_steps(states_real, max_steps)
    result['stability'] = {
        'survival_steps': survival_steps,
        'instability_detected': survival_steps < max_steps,
        'fall_rate': 1.0 if survival_steps < max_steps else 0.0,
    }

    # 控制力矩指标 (Mode D)
    if actions_model is not None and actions_real is not None:
        result['control'] = _compute_control_metrics(actions_model, actions_real)
    else:
        result['control'] = {}

    return result


def _compute_overall_metrics(abs_errors: np.ndarray, errors: np.ndarray,
                             state_std: np.ndarray) -> dict:
    """计算整体指标。"""
    valid = ~np.any(np.isnan(abs_errors), axis=1)
    if not np.any(valid):
        return {k: float('nan') for k in [
            'mae', 'rmse', 'max_error', 'median_error', 'p95_error',
            'endpoint_error', 'normalized_mae'
        ]}

    abs_err_valid = abs_errors[valid]
    err_valid = errors[valid]

    return {
        'mae': float(np.mean(abs_err_valid)),
        'rmse': float(np.sqrt(np.mean(err_valid ** 2))),
        'max_error': float(np.max(abs_err_valid)),
        'median_error': float(np.median(abs_err_valid)),
        'p95_error': float(np.percentile(abs_err_valid, 95)),
        'endpoint_error': float(np.mean(abs_errors[-1])),
        'normalized_mae': float(np.mean(abs_err_valid / state_std)),
    }


def _compute_per_state_metrics(abs_errors_1d: np.ndarray, errors_1d: np.ndarray,
                               state_std_val: float, state_name: str) -> dict:
    """计算单个状态的指标。"""
    valid = ~np.isnan(abs_errors_1d)
    if not np.any(valid):
        return {k: float('nan') for k in [
            'mae', 'rmse', 'max_error', 'median_error', 'p95_error',
            'endpoint_error', 'normalized_mae', 'first_exceed_1sigma',
            'first_exceed_2sigma', 'first_exceed_3sigma'
        ]}

    abs_valid = abs_errors_1d[valid]
    err_valid = errors_1d[valid]

    # 首次超过阈值的步数
    sigma1 = state_std_val
    sigma2 = 2 * state_std_val
    sigma3 = 3 * state_std_val

    def first_exceed(threshold):
        idx = np.where(abs_errors_1d > threshold)[0]
        return int(idx[0]) if len(idx) > 0 else -1

    return {
        'mae': float(np.mean(abs_valid)),
        'rmse': float(np.sqrt(np.mean(err_valid ** 2))),
        'max_error': float(np.max(abs_valid)),
        'median_error': float(np.median(abs_valid)),
        'p95_error': float(np.percentile(abs_valid, 95)),
        'endpoint_error': float(abs_errors_1d[-1]) if not np.isnan(abs_errors_1d[-1]) else float('nan'),
        'normalized_mae': float(np.mean(abs_valid / state_std_val)),
        'first_exceed_1sigma': first_exceed(sigma1),
        'first_exceed_2sigma': first_exceed(sigma2),
        'first_exceed_3sigma': first_exceed(sigma3),
    }


def _compute_survival_steps(states_real: np.ndarray, max_steps: int) -> int:
    """计算存活步数（真实轨迹是否发散）。"""
    for i in range(len(states_real)):
        if np.any(np.isnan(states_real[i])):
            return i
        if np.any(np.abs(states_real[i]) > 100.0):
            return i
    return max_steps


def _compute_control_metrics(actions_model: np.ndarray, actions_real: np.ndarray) -> dict:
    """计算控制力矩指标。"""
    valid = ~(np.isnan(actions_model) | np.isnan(actions_real))
    if not np.any(valid):
        return {'torque_mae': float('nan'), 'torque_max_error': float('nan')}

    diff = np.abs(actions_model[valid] - actions_real[valid])
    return {
        'torque_mae': float(np.mean(diff)),
        'torque_max_error': float(np.max(diff)),
    }


def compute_uncertainty_metrics(ensemble_stds: np.ndarray,
                                abs_errors: np.ndarray) -> dict:
    """计算不确定性指标。

    Args:
        ensemble_stds: (n, 4) 集成标准差
        abs_errors: (n, 4) 绝对误差

    Returns:
        dict: 不确定性相关指标
    """
    if ensemble_stds is None or len(ensemble_stds) == 0:
        return {}

    # 不确定性与误差的相关性
    correlations = {}
    state_names = ['phi', 'delta', 'phi_dot', 'delta_dot']
    for i, name in enumerate(state_names):
        std_col = ensemble_stds[:, i]
        err_col = abs_errors[:len(std_col), i]
        valid = ~(np.isnan(std_col) | np.isnan(err_col))
        if np.sum(valid) > 2:
            corr = np.corrcoef(std_col[valid], err_col[valid])[0, 1]
            correlations[f'{name}_corr'] = float(corr)
        else:
            correlations[f'{name}_corr'] = float('nan')

    # 平均不确定性
    mean_std = np.mean(ensemble_stds, axis=0)
    result = {
        'mean_uncertainty': {name: float(mean_std[i]) for i, name in enumerate(state_names)},
        'uncertainty_error_correlation': correlations,
    }

    return result
