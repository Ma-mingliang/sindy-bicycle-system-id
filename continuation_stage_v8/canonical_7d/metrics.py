"""Canonical 7D metrics with per-state, NMAE, survival, and divergence tracking."""
import numpy as np
from .config import STATE_NAMES_7D


def compute_metrics(states_model: np.ndarray, states_real: np.ndarray,
                    state_std: np.ndarray, max_steps: int,
                    survival_steps: int = None,
                    failure_before_horizon: bool = False) -> dict:
    """Compute all metrics with proper divergence handling for 7D."""
    errors = states_model - states_real
    abs_errors = np.abs(errors)

    valid_mask = ~(np.any(np.isnan(abs_errors), axis=1) | np.any(np.isinf(abs_errors), axis=1))
    n_valid = int(np.sum(valid_mask)) - 1
    n_valid = max(n_valid, 0)

    if survival_steps is None:
        survival_steps = n_valid

    result = {'stability': {
        'survival_steps': survival_steps,
        'failure_before_horizon': failure_before_horizon,
    }}

    if n_valid > 0:
        valid_idx = np.where(valid_mask)[0]
        eval_idx = valid_idx[valid_idx > 0]
        if len(eval_idx) > 0:
            ae = abs_errors[eval_idx]
            e = errors[eval_idx]
            result['overall'] = {
                'mae': float(np.mean(ae)),
                'rmse': float(np.sqrt(np.mean(e**2))),
                'nmae': float(np.mean(ae / state_std)),
                'nrmse': float(np.sqrt(np.mean((e / state_std)**2))),
                'max_error': float(np.max(ae)),
                'p95_error': float(np.percentile(ae, 95)),
                'endpoint_error': float(np.max(abs_errors[-1])) if valid_mask[-1] else float('nan'),
            }
        else:
            result['overall'] = {k: 0.0 for k in ['mae', 'rmse', 'nmae', 'nrmse', 'max_error', 'p95_error', 'endpoint_error']}
    else:
        result['overall'] = {k: float('nan') for k in ['mae', 'rmse', 'nmae', 'nrmse', 'max_error', 'p95_error', 'endpoint_error']}

    # Per-state metrics for 7D
    for i, name in enumerate(STATE_NAMES_7D):
        ae_1d = abs_errors[:, i]
        e_1d = errors[:, i]
        valid_1d = valid_mask & ~np.isnan(ae_1d) & ~np.isinf(ae_1d)
        n_v = int(np.sum(valid_1d)) - 1
        n_v = max(n_v, 0)

        if n_v > 0:
            eval_v = np.where(valid_1d)[0]
            eval_v = eval_v[eval_v > 0]
            if len(eval_v) > 0:
                ae_v = ae_1d[eval_v]
                e_v = e_1d[eval_v]
                result[name] = {
                    'mae': float(np.mean(ae_v)),
                    'rmse': float(np.sqrt(np.mean(e_v**2))),
                    'nmae': float(np.mean(ae_v / state_std[i])),
                    'max_error': float(np.max(ae_v)),
                    'p95_error': float(np.percentile(ae_v, 95)),
                    'endpoint_error': float(ae_1d[-1]) if valid_1d[-1] else float('nan'),
                }
            else:
                result[name] = {k: 0.0 for k in ['mae', 'rmse', 'nmae', 'max_error', 'p95_error', 'endpoint_error']}
        else:
            result[name] = {k: float('nan') for k in ['mae', 'rmse', 'nmae', 'max_error', 'p95_error', 'endpoint_error']}

    return result


def compute_cost(states: np.ndarray, actions: np.ndarray, weights: dict) -> float:
    """Compute cumulative cost J for 7D state."""
    if len(states) <= 1 or len(actions) == 0:
        return 0.0
    n = min(len(states) - 1, len(actions))
    J = 0.0
    for k in range(n):
        s = states[k + 1]
        u = actions[k]
        J += weights.get('e_y', 100) * s[0]**2
        J += weights.get('e_psi', 100) * s[1]**2
        J += weights.get('v', 1) * s[2]**2
        J += weights.get('theta', 10) * s[3]**2
        J += weights.get('theta_dot', 1) * s[4]**2
        J += weights.get('delta', 10) * s[5]**2
        J += weights.get('delta_dot', 1) * s[6]**2
        J += weights.get('u', 0.1) * u**2
    return J
