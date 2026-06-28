"""Evaluation modes for 7D world model."""
import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Complete evaluation result for 7D."""
    states_model: np.ndarray
    states_real: np.ndarray
    actions: np.ndarray
    survival_steps: int
    actual_steps: int
    valid_steps: int
    first_nan_step: int
    first_inf_step: int
    first_divergence_step: int
    termination_reason: str
    failure_before_horizon: bool
    uncertainties: Optional[np.ndarray] = None


def _find_divergence(states: np.ndarray, state_limit: float = 100.0):
    """Find first step where divergence occurs in 7D state."""
    for i in range(len(states)):
        s = states[i]
        if np.any(np.isnan(s)):
            return i, 'nan'
        if np.any(np.isinf(s)):
            return i, 'inf'
        if np.any(np.abs(s) > state_limit):
            return i, 'divergence'
    return len(states), 'none'


def run_mode_a(model, s0: np.ndarray, actions: np.ndarray,
               n_steps: int, real_states: np.ndarray) -> EvalResult:
    """Mode A: Teacher Forcing. Real state at each step.

    Args:
        model: world model
        s0: initial state
        actions: pre-recorded actions
        n_steps: number of steps
        real_states: real states trajectory (n_steps+1, 7)
    """
    states_model = [s0.copy()]
    states_real = [s0.copy()]
    uncertainties = []

    for i in range(n_steps):
        tau = actions[i]
        s_real_next = real_states[i + 1]
        s_next, unc = model.predict_with_uncertainty(real_states[i], tau)
        states_model.append(s_next.copy())
        states_real.append(s_real_next.copy())
        if unc is not None:
            uncertainties.append(unc.copy())

    sm = np.array(states_model)
    sr = np.array(states_real)
    div_step, div_reason = _find_divergence(sr)

    return EvalResult(
        states_model=sm, states_real=sr, actions=actions,
        survival_steps=min(div_step, n_steps),
        actual_steps=n_steps,
        valid_steps=min(div_step, n_steps),
        first_nan_step=div_step if div_reason == 'nan' else n_steps,
        first_inf_step=div_step if div_reason == 'inf' else n_steps,
        first_divergence_step=div_step if div_reason == 'divergence' else n_steps,
        termination_reason=div_reason,
        failure_before_horizon=div_step < n_steps,
        uncertainties=np.array(uncertainties) if uncertainties else None,
    )


def run_mode_b(model, s0: np.ndarray, actions: np.ndarray,
               n_steps: int, real_states: np.ndarray) -> EvalResult:
    """Mode B: Fixed action open-loop. Model self-rolls with frozen actions.

    Args:
        model: world model
        s0: initial state
        actions: pre-recorded actions
        n_steps: number of steps
        real_states: real states trajectory (n_steps+1, 7)
    """
    states_model = [s0.copy()]
    states_real = [s0.copy()]
    s_model = s0.copy()
    uncertainties = []

    for i in range(n_steps):
        tau = actions[i]
        s_next, unc = model.predict_with_uncertainty(s_model, tau)
        states_model.append(s_next.copy())
        states_real.append(real_states[i + 1].copy())
        if unc is not None:
            uncertainties.append(unc.copy())
        s_model = s_next

    sm = np.array(states_model)
    sr = np.array(states_real)
    div_step_model, div_reason_model = _find_divergence(sm)
    div_step_real, _ = _find_divergence(sr)

    return EvalResult(
        states_model=sm, states_real=sr,
        actions=actions,
        survival_steps=min(div_step_real, n_steps),
        actual_steps=n_steps,
        valid_steps=min(div_step_real, n_steps, div_step_model),
        first_nan_step=div_step_model if div_reason_model == 'nan' else n_steps,
        first_inf_step=div_step_model if div_reason_model == 'inf' else n_steps,
        first_divergence_step=min(div_step_model, div_step_real),
        termination_reason=div_reason_model if div_step_model < div_step_real else 'none',
        failure_before_horizon=div_step_model < n_steps or div_step_real < n_steps,
        uncertainties=np.array(uncertainties) if uncertainties else None,
    )


def run_mode_c(model, s0: np.ndarray, actions: np.ndarray,
               n_steps: int, real_states: np.ndarray) -> EvalResult:
    """Mode C: Online controller on real state, model self-rolls.

    In deterministic setting with pre-recorded actions, Mode C uses
    real state for controller (same as Mode B's action source).
    """
    states_model = [s0.copy()]
    states_real = [s0.copy()]
    s_model = s0.copy()
    uncertainties = []

    for i in range(n_steps):
        tau = actions[i]
        s_next, unc = model.predict_with_uncertainty(s_model, tau)
        states_model.append(s_next.copy())
        states_real.append(real_states[i + 1].copy())
        if unc is not None:
            uncertainties.append(unc.copy())
        s_model = s_next

    sm = np.array(states_model)
    sr = np.array(states_real)
    div_step, div_reason = _find_divergence(sr)

    return EvalResult(
        states_model=sm, states_real=sr,
        actions=actions,
        survival_steps=min(div_step, n_steps),
        actual_steps=n_steps,
        valid_steps=min(div_step, n_steps),
        first_nan_step=div_step if div_reason == 'nan' else n_steps,
        first_inf_step=div_step if div_reason == 'inf' else n_steps,
        first_divergence_step=div_step if div_reason == 'divergence' else n_steps,
        termination_reason=div_reason,
        failure_before_horizon=div_step < n_steps,
        uncertainties=np.array(uncertainties) if uncertainties else None,
    )
