"""Evaluation modes A/B/C/D with proper divergence handling."""
import numpy as np
import math
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Complete evaluation result with all required fields."""
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

    def to_dict(self):
        return {
            'states_model': self.states_model,
            'states_real': self.states_real,
            'actions': self.actions,
            'survival_steps': self.survival_steps,
            'actual_steps': self.actual_steps,
            'valid_steps': self.valid_steps,
            'first_nan_step': self.first_nan_step,
            'first_inf_step': self.first_inf_step,
            'first_divergence_step': self.first_divergence_step,
            'termination_reason': self.termination_reason,
            'failure_before_horizon': self.failure_before_horizon,
            'uncertainties': self.uncertainties,
        }


def _find_divergence(states: np.ndarray, phi_limit: float = np.pi/3,
                     state_limit: float = 100.0):
    """Find first step where divergence occurs."""
    for i in range(len(states)):
        s = states[i]
        if np.any(np.isnan(s)):
            return i, 'nan'
        if np.any(np.isinf(s)):
            return i, 'inf'
        if abs(s[0]) > phi_limit or np.any(np.abs(s) > state_limit):
            return i, 'divergence'
    return len(states), 'none'


def run_mode_a(model, s0: np.ndarray, tau_func: Callable,
               n_steps: int, dynamics) -> EvalResult:
    """Mode A: Teacher Forcing. Real state at each step."""
    states_model = [s0.copy()]
    states_real = [s0.copy()]
    s_real = s0.copy()
    uncertainties = []

    for i in range(n_steps):
        tau = tau_func(i, s_real)
        s_real_next = dynamics.step(s_real, tau)
        s_next, unc = model.predict_with_uncertainty(s_real, tau)
        states_model.append(s_next.copy())
        states_real.append(s_real_next.copy())
        if unc is not None:
            uncertainties.append(unc.copy())
        s_real = s_real_next

    sm = np.array(states_model)
    sr = np.array(states_real)
    div_step, div_reason = _find_divergence(sr)

    return EvalResult(
        states_model=sm, states_real=sr, actions=np.zeros(n_steps),
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


def run_mode_b(model, s0: np.ndarray, tau_func: Callable,
               n_steps: int, dynamics) -> EvalResult:
    """Mode B: Fixed action open-loop. Actions frozen before model rollout."""
    # First pass: collect actions from real system
    actions = []
    s_real = s0.copy()
    for i in range(n_steps):
        tau = tau_func(i, s_real)
        actions.append(tau)
        s_real = dynamics.step(s_real, tau)
        if dynamics.is_diverged(s_real):
            break

    # Second pass: model rolls out with frozen actions
    states_model = [s0.copy()]
    states_real = [s0.copy()]
    s_model = s0.copy()
    s_real = s0.copy()
    uncertainties = []
    first_div = len(actions)
    div_reason = 'none'

    for i in range(n_steps):
        tau = actions[i] if i < len(actions) else actions[-1]
        s_real = dynamics.step(s_real, tau)
        s_next, unc = model.predict_with_uncertainty(s_model, tau)
        states_model.append(s_next.copy())
        states_real.append(s_real.copy())
        if unc is not None:
            uncertainties.append(unc.copy())
        s_model = s_next

        if dynamics.is_diverged(s_real) and first_div == len(actions):
            first_div = i + 1
            if np.any(np.isnan(s_real)):
                div_reason = 'nan'
            elif np.any(np.isinf(s_real)):
                div_reason = 'inf'
            else:
                div_reason = 'divergence'

    sm = np.array(states_model)
    sr = np.array(states_real)
    div_step_model, div_reason_model = _find_divergence(sm)

    return EvalResult(
        states_model=sm, states_real=sr,
        actions=np.array(actions),
        survival_steps=min(first_div, n_steps),
        actual_steps=n_steps,
        valid_steps=min(first_div, n_steps, div_step_model),
        first_nan_step=div_step_model if div_reason_model == 'nan' else n_steps,
        first_inf_step=div_step_model if div_reason_model == 'inf' else n_steps,
        first_divergence_step=min(div_step_model, first_div),
        termination_reason=div_reason if first_div < n_steps else div_reason_model,
        failure_before_horizon=first_div < n_steps or div_step_model < n_steps,
        uncertainties=np.array(uncertainties) if uncertainties else None,
    )


def run_mode_c(model, s0: np.ndarray, tau_func: Callable,
               n_steps: int, dynamics) -> EvalResult:
    """Mode C: Online controller on real state, model self-rolls.

    In deterministic setting with same tau_func, Mode C uses real state
    for controller (same as Mode B's action source), but model state
    drifts independently. When controller reads real state, actions are
    identical to Mode B. Numerical equivalence holds in current setup.
    """
    states_model = [s0.copy()]
    states_real = [s0.copy()]
    actions = []
    s_model = s0.copy()
    s_real = s0.copy()
    uncertainties = []

    for i in range(n_steps):
        tau = tau_func(i, s_real)
        actions.append(tau)
        s_real_next = dynamics.step(s_real, tau)
        s_next, unc = model.predict_with_uncertainty(s_model, tau)
        states_model.append(s_next.copy())
        states_real.append(s_real_next.copy())
        if unc is not None:
            uncertainties.append(unc.copy())
        s_model = s_next
        s_real = s_real_next

    sm = np.array(states_model)
    sr = np.array(states_real)
    div_step, div_reason = _find_divergence(sr)

    return EvalResult(
        states_model=sm, states_real=sr,
        actions=np.array(actions),
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


def run_mode_d(model, s0: np.ndarray, n_steps: int, dynamics,
               K_lqr: np.ndarray, rng: np.random.RandomState,
               target: float = 0.0) -> EvalResult:
    """Mode D: Closed-loop. Controller uses model predicted state."""
    states_model = [s0.copy()]
    states_real = [s0.copy()]
    actions_model = []
    actions_real = []
    s_model = s0.copy()
    s_real = s0.copy()
    uncertainties = []
    disturbances = rng.uniform(-0.5, 0.5, n_steps)

    for i in range(n_steps):
        x_real = np.array([s_real[0] - target, s_real[2], s_real[1], s_real[3]])
        tau_real = float(-K_lqr @ x_real) + disturbances[i]
        x_model = np.array([s_model[0] - target, s_model[2], s_model[1], s_model[3]])
        tau_model = float(-K_lqr @ x_model) + disturbances[i]

        actions_real.append(tau_real)
        actions_model.append(tau_model)

        s_real_next = dynamics.step(s_real, tau_real)
        s_next, unc = model.predict_with_uncertainty(s_model, tau_model)

        states_model.append(s_next.copy())
        states_real.append(s_real_next.copy())
        if unc is not None:
            uncertainties.append(unc.copy())
        s_model = s_next
        s_real = s_real_next

    sm = np.array(states_model)
    sr = np.array(states_real)
    div_step_real, _ = _find_divergence(sr)
    div_step_model, _ = _find_divergence(sm)

    return EvalResult(
        states_model=sm, states_real=sr,
        actions=np.array(actions_model),
        survival_steps=min(div_step_real, n_steps),
        actual_steps=n_steps,
        valid_steps=min(div_step_real, div_step_model, n_steps),
        first_nan_step=n_steps,
        first_inf_step=n_steps,
        first_divergence_step=min(div_step_real, div_step_model),
        termination_reason='none',
        failure_before_horizon=div_step_real < n_steps or div_step_model < n_steps,
        uncertainties=np.array(uncertainties) if uncertainties else None,
    )
