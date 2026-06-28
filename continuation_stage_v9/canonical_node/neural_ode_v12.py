"""V12 Neural ODE: Sequential Segments + Extended Curriculum + RK4 + Contractivity.

Fixes 4 critical bugs in V9 that cause severe rollout degradation
(H=10 NMAE=0.0628 but H=500 NMAE=0.5529):

Bug 1 (MOST CRITICAL): Shuffled DataLoader breaks multi-step loss.
    V9 uses shuffle=True on individual timesteps, but multi-step loss assumes
    sb[i+1] is the temporal successor of sb[i]. Fix: SequentialSegmentDataset
    that returns contiguous trajectory segments. Shuffling occurs over segments,
    not within segments, preserving temporal ordering.

Bug 2: Training rollout caps at 20, evaluation goes to 500+.
    V9 curriculum is '1,5,10,20' but evaluation tests H up to 500.
    Fix: Extended curriculum '1,5,10,20,50,100' with 400 epochs.

Bug 3: Single-step loss dominates with lambda_multi=0.3.
    Combined with Bug 1, multi-step loss is pure noise.
    Fix: lambda_multi ramps from 0.1 to 0.7 over training.

Bug 4: Euler integration with dt=1/30 causes large truncation error.
    Fix: Support RK4 and Euler-substep integration methods.

Additional improvements:
    - Contractivity regularization (from V11) with warmup scheduling
    - Adaptive curriculum (from adaptive_curriculum.py) with EMA error monitoring
    - Gradient norm monitoring and adaptive clipping
    - Proper validation across all curriculum horizons

Architecture:
    Same ODEFunc as V9: dx/dt = f(x, u) where x in R^7, u in R^1.
    Only loss and training procedure changes -- no architecture modifications.
"""
import logging
import os
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from .config_v9 import STATE_DIM, ACTION_DIM
from .neural_ode_v9 import ODEFunc, NeuralODEConfig
from .neural_ode_contractive import (
    compute_jacobian,
    max_eigenvalue_symmetric,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class V12Config(NeuralODEConfig):
    """V12 Neural ODE configuration.

    Extends V9 config with sequential batch sampling, extended curriculum,
    integration method options, and contractivity regularization.
    """

    # --- Sequential batch ---
    max_segment_len: int = 100
    # Max trajectory segment length for training. The dataset creates contiguous
    # segments of this length from the full trajectory data, ensuring temporal
    # ordering within each batch for correct multi-step rollout loss.

    # --- Integration ---
    integration_method: str = 'euler'
    # 'euler': standard Euler step s_next = s + f(s,u)*dt
    # 'rk4': 4th-order Runge-Kutta (lower truncation error)
    # 'euler_substep': Euler with N sub-steps of dt/N each
    sub_steps: int = 2
    # Number of sub-steps for 'euler_substep' method

    # --- Extended rollout curriculum ---
    rollout_curriculum: str = '1,5,10,20,50,100'
    n_epochs: int = 400
    batch_size: int = 256

    # --- Adaptive curriculum ---
    use_adaptive_curriculum: bool = True
    adaptive_thresholds: str = '5:0.02,10:0.04,20:0.08,50:0.15,100:0.30'
    ema_decay: float = 0.95
    eval_interval: int = 10
    min_epochs_per_level: int = 20
    max_epochs_per_level: int = 80

    # --- Loss weights ---
    lambda_multi_start: float = 0.1
    lambda_multi_end: float = 0.7
    lambda_consistency: float = 0.1
    lambda_jacobian: float = 0.01

    # --- Contractivity ---
    use_contractivity: bool = True
    lambda_contractive: float = 0.05
    contractive_warmup_epochs: int = 20
    contractive_ramp_epochs: int = 50

    # --- Survival-aware loss ---
    use_survival_loss: bool = False
    lambda_survival: float = 0.1
    survival_limit_scale: float = 0.95  # fraction of physical limit to enforce


# ---------------------------------------------------------------------------
# Trajectory Segment Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    """A contiguous trajectory segment with precomputed tensor data.

    Each segment holds states[0..max_H], actions[0..max_H-1], and
    deltas[0..max_H-1] from a single driving trajectory. This structure
    enables correct multi-step rollout: rollout from states[0] for max_H
    steps using actions[0..max_H-1], comparing to states[1..max_H].

    Memory layout per segment: (max_H+1)*7 + max_H*1 + max_H*7 = 15H + 7 floats.
    """
    states_t: torch.FloatTensor    # [max_H+1, STATE_DIM] -- next states included
    actions_t: torch.FloatTensor   # [max_H, ACTION_DIM]
    deltas_t: torch.FloatTensor    # [max_H, STATE_DIM]
    length: int                    # Original trajectory length


class SequentialSegmentDataset(torch.utils.data.Dataset):
    """Dataset that returns contiguous trajectory segments for multi-step loss.

    Unlike V9's TensorDataset with shuffle=True (which breaks temporal ordering),
    this dataset:
    1. Stores full trajectory data from each driving segment
    2. Extracts contiguous segments of max_segment_len steps
    3. Precomputes tensor data for fast indexing
    4. Shuffles over segments (not within segments)

    This ensures that within each batch, states[i+1] IS the temporal successor
    of states[i], making multi-step rollout loss correct.

    The data has ~106K samples from 5 driving segments (~1100 steps each).
    Segments are only created within a single driving segment -- never across
    segment boundaries where discontinuities exist.
    """

    def __init__(self, states: np.ndarray, actions: np.ndarray,
                 deltas: np.ndarray, max_segment_len: int,
                 episode_boundaries=None):
        """
        Args:
            states: [N, STATE_DIM] -- normalized states from ALL driving segments
            actions: [N, 1] -- normalized actions
            deltas: [N, STATE_DIM] -- normalized ds/dt targets
            max_segment_len: maximum segment length (controls rollout horizon)
            episode_boundaries: optional list of (start, end) tuples. If provided,
                segments are only created within episodes. If None, auto-detects
                boundaries from state discontinuities.
        """
        self.max_segment_len = max_segment_len
        self.segments: List[Segment] = []

        if episode_boundaries is not None:
            traj_starts = [b[0] for b in episode_boundaries]
            traj_ends = [b[1] for b in episode_boundaries]
            logger.info(
                "SequentialSegmentDataset: using %d explicit episode boundaries",
                len(episode_boundaries),
            )
        else:
            # Fallback: detect segment boundaries from state jumps
            state_diffs = np.abs(np.diff(states, axis=0)).max(axis=1)
            boundary_mask = state_diffs > 2.0  # lowered from 10.0 for normalized data
            boundaries = np.where(boundary_mask)[0] + 1
            traj_starts = np.concatenate([[0], boundaries]).tolist()
            traj_ends = np.concatenate([boundaries, [len(states)]]).tolist()
            logger.info(
                "SequentialSegmentDataset: auto-detected %d segment boundaries",
                len(boundaries),
            )

        logger.info(
            "SequentialSegmentDataset: %d episodes, max_segment_len=%d",
            len(traj_starts), max_segment_len,
        )

        for seg_idx, (start, end) in enumerate(zip(traj_starts, traj_ends)):
            seg_len = end - start
            if seg_len < 2:
                continue

            # Maximum steps we can take from this segment
            # Need (max_segment_len + 1) states for max_segment_len rollout steps
            max_steps = min(max_segment_len, seg_len - 1)

            if max_steps < 1:
                continue

            # Create segment: states[0..max_steps], actions[0..max_steps-1], deltas[0..max_steps-1]
            seg_states = states[start:start + max_steps + 1].copy()
            seg_actions = actions[start:start + max_steps].copy()
            seg_deltas = deltas[start:start + max_steps].copy()

            # Precompute tensor data to avoid repeated numpy->torch conversion
            states_t = torch.FloatTensor(seg_states)
            actions_t = torch.FloatTensor(seg_actions)
            deltas_t = torch.FloatTensor(seg_deltas)

            self.segments.append(Segment(
                states_t=states_t,
                actions_t=actions_t,
                deltas_t=deltas_t,
                length=seg_len,
            ))

        logger.info(
            "SequentialSegmentDataset: created %d segments "
            "(max_segment_len=%d, max_steps_per_segment=%d)",
            len(self.segments), max_segment_len,
            min(max_segment_len, max(s.length - 1 for s in self.segments) if self.segments else 0),
        )

        # Index into self.segments by valid max_H
        self._valid_indices: List[int] = []
        self.set_max_horizon(max_segment_len)

    def set_max_horizon(self, max_h: int) -> None:
        """Update the active horizon for multi-step rollout.

        Filters segments to only include those with enough remaining steps
        for the requested horizon. Called by the curriculum scheduler when
        advancing to longer rollout horizons.

        Args:
            max_h: current maximum rollout horizon
        """
        self.max_segment_len = max_h

        # A segment can support max_h rollout steps if it has at least
        # max_h + 1 states (max_h transitions + 1 initial state).
        # Also need at least 2 states for single-step loss during warmup.
        min_len = max(max_h, 2) + 1
        self._valid_indices = [
            i for i, seg in enumerate(self.segments) if seg.length >= min_len
        ]

        if not self._valid_indices:
            logger.warning(
                "No segments valid for max_h=%d (min required length: %d). "
                "Falling back to all segments.", max_h, min_len
            )
            self._valid_indices = list(range(len(self.segments)))

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Return a contiguous trajectory segment.

        Returns:
            states_t: [max_H+1, STATE_DIM] -- including next state for ground truth
            actions_t: [max_H, ACTION_DIM]
            deltas_t: [max_H, STATE_DIM] -- ds/dt targets
        """
        seg = self.segments[self._valid_indices[idx]]

        # Use the full segment length (already bounded by max_segment_len at creation)
        H = seg.states_t.shape[0] - 1  # max_H = number of states - 1

        # Safety: ensure at least 1 step
        H = max(H, 1)

        return (
            seg.states_t[:H + 1],   # [H+1, STATE_DIM]
            seg.actions_t[:H],      # [H, ACTION_DIM]
            seg.deltas_t[:H],       # [H, STATE_DIM]
        )


# ---------------------------------------------------------------------------
# Contractivity Loss (from V11, self-contained)
# ---------------------------------------------------------------------------

def _contractivity_loss(model: ODEFunc, s: torch.Tensor, a: torch.Tensor):
    """Compute contractivity-promoting loss: mean(ReLU(max_eigenvalue(J_sym))).

    Penalizes positive eigenvalues of the Jacobian's symmetric part,
    encouraging the ODE to be locally contractive (nearby trajectories converge).

    Args:
        model: ODEFunc dynamics network
        s: [batch, STATE_DIM] with requires_grad=True
        a: [batch, ACTION_DIM]

    Returns:
        loss: scalar contractivity loss
        max_eigs: max eigenvalue per sample [batch] (detached, for diagnostics)
    """
    J = compute_jacobian(model, s, a)
    max_eigs, _ = max_eigenvalue_symmetric(J)
    loss = torch.relu(max_eigs).mean()
    return loss, max_eigs.detach()


# ---------------------------------------------------------------------------
# Integration Methods
# ---------------------------------------------------------------------------

def _integrate_euler(model: ODEFunc, s: torch.Tensor, a: torch.Tensor,
                     dt: float) -> torch.Tensor:
    """Standard Euler integration: s_next = s + f(s, a).

    NOTE: The model is trained on targets = deltas / (delta_std * dt),
    so the model output is ALREADY scaled by dt. Do NOT multiply by dt again.
    """
    return s + model(s, a)


def _integrate_rk4(model: ODEFunc, s: torch.Tensor, a: torch.Tensor,
                   dt: float) -> torch.Tensor:
    """4th-order Runge-Kutta integration.

    Since model output is already scaled by dt (target = rate * dt),
    we use half-steps as model(s + k/2) for the RK4 stages.

    k1 = f(s)
    k2 = f(s + k1/2)
    k3 = f(s + k2/2)
    k4 = f(s + k3)
    s_next = s + (k1 + 2*k2 + 2*k3 + k4) / 6

    Reduces truncation error compared to single Euler step.
    """
    k1 = model(s, a)
    k2 = model(s + k1 / 2.0, a)
    k3 = model(s + k2 / 2.0, a)
    k4 = model(s + k3, a)
    return s + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _integrate_euler_substep(model: ODEFunc, s: torch.Tensor, a: torch.Tensor,
                             dt: float, n_substeps: int = 2) -> torch.Tensor:
    """Euler integration with multiple sub-steps per data step.

    Since model output is scaled by dt, each sub-step outputs dt/N worth
    of change. We accumulate N sub-steps for the full dt.
    """
    s_cur = s
    for _ in range(n_substeps):
        # Model outputs rate*dt, so for sub-step we need rate*(dt/N)
        # Approximate: model(s) gives rate*dt, divide by N for sub-step
        s_cur = s_cur + model(s_cur, a) / n_substeps
    return s_cur


def _integrate(model: ODEFunc, s: torch.Tensor, a: torch.Tensor,
               dt: float, method: str = 'euler', n_substeps: int = 2) -> torch.Tensor:
    """Unified integration dispatch.

    NOTE: The model is trained on targets = deltas / (delta_std * dt),
    so model output represents rate * dt (already scaled). Integration
    functions add model output directly without multiplying by dt.

    Args:
        model: ODEFunc dynamics network
        s: current state [batch, STATE_DIM]
        a: action [batch, ACTION_DIM]
        dt: time step (unused for euler, used conceptually for substep)
        method: 'euler', 'rk4', or 'euler_substep'
        n_substeps: number of sub-steps for 'euler_substep'

    Returns:
        s_next: next state [batch, STATE_DIM]
    """
    if method == 'rk4':
        return _integrate_rk4(model, s, a, dt)
    elif method == 'euler_substep':
        return _integrate_euler_substep(model, s, a, dt, n_substeps)
    else:
        return _integrate_euler(model, s, a, dt)


# ---------------------------------------------------------------------------
# Multi-Step Rollout Loss
# ---------------------------------------------------------------------------

def _multi_step_rollout_loss(model: ODEFunc, states: torch.Tensor,
                             actions: torch.Tensor, dt: float,
                             rollout_steps: int, method: str = 'euler',
                             n_substeps: int = 2,
                             limits_norm: Optional[torch.Tensor] = None
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute multi-step rollout MSE loss on a CONTIGUOUS trajectory segment.

    This is the corrected version of V9's multi-step loss. It assumes
    states and actions are temporally ordered within the segment.

    Rollout procedure:
        s_0 = states[0]
        for h in 0..rollout_steps-1:
            s_pred[h] = integrate(s_h, actions[h])
            loss += MSE(s_pred[h], states[h+1])
            s_{h+1} = s_pred[h]  (use prediction, not ground truth -- teacher forcing)

    Args:
        model: ODEFunc dynamics network (normalized space)
        states: [max_H+1, STATE_DIM] -- contiguous normalized states
        actions: [max_H, ACTION_DIM] -- contiguous normalized actions
        dt: time step
        rollout_steps: number of rollout steps H
        method: integration method
        n_substeps: sub-steps for euler_substep
        limits_norm: optional [STATE_DIM] normalized physical limits for survival penalty

    Returns:
        loss: scalar MSE loss averaged over rollout steps
        loss_survival: scalar survival penalty (0 if limits_norm is None)
    """
    n_roll = min(rollout_steps, states.shape[0] - 1, actions.shape[0])
    if n_roll <= 0:
        return torch.tensor(0.0, device=states.device), torch.tensor(0.0, device=states.device)

    s_cur = states[0:1].clone()  # [1, STATE_DIM]
    loss = torch.tensor(0.0, device=states.device)
    loss_surv = torch.tensor(0.0, device=states.device)

    for step in range(n_roll):
        a_step = actions[step:step + 1]  # [1, ACTION_DIM]
        s_cur = _integrate(model, s_cur, a_step, dt, method, n_substeps)
        target = states[step + 1:step + 2]  # [1, STATE_DIM]
        loss = loss + nn.functional.mse_loss(s_cur, target)

        # Survival penalty: penalize predicted states exceeding physical limits
        if limits_norm is not None:
            violation = torch.relu(torch.abs(s_cur) - limits_norm.unsqueeze(0))
            loss_surv = loss_surv + violation.pow(2).mean()

    return loss / n_roll, loss_surv / max(n_roll, 1)


def _multi_step_rollout_trajectory(model: ODEFunc, s0: torch.Tensor,
                                   actions_seq: torch.Tensor, dt: float,
                                   method: str = 'euler',
                                   n_substeps: int = 2) -> torch.Tensor:
    """Roll out the model for H steps from a starting state.

    Used for evaluation: given s0 and a sequence of H actions, produce
    H predicted states via iterative integration.

    Args:
        model: ODEFunc (eval mode)
        s0: [1, STATE_DIM] -- starting state (normalized)
        actions_seq: [H, ACTION_DIM] -- action sequence (normalized)
        dt: time step
        method: integration method
        n_substeps: sub-steps for euler_substep

    Returns:
        predictions: [H, STATE_DIM] -- predicted states at each step
    """
    predictions = []
    s_cur = s0.clone()
    for step in range(actions_seq.shape[0]):
        a_step = actions_seq[step:step + 1]
        s_cur = _integrate(model, s_cur, a_step, dt, method, n_substeps)
        predictions.append(s_cur.squeeze(0))
    return torch.stack(predictions, dim=0)


# ---------------------------------------------------------------------------
# Adaptive Curriculum Scheduler (V12 version)
# ---------------------------------------------------------------------------

def _parse_thresholds(s: str) -> dict:
    """Parse '5:0.02,10:0.04,20:0.08' -> {5: 0.02, 10: 0.04, 20: 0.08}."""
    result = {}
    for pair in s.split(','):
        h, t = pair.strip().split(':')
        result[int(h)] = float(t)
    return result


class _AdaptiveCurriculumSchedulerV12:
    """Adaptive rollout curriculum scheduler for V12.

    Extends the base adaptive curriculum with:
    - Configurable start level (H=1 during warmup)
    - Level 1 has no threshold requirement (always pass warmup)
    - Integration of all curriculum milestones from config
    """

    def __init__(self, thresholds_str: str, ema_decay: float,
                 eval_interval: int, min_epochs_per_level: int,
                 max_epochs_per_level: int, n_epochs: int,
                 lambda_multi_start: float, lambda_multi_end: float,
                 warmup_epochs: int = 5, start_level: int = 1):
        self.thresholds = _parse_thresholds(thresholds_str)
        # Include start_level in the ordered levels
        self.levels = sorted(set([start_level] + list(self.thresholds.keys())))
        self.ema_decay = ema_decay
        self.eval_interval = eval_interval
        self.min_epochs_per_level = min_epochs_per_level
        self.max_epochs_per_level = max_epochs_per_level
        self.n_epochs = n_epochs
        self.lambda_multi_start = lambda_multi_start
        self.lambda_multi_end = lambda_multi_end
        self.warmup_epochs = warmup_epochs

        self.current_level_idx = self.levels.index(start_level)
        self.current_level = start_level
        self.epochs_at_level = 0
        self.ema_errors: dict = {}
        self.log: list = []

    def get_rollout_steps(self, epoch: int) -> int:
        """Get current rollout steps for this epoch.

        During warmup (epoch < warmup_epochs), always returns 1.
        After warmup, returns the current curriculum level.
        """
        if epoch < self.warmup_epochs:
            return 1
        return self.current_level

    def get_lambda_multi(self, epoch: int) -> float:
        """Get current multi-step loss weight (linearly interpolated)."""
        t = min(epoch / max(self.n_epochs, 1), 1.0)
        return self.lambda_multi_start + t * (self.lambda_multi_end - self.lambda_multi_start)

    def update(self, epoch: int, validation_errors: Optional[dict] = None) -> None:
        """Update curriculum state based on validation results.

        Progression rule:
            Advance when validation_nmae[current_level] < threshold[current_level]
            AND epochs_at_level >= min_epochs_per_level.
            Force advance if epochs_at_level >= max_epochs_per_level.
        """
        self.epochs_at_level += 1

        if validation_errors is not None:
            for h, err in validation_errors.items():
                if h not in self.ema_errors:
                    self.ema_errors[h] = err
                else:
                    self.ema_errors[h] = (
                        self.ema_decay * self.ema_errors[h]
                        + (1 - self.ema_decay) * err
                    )

        self.log.append({
            'epoch': epoch,
            'level': self.current_level,
            'epochs_at_level': self.epochs_at_level,
            'ema_errors': dict(self.ema_errors),
        })

        if epoch < self.warmup_epochs:
            return
        if self.current_level_idx >= len(self.levels) - 1:
            return  # Already at max level

        current_h = self.current_level
        if current_h not in self.ema_errors:
            return  # No validation data yet

        ema_err = self.ema_errors[current_h]
        threshold = self.thresholds.get(current_h, float('inf'))

        can_advance = (
            ema_err < threshold
            and self.epochs_at_level >= self.min_epochs_per_level
        )
        force_advance = (
            self.epochs_at_level >= self.max_epochs_per_level
        )

        if can_advance or force_advance:
            self.current_level_idx += 1
            self.current_level = self.levels[self.current_level_idx]
            self.epochs_at_level = 0
            reason = 'error_below_threshold' if can_advance else 'patience_exceeded'
            logger.info(
                "Curriculum advanced to level %d at epoch %d "
                "(reason: %s, ema_err=%.4f, threshold=%.4f)",
                self.current_level, epoch, reason, ema_err, threshold,
            )
            print(
                f"    [Adaptive] Advanced to level {self.current_level} at epoch {epoch} "
                f"(reason: {reason}, ema_err={ema_err:.4f}, threshold={threshold:.4f})"
            )


# ---------------------------------------------------------------------------
# Main Class: NeuralODEV12
# ---------------------------------------------------------------------------

class NeuralODEV12:
    """V12 Neural ODE with all bug fixes and improvements.

    Combines:
    - Sequential segment sampling (fixes Bug 1: shuffled DataLoader)
    - Extended rollout curriculum (fixes Bug 2: training/eval mismatch)
    - Ramp-up multi-step loss weight (fixes Bug 3: single-step dominates)
    - RK4 / Euler-substep integration (fixes Bug 4: truncation error)
    - Contractivity regularization (from V11)
    - Adaptive curriculum scheduling (from adaptive_curriculum.py)
    - Gradient norm monitoring

    All data is normalized internally. The model operates in normalized space
    and denormalizes on output.
    """

    def __init__(self, config: Optional[V12Config] = None):
        self.config = config or V12Config()
        self._model: Optional[ODEFunc] = None
        self._state_std: Optional[np.ndarray] = None
        self._action_std: Optional[float] = None
        self._delta_std: Optional[np.ndarray] = None
        self._dt: float = self.config.dt
        self._training_log: list = []
        self._curriculum_log: list = []

    # ------------------------------------------------------------------
    # Contractivity lambda schedule
    # ------------------------------------------------------------------

    def _get_contractive_lambda(self, epoch: int) -> float:
        """Compute effective contractivity lambda with warmup and linear ramp.

        Schedule:
            epoch < warmup: lambda = 0 (let base dynamics learn first)
            warmup <= epoch < warmup + ramp: linear ramp to full strength
            epoch >= warmup + ramp: lambda = lambda_contractive
        """
        if not self.config.use_contractivity:
            return 0.0
        cfg = self.config
        if epoch < cfg.contractive_warmup_epochs:
            return 0.0
        ramp_progress = (epoch - cfg.contractive_warmup_epochs) / max(cfg.contractive_ramp_epochs, 1)
        if ramp_progress >= 1.0:
            return cfg.lambda_contractive
        return cfg.lambda_contractive * ramp_progress

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _build_dataset(self, states: np.ndarray, actions: np.ndarray,
                       deltas: np.ndarray, max_h: int,
                       episode_boundaries=None) -> SequentialSegmentDataset:
        """Build a SequentialSegmentDataset from raw normalized data.

        Args:
            states: [N, STATE_DIM] normalized states
            actions: [N, 1] or [N] normalized actions
            deltas: [N, STATE_DIM] normalized ds/dt targets
            max_h: maximum rollout horizon for segment extraction
            episode_boundaries: optional list of (start, end) tuples for episodes

        Returns:
            SequentialSegmentDataset ready for DataLoader
        """
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        return SequentialSegmentDataset(states, actions, deltas, max_h,
                                        episode_boundaries=episode_boundaries)

    # ------------------------------------------------------------------
    # Validation error computation
    # ------------------------------------------------------------------

    def _compute_validation_errors(self, val_s: torch.Tensor, val_a: torch.Tensor,
                                   val_y: torch.Tensor,
                                   horizons: List[int]) -> dict:
        """Compute validation NMAE at multiple rollout horizons.

        Uses the same integration method as training, ensuring consistency
        between training loss and evaluation metrics.

        Args:
            val_s: [N, STATE_DIM] normalized validation states
            val_a: [N, ACTION_DIM] normalized validation actions
            val_y: [N, STATE_DIM] normalized validation ds/dt targets
            horizons: list of rollout horizons to evaluate

        Returns:
            dict mapping horizon (int) -> NMAE (float)
        """
        self._model.eval()
        method = self.config.integration_method
        n_sub = self.config.sub_steps

        with torch.no_grad():
            # Single-step NMAE
            # model outputs delta/state_std, val_y = delta/state_std
            # NMAE = |delta_pred - delta_target| / state_std = |model_out - val_y|
            n_single = min(256, len(val_s))
            pred_single = self._model(val_s[:n_single], val_a[:n_single])
            single_err = torch.abs(pred_single - val_y[:n_single]).mean(dim=0)
            single_nmae = single_err.mean().item()

            results = {1: single_nmae}

            for h in horizons:
                if h <= 1:
                    continue

                n_candidates = len(val_s) - h
                if n_candidates <= 0:
                    results[h] = float('nan')
                    continue

                n_roll = min(n_candidates, 128)
                err_total = 0.0
                valid = 0

                for i in range(n_roll):
                    s0 = val_s[i:i + 1]  # [1, STATE_DIM] normalized
                    actions_seq = val_a[i:i + h]  # [h, ACTION_DIM] normalized
                    target_seq = val_s[i + 1:i + 1 + h]  # [h, STATE_DIM] normalized

                    if len(target_seq) < h:
                        continue

                    preds = _multi_step_rollout_trajectory(
                        self._model, s0, actions_seq, self._dt, method, n_sub
                    )
                    # preds, target_seq: normalized state = s_phys / state_std
                    # |preds - target_seq| = |s_pred - s_target| / state_std = NMAE per state
                    err = torch.abs(preds - target_seq).mean(dim=0)
                    err_total += err.mean().item()
                    valid += 1

                results[h] = err_total / max(valid, 1)

        self._model.train()
        return results

    # ------------------------------------------------------------------
    # Single-step prediction
    # ------------------------------------------------------------------

    def predict(self, s: np.ndarray, tau: float) -> np.ndarray:
        """Predict next state from current state and steering input.

        Uses the same integration method as training for consistency.

        Model output = delta / state_std (normalized state change).
        Integration: s_next_norm = s_norm + model_output = (s + delta) / state_std.
        Denormalization: s_next_physical = s_next_norm * state_std.

        Args:
            s: [STATE_DIM] current state (physical units)
            tau: scalar steering input (physical units)

        Returns:
            next_state: [STATE_DIM] predicted next state (physical units)
        """
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0)

        with torch.no_grad():
            s_next_norm = _integrate(
                self._model, s_norm, a_norm, self._dt,
                self.config.integration_method, self.config.sub_steps,
            ).numpy()[0]

        # s_next_norm = (s + delta) / state_std
        # s_next_physical = s_next_norm * state_std
        return s_next_norm * self._state_std

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Batch prediction with the same integration method as training.

        Model output = delta / state_std. Integration gives s_next_norm.
        Denormalize: s_next_physical = s_next_norm * state_std.

        Args:
            states: [N, STATE_DIM] current states (physical units)
            actions: [N] or [N, 1] steering inputs (physical units)

        Returns:
            next_states: [N, STATE_DIM] predicted next states (physical units)
        """
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)

        s_norm = torch.FloatTensor(states / self._state_std)
        a_norm = torch.FloatTensor(actions / self._action_std)

        method = self.config.integration_method
        n_sub = self.config.sub_steps

        with torch.no_grad():
            if method == 'euler':
                # Fast path: s_next_norm = s_norm + model(s_norm, a_norm)
                model_output = self._model(s_norm, a_norm).numpy()
                s_next_norm = s_norm.numpy() + model_output
            else:
                # For RK4 / euler_substep, integrate each sample
                n = s_norm.shape[0]
                s_next_list = []
                for i in range(n):
                    s_i = s_norm[i:i + 1]
                    a_i = a_norm[i:i + 1]
                    s_next = _integrate(self._model, s_i, a_i, self._dt, method, n_sub)
                    s_next_list.append(s_next.numpy()[0])
                s_next_norm = np.stack(s_next_list, axis=0)

        # Denormalize: s_next_physical = s_next_norm * state_std
        return s_next_norm * self._state_std

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, states: np.ndarray, actions: np.ndarray, deltas: np.ndarray,
              state_std: np.ndarray, action_std: float, delta_std: np.ndarray,
              val_states: Optional[np.ndarray] = None,
              val_actions: Optional[np.ndarray] = None,
              val_deltas: Optional[np.ndarray] = None,
              episode_boundaries: Optional[List[Tuple[int, int]]] = None) -> None:
        """Train the V12 Neural ODE with all bug fixes.

        Args:
            states: [N, STATE_DIM] raw training states (physical units)
            actions: [N] or [N, 1] raw training actions (physical units)
            deltas: [N, STATE_DIM] raw training state deltas (physical units)
            state_std: [STATE_DIM] state standard deviations for normalization
            action_std: scalar action standard deviation
            delta_std: [STATE_DIM] delta standard deviations for normalization
            val_states: optional validation states (physical units)
            val_actions: optional validation actions (physical units)
            val_deltas: optional validation deltas (physical units)
        """
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        cfg = self.config

        # Store and sanitize normalization parameters
        self._state_std = state_std.copy()
        self._action_std = action_std
        self._delta_std = delta_std.copy()
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0
        # Precompute std ratio for consistency loss: theta_dot/state_std[4] -> theta/state_std[3]
        self._std_ratio = float(state_std[4] / state_std[3]) if state_std[3] > 1e-10 else 1.0

        # Precompute normalized physical limits for survival-aware loss
        # limits_norm[i] = physical_limit[i] / state_std[i] * survival_limit_scale
        if cfg.use_survival_loss:
            from .config_v9 import PHYSICAL_LIMITS
            limit_keys = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
            limits_phys = np.array([PHYSICAL_LIMITS[k] for k in limit_keys])
            self._limits_norm = torch.FloatTensor(
                limits_phys / self._state_std * cfg.survival_limit_scale
            )
        else:
            self._limits_norm = None

        # Normalize training data
        # IMPORTANT: targets are deltas / state_std (NOT delta_std, NOT dt).
        # This ensures model output = delta / state_std, which integrates
        # correctly: s_next_norm = s_norm + model_output = (s + delta) / state_std.
        # Integration adds model output directly to normalized state.
        train_s = states / self._state_std
        train_a = actions.reshape(-1, 1) / self._action_std
        train_dsdot = deltas / self._state_std

        # Build sequential segment dataset
        # Use initial max_segment_len for dataset creation
        dataset = self._build_dataset(train_s, train_a, train_dsdot, cfg.max_segment_len,
                                      episode_boundaries=episode_boundaries)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,  # Safe: shuffles SEGMENTS, not timesteps within segments
            drop_last=False,
        )

        # Prepare validation data
        has_val = val_states is not None and len(val_states) > 0
        if has_val:
            val_s = torch.FloatTensor(val_states / self._state_std)
            val_a = torch.FloatTensor(val_actions.reshape(-1, 1) / self._action_std)
            val_y = torch.FloatTensor(val_deltas / self._state_std)
        else:
            # Use last 10% of training data as validation
            n_val = max(int(len(train_s) * 0.1), 64)
            val_s = torch.FloatTensor(train_s[-n_val:])
            val_a = torch.FloatTensor(train_a[-n_val:])
            val_y = torch.FloatTensor(train_dsdot[-n_val:])
            has_val = True

        # Initialize model
        self._model = ODEFunc(cfg.hidden, cfg.depth, cfg.activation)

        # Optimizer and LR scheduler
        opt = torch.optim.Adam(
            self._model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.n_epochs
        )

        # Parse curriculum milestones for validation horizons
        curriculum_milestones = [int(x) for x in cfg.rollout_curriculum.split(',')]

        # Initialize adaptive curriculum scheduler
        curriculum = _AdaptiveCurriculumSchedulerV12(
            thresholds_str=cfg.adaptive_thresholds,
            ema_decay=cfg.ema_decay,
            eval_interval=cfg.eval_interval,
            min_epochs_per_level=cfg.min_epochs_per_level,
            max_epochs_per_level=cfg.max_epochs_per_level,
            n_epochs=cfg.n_epochs,
            lambda_multi_start=cfg.lambda_multi_start,
            lambda_multi_end=cfg.lambda_multi_end,
            warmup_epochs=5,
            start_level=1,
        )

        # Print training configuration
        logger.info(
            "V12 Training: epochs=%d, integration=%s, sub_steps=%d, "
            "max_segment_len=%d, contractivity=%s, adaptive_curriculum=%s",
            cfg.n_epochs, cfg.integration_method, cfg.sub_steps,
            cfg.max_segment_len, cfg.use_contractivity, cfg.use_adaptive_curriculum,
        )
        print(
            f"    V12 config: epochs={cfg.n_epochs}, "
            f"integration={cfg.integration_method}(sub_steps={cfg.sub_steps}), "
            f"max_segment_len={cfg.max_segment_len}, "
            f"contractivity={'on' if cfg.use_contractivity else 'off'}, "
            f"curriculum={cfg.rollout_curriculum}"
        )

        # ---- Training loop ----
        self._model.train()
        for epoch in range(cfg.n_epochs):
            epoch_loss = 0.0
            epoch_single = 0.0
            epoch_multi = 0.0
            epoch_contractive = 0.0
            epoch_survival = 0.0
            epoch_grad_norm = 0.0
            n_batches = 0

            # Get current rollout steps and lambda_multi from adaptive scheduler
            rollout_steps = curriculum.get_rollout_steps(epoch)
            lambda_multi = curriculum.get_lambda_multi(epoch)
            eff_lambda_con = self._get_contractive_lambda(epoch)

            # Update dataset's active horizon if rollout_steps changed
            dataset.set_max_horizon(max(rollout_steps, 2))
            # Rebuild loader with potentially different segment set
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                drop_last=False,
            )

            # Periodic validation for adaptive curriculum
            validation_errors = {}
            if has_val and (epoch + 1) % cfg.eval_interval == 0:
                validation_errors = self._compute_validation_errors(
                    val_s, val_a, val_y, curriculum_milestones
                )
                curriculum.update(epoch, validation_errors)
            else:
                curriculum.update(epoch, None)

            # ---- Batch loop ----
            for states_batch, actions_batch, deltas_batch in loader:
                # states_batch: [B, max_H+1, STATE_DIM]
                # actions_batch: [B, max_H, ACTION_DIM]
                # deltas_batch: [B, max_H, STATE_DIM]
                B = states_batch.shape[0]

                # --- Single-step loss ---
                # Flatten for point-wise prediction: use first H states and H actions
                H_single = states_batch.shape[1] - 1  # number of steps in segment
                s_flat = states_batch[:, :H_single].reshape(-1, STATE_DIM)  # [B*H, 7]
                a_flat = actions_batch.reshape(-1, ACTION_DIM)  # [B*H, 1]
                y_flat = deltas_batch.reshape(-1, STATE_DIM)  # [B*H, 7]

                pred = self._model(s_flat, a_flat)  # [B*H, 7]
                loss_single = nn.functional.mse_loss(pred, y_flat)

                # --- Multi-step rollout loss ---
                loss_multi = torch.tensor(0.0, device=s_flat.device)
                loss_survival = torch.tensor(0.0, device=s_flat.device)
                if rollout_steps > 1 and H_single >= rollout_steps:
                    # Roll out from each segment's starting state
                    n_roll_segs = min(B, 32)  # limit for memory
                    loss_multi = torch.tensor(0.0, device=s_flat.device)
                    loss_surv_rollout = torch.tensor(0.0, device=s_flat.device)
                    limits = self._limits_norm.to(s_flat.device) if self._limits_norm is not None else None
                    for seg_i in range(n_roll_segs):
                        seg_states = states_batch[seg_i]  # [max_H+1, 7]
                        seg_actions = actions_batch[seg_i]  # [max_H, 1]
                        seg_mse, seg_surv = _multi_step_rollout_loss(
                            self._model, seg_states, seg_actions, self._dt,
                            rollout_steps, cfg.integration_method, cfg.sub_steps,
                            limits_norm=limits,
                        )
                        loss_multi = loss_multi + seg_mse
                        loss_surv_rollout = loss_surv_rollout + seg_surv
                    loss_multi = loss_multi / n_roll_segs
                    loss_survival = loss_surv_rollout / n_roll_segs

                # --- Consistency loss ---
                # Model output = delta/state_std. Enforce kinematic constraint:
                # theta_next = theta + theta_dot * dt (in physical units).
                # In normalized space: theta_next/norm[3] = theta/norm[3] + (theta_dot/norm[4]) * dt * (norm[4]/norm[3])
                loss_consistency = torch.tensor(0.0, device=s_flat.device)
                if cfg.lambda_consistency > 0 and hasattr(self, '_std_ratio'):
                    s_next_pred = s_flat + pred  # (s+delta)/state_std
                    theta_pred = s_next_pred[:, 3]
                    theta_gt = s_flat[:, 3] + s_flat[:, 4] * self._dt * self._std_ratio
                    loss_consistency = nn.functional.mse_loss(theta_pred, theta_gt)

                # --- Jacobian Frobenius norm regularization ---
                loss_jacobian = torch.tensor(0.0, device=s_flat.device)
                if cfg.lambda_jacobian > 0:
                    n_jac = min(32, len(s_flat))
                    s_jac = s_flat[:n_jac].requires_grad_(True)
                    a_jac = a_flat[:n_jac].requires_grad_(True)
                    dsdt_j = self._model(s_jac, a_jac)
                    jac_norm = torch.tensor(0.0, device=s_flat.device)
                    for i in range(STATE_DIM):
                        grad = torch.autograd.grad(
                            dsdt_j[:, i].sum(), s_jac,
                            create_graph=True, retain_graph=True,
                        )[0]
                        jac_norm = jac_norm + grad.pow(2).sum()
                    loss_jacobian = jac_norm / (STATE_DIM * n_jac)

                # --- Contractivity loss ---
                loss_contractive = torch.tensor(0.0, device=s_flat.device)
                if eff_lambda_con > 0:
                    n_con = min(32, len(s_flat))
                    s_con = s_flat[:n_con].requires_grad_(True)
                    a_con = a_flat[:n_con].requires_grad_(True)
                    loss_contractive, _ = _contractivity_loss(
                        self._model, s_con, a_con
                    )

                # --- Total loss ---
                loss = (
                    loss_single
                    + lambda_multi * loss_multi
                    + cfg.lambda_consistency * loss_consistency
                    + cfg.lambda_jacobian * loss_jacobian
                    + eff_lambda_con * loss_contractive
                    + cfg.lambda_survival * loss_survival
                )

                opt.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), 1.0
                )
                opt.step()

                epoch_loss += loss.item()
                epoch_single += loss_single.item()
                epoch_multi += loss_multi.item()
                epoch_contractive += loss_contractive.item()
                epoch_survival += loss_survival.item()
                epoch_grad_norm += grad_norm.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_single = epoch_single / max(n_batches, 1)
            avg_multi = epoch_multi / max(n_batches, 1)
            avg_contractive = epoch_contractive / max(n_batches, 1)
            avg_survival = epoch_survival / max(n_batches, 1)
            avg_grad_norm = epoch_grad_norm / max(n_batches, 1)

            self._training_log.append({
                'epoch': epoch,
                'loss': avg_loss,
                'loss_single': avg_single,
                'loss_multi': avg_multi,
                'loss_contractive': avg_contractive,
                'loss_survival': avg_survival,
                'eff_lambda_con': eff_lambda_con,
                'rollout_steps': rollout_steps,
                'lambda_multi': lambda_multi,
                'grad_norm': avg_grad_norm,
                'level': curriculum.current_level,
                'epochs_at_level': curriculum.epochs_at_level,
                'val_errors': validation_errors,
                'lr': scheduler.get_last_lr()[0],
            })

            # Periodic logging
            if epoch % 50 == 0 or epoch == cfg.n_epochs - 1:
                val_str = ''
                if validation_errors:
                    val_str = ', '.join(
                        f'h={h}:{e:.4f}'
                        for h, e in sorted(validation_errors.items())
                        if not (isinstance(e, float) and np.isnan(e))
                    )
                print(
                    f"    Epoch {epoch}: loss={avg_loss:.6f} "
                    f"(single={avg_single:.6f}, multi={avg_multi:.6f}, "
                    f"con={avg_contractive:.6f}, sur={avg_survival:.6f}), "
                    f"rollout={rollout_steps}, level={curriculum.current_level}, "
                    f"lambda_multi={lambda_multi:.3f}, "
                    f"lambda_con={eff_lambda_con:.4f}, "
                    f"lambda_sur={cfg.lambda_survival if cfg.use_survival_loss else 0:.4f}, "
                    f"grad_norm={avg_grad_norm:.2f}, "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                    f"{', val: ' + val_str if val_str else ''}"
                )

        self._model.eval()
        self._curriculum_log = curriculum.log

        print(f"    V12 training complete. Final level: {curriculum.current_level}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model checkpoint to disk.

        Args:
            path: file path for the checkpoint
        """
        checkpoint = {
            'config': asdict(self.config),
            'state_std': self._state_std,
            'action_std': self._action_std,
            'delta_std': self._delta_std,
            'model_state': self._model.state_dict() if self._model else None,
            'training_log': self._training_log,
            'curriculum_log': self._curriculum_log,
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(checkpoint, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Load model checkpoint from disk.

        Args:
            path: file path of the checkpoint
        """
        checkpoint = torch.load(path, weights_only=False)
        self.config = V12Config(**checkpoint['config'])
        self._state_std = checkpoint['state_std']
        self._action_std = checkpoint['action_std']
        self._delta_std = checkpoint['delta_std']
        self._training_log = checkpoint.get('training_log', [])
        self._curriculum_log = checkpoint.get('curriculum_log', [])

        self._model = ODEFunc(
            self.config.hidden, self.config.depth, self.config.activation
        )
        self._model.load_state_dict(checkpoint['model_state'])
        self._model.eval()
        logger.info("Model loaded from %s", path)

    def name(self) -> str:
        """Return model identifier string."""
        return 'neural_ode_v12'
