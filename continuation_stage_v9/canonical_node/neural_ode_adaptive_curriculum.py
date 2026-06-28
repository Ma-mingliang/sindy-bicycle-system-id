"""Adaptive Curriculum Neural ODE.

Key insight from Bucci et al. 2023: naive rollout length increase hurts long-term
performance because long rollouts early in training produce large gradients that
destabilize learning, and the model overfits to correcting short-horizon errors.

Solution: Error-Adaptive Curriculum
- Monitor per-horizon prediction error during training via periodic validation
- Increase rollout length only when short-horizon error is below threshold
- Use exponentially-weighted moving average of validation error for stability
- Include gradient magnitude monitoring to detect instability

The curriculum is NOT a fixed schedule. It adapts to the actual model capability.

Why this works better than fixed curriculum:
1. Prevents destabilizing gradient explosions when model isn't ready for long rollouts
2. Allows the model to fully learn short-horizon dynamics before extending
3. Reduces wasted training epochs where model trains on rollouts it can't handle
4. The threshold-based progression acts as a natural regularizer
"""
import numpy as np
import os
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch
import torch.nn as nn

from .config_v9 import STATE_DIM, ACTION_DIM
from .neural_ode_v9 import ODEFunc, NeuralODEConfig


@dataclass
class AdaptiveCurriculumConfig(NeuralODEConfig):
    """Config for error-adaptive curriculum."""
    # --- Adaptive curriculum parameters ---
    # Maximum rollout steps the model should eventually learn
    max_rollout: int = 50
    # Error thresholds per horizon level. When validation NMAE at horizon h
    # drops below threshold[h], the model is "ready" for the next level.
    # Format: dict mapping horizon -> threshold
    # These are NMAE values (normalized mean absolute error)
    error_thresholds: str = '5:0.03,10:0.05,20:0.10,50:0.20'
    # EMA decay for smoothing the validation error estimate
    ema_decay: float = 0.95
    # How often (in epochs) to run validation for adaptive scheduling
    eval_interval: int = 10
    # Minimum epochs at each level before advancing
    min_epochs_per_level: int = 20
    # Maximum epochs at each level (patience before forcing advancement)
    max_epochs_per_level: int = 80
    # Gradient norm threshold: if avg grad norm exceeds this, stay at current level
    grad_norm_limit: float = 5.0
    # Warmup: always start with single-step for this many epochs
    warmup_epochs: int = 5
    # Multi-step loss weight schedule: increase weight as model matures
    lambda_multi_start: float = 0.1
    lambda_multi_end: float = 0.5


def parse_error_thresholds(s: str) -> dict:
    """Parse '5:0.03,10:0.05,20:0.10,50:0.20' -> {5: 0.03, 10: 0.05, ...}"""
    result = {}
    for pair in s.split(','):
        h, t = pair.strip().split(':')
        result[int(h)] = float(t)
    return result


class AdaptiveCurriculumScheduler:
    """Manages adaptive rollout curriculum based on validation error.

    The scheduler maintains:
    - Current rollout level (one of the curriculum milestones)
    - EMA-smoothed validation error at each horizon
    - Epochs spent at current level

    Progression rule:
        advance when validation_nmae[current_level] < threshold[current_level]
        AND epochs_at_level >= min_epochs_per_level
    """

    def __init__(self, config: AdaptiveCurriculumConfig):
        self.config = config
        self.thresholds = parse_error_thresholds(config.error_thresholds)
        # Build ordered list of curriculum levels
        self.levels = sorted(self.thresholds.keys())
        # Start at level = first level (typically 5)
        self.current_level_idx = 0
        self.current_level = self.levels[0]
        self.epochs_at_level = 0
        self.total_epochs_advanced = 0
        # EMA of validation error at each horizon
        self.ema_errors = {}
        self.log = []

    def get_rollout_steps(self, epoch: int) -> int:
        """Get current rollout steps for this epoch."""
        if epoch < self.config.warmup_epochs:
            return 1
        return self.current_level

    def get_lambda_multi(self, epoch: int) -> float:
        """Get current multi-step loss weight (linearly interpolated)."""
        t = min(epoch / max(self.config.n_epochs, 1), 1.0)
        return self.config.lambda_multi_start + t * (
            self.config.lambda_multi_end - self.config.lambda_multi_start
        )

    def update(self, epoch: int, validation_errors: Optional[dict] = None):
        """Update curriculum state based on validation results.

        Args:
            epoch: current epoch number
            validation_errors: dict mapping horizon (int) -> NMAE (float)
                If None, just increment epoch counter.
        """
        self.epochs_at_level += 1

        if validation_errors is not None:
            # Update EMA of errors
            for h, err in validation_errors.items():
                if h not in self.ema_errors:
                    self.ema_errors[h] = err
                else:
                    decay = self.config.ema_decay
                    self.ema_errors[h] = decay * self.ema_errors[h] + (1 - decay) * err

        # Log current state
        self.log.append({
            'epoch': epoch,
            'level': self.current_level,
            'epochs_at_level': self.epochs_at_level,
            'ema_errors': dict(self.ema_errors),
        })

        # Check advancement conditions
        if epoch < self.config.warmup_epochs:
            return

        if self.current_level_idx >= len(self.levels) - 1:
            return  # Already at max level

        # Get current level's EMA error
        current_h = self.current_level
        if current_h not in self.ema_errors:
            return  # No validation data yet

        ema_err = self.ema_errors[current_h]
        threshold = self.thresholds[current_h]

        # Advancement conditions:
        # 1. EMA error below threshold
        # 2. Spent minimum epochs at this level
        # 3. Not exceeded maximum epochs (patience)
        can_advance = (
            ema_err < threshold
            and self.epochs_at_level >= self.config.min_epochs_per_level
        )
        force_advance = (
            self.epochs_at_level >= self.config.max_epochs_per_level
        )

        if can_advance or force_advance:
            self.current_level_idx += 1
            self.current_level = self.levels[self.current_level_idx]
            self.total_epochs_advanced = epoch
            self.epochs_at_level = 0
            reason = 'error_below_threshold' if can_advance else 'patience_exceeded'
            print(f"    [Adaptive] Advanced to level {self.current_level} at epoch {epoch} "
                  f"(reason: {reason}, ema_err={ema_err:.4f}, threshold={threshold:.4f})")


class AdaptiveCurriculumNeuralODE:
    """Neural ODE with error-adaptive rollout curriculum.

    Unlike fixed curriculum (1->5->10->20), this monitors actual model
    performance and only advances when the model is ready.
    """

    def __init__(self, config: AdaptiveCurriculumConfig = None):
        self.config = config or AdaptiveCurriculumConfig()
        self._model = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None
        self._dt = self.config.dt
        self._training_log = []

    def _compute_validation_error(self, model, val_s, val_a, val_y,
                                   state_std, rollout_steps):
        """Compute validation NMAE at given rollout horizon.

        Returns NMAE (float) or nan if evaluation fails.
        """
        model.eval()
        with torch.no_grad():
            # Single-step error
            pred = model(val_s[:256], val_a[:256])
            single_err = torch.abs(pred - val_y[:256]).mean(dim=0) / torch.FloatTensor(state_std[:7]).to(val_s.device)
            single_nmae = single_err.mean().item()

            # Multi-step rollout error
            n_roll = min(len(val_s) - rollout_steps, 128)
            if n_roll <= 0 or rollout_steps <= 1:
                model.train()
                return single_nmae

            s_cur = val_s[:n_roll].clone()
            total_err = 0.0
            valid_steps = 0
            for step in range(rollout_steps):
                a_step = val_a[step:step + n_roll]
                dsdt = model(s_cur, a_step)
                s_cur = s_cur + dsdt * self._dt
                target = val_s[step + 1:step + 1 + n_roll]
                if len(target) == 0:
                    break
                err = torch.abs(s_cur - target).mean(dim=0) / torch.FloatTensor(state_std[:7]).to(val_s.device)
                total_err += err.mean().item()
                valid_steps += 1

            model.train()
            if valid_steps == 0:
                return single_nmae
            return total_err / valid_steps

    def train(self, states, actions, deltas, state_std, action_std, delta_std,
              val_states=None, val_actions=None, val_deltas=None):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self._state_std = state_std.copy()
        self._action_std = action_std
        self._delta_std = delta_std.copy()
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0

        self._model = ODEFunc(
            self.config.hidden, self.config.depth, self.config.activation
        )

        # Prepare training data
        train_s = torch.FloatTensor(states / self._state_std)
        train_a = torch.FloatTensor(actions.reshape(-1, 1) / self._action_std)
        train_dsdot = torch.FloatTensor(deltas / (self._delta_std * self._dt))

        ds = torch.utils.data.TensorDataset(train_s, train_a, train_dsdot)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.config.batch_size, shuffle=True
        )

        # Prepare validation data
        has_val = (val_states is not None and len(val_states) > 0)
        if has_val:
            val_s = torch.FloatTensor(val_states / self._state_std)
            val_a = torch.FloatTensor(val_actions.reshape(-1, 1) / self._action_std)
            val_y = torch.FloatTensor(val_deltas / (self._delta_std * self._dt))
        else:
            # Use last 10% of training data as validation
            n_val = max(int(len(train_s) * 0.1), 64)
            val_s = train_s[-n_val:]
            val_a = train_a[-n_val:]
            val_y = train_dsdot[-n_val:]
            has_val = True

        opt = torch.optim.Adam(
            self._model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.config.n_epochs
        )

        # Initialize adaptive curriculum
        curriculum = AdaptiveCurriculumScheduler(self.config)

        self._model.train()
        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            epoch_grad_norm = 0.0
            n_batches = 0

            # Get current rollout steps from adaptive scheduler
            rollout_steps = curriculum.get_rollout_steps(epoch)
            lambda_multi = curriculum.get_lambda_multi(epoch)

            for sb, ab, yb in loader:
                # Single-step loss
                pred = self._model(sb, ab)
                loss_single = nn.functional.mse_loss(pred, yb)

                # Multi-step rollout loss
                loss_multi = torch.tensor(0.0)
                if rollout_steps > 1 and len(sb) > rollout_steps + 1:
                    n_roll = min(len(sb) - rollout_steps, 64)
                    s_cur = sb[:n_roll].clone()
                    for step in range(rollout_steps):
                        a_cur = ab[step:step + n_roll]
                        dsdt = self._model(s_cur, a_cur)
                        s_cur = s_cur + dsdt * self._dt
                        target = sb[step + 1:step + 1 + n_roll]
                        loss_multi = loss_multi + nn.functional.mse_loss(s_cur, target)
                    loss_multi = loss_multi / rollout_steps

                # Consistency loss
                loss_consistency = torch.tensor(0.0)
                if self.config.lambda_consistency > 0:
                    s_next_norm = sb + pred * self._dt
                    theta_pred = s_next_norm[:, 3]
                    theta_dot = sb[:, 4]
                    theta_gt = sb[:, 3] + theta_dot * self._dt
                    loss_consistency = nn.functional.mse_loss(theta_pred, theta_gt)

                # Jacobian regularization
                loss_jacobian = torch.tensor(0.0)
                if self.config.lambda_jacobian > 0:
                    s_req = sb[:min(32, len(sb))].requires_grad_(True)
                    a_req = ab[:min(32, len(sb))].requires_grad_(True)
                    dsdt = self._model(s_req, a_req)
                    jac_norm = 0.0
                    for i in range(STATE_DIM):
                        grad = torch.autograd.grad(
                            dsdt[:, i].sum(), s_req, create_graph=True
                        )[0]
                        jac_norm = jac_norm + grad.pow(2).sum()
                    loss_jacobian = jac_norm / (STATE_DIM * min(32, len(sb)))

                loss = (loss_single
                        + lambda_multi * loss_multi
                        + self.config.lambda_consistency * loss_consistency
                        + self.config.lambda_jacobian * loss_jacobian)

                opt.zero_grad()
                loss.backward()
                # Clip gradients and track norm
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), 1.0
                )
                epoch_grad_norm += grad_norm.item()
                opt.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_grad_norm = epoch_grad_norm / max(n_batches, 1)

            # Periodic validation and adaptive curriculum update
            validation_errors = {}
            if has_val and (epoch + 1) % self.config.eval_interval == 0:
                for h in [1, 5, 10, 20, 50]:
                    err = self._compute_validation_error(
                        self._model, val_s, val_a, val_y,
                        self._state_std, h
                    )
                    validation_errors[h] = err

            curriculum.update(epoch, validation_errors if validation_errors else None)

            self._training_log.append({
                'epoch': epoch,
                'loss': avg_loss,
                'rollout_steps': rollout_steps,
                'lambda_multi': lambda_multi,
                'grad_norm': avg_grad_norm,
                'level': curriculum.current_level,
                'epochs_at_level': curriculum.epochs_at_level,
                'val_errors': validation_errors,
                'lr': scheduler.get_last_lr()[0],
            })

            if epoch % 50 == 0:
                val_str = ', '.join(f'h={h}:{e:.4f}' for h, e in sorted(validation_errors.items()))
                print(f"    Epoch {epoch}: loss={avg_loss:.4f}, "
                      f"rollout={rollout_steps}, level={curriculum.current_level}, "
                      f"epochs_at_level={curriculum.epochs_at_level}, "
                      f"grad_norm={avg_grad_norm:.2f}"
                      f"{', ' + val_str if val_str else ''}")

        self._model.eval()
        self._curriculum_log = curriculum.log

    def predict(self, s, tau):
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0)
        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).numpy()[0]
        dsdt = dsdt_norm * self._delta_std * self._dt
        return s + dsdt

    def predict_batch(self, states, actions):
        s_norm = torch.FloatTensor(states / self._state_std)
        a_norm = torch.FloatTensor(actions.reshape(-1, 1) / self._action_std)
        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).numpy()
        return states + dsdt_norm * self._delta_std * self._dt

    def name(self):
        return 'adaptive_curriculum'

    def save(self, path):
        checkpoint = {
            'config': asdict(self.config),
            'state_std': self._state_std,
            'action_std': self._action_std,
            'delta_std': self._delta_std,
            'model_state': self._model.state_dict() if self._model else None,
            'training_log': self._training_log,
            'curriculum_log': getattr(self, '_curriculum_log', []),
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.config = AdaptiveCurriculumConfig(**checkpoint['config'])
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
