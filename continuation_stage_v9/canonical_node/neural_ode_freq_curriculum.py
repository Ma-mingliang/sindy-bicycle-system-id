"""Frequency-Domain Curriculum Neural ODE.

Key insight: bicycle dynamics have distinct frequency scales:
- SLOW (low-freq): v (speed), theta (roll angle) -- change over 0.5-2s
- MEDIUM: e_y (lateral error), e_psi (heading error), delta (steering angle) -- 0.1-0.5s
- FAST (high-freq): theta_dot (roll rate), delta_dot (steering rate) -- 0.01-0.1s

The frequency-domain curriculum trains on low-frequency components first,
then progressively adds high-frequency. This is better than rollout-length
curriculum because:
1. Low-frequency states have larger spatial scales and are easier to predict
2. High-frequency states are sensitive to accumulated integration error
3. Learning slow dynamics first provides a stable backbone for fast dynamics
4. The frequency decomposition is physics-grounded, not arbitrary

Implementation:
- Separate the loss into per-state contributions
- Weight low-frequency states more early, high-frequency states more later
- Use a smooth transition schedule for the frequency weighting
- Combine with a mild rollout curriculum for additional benefit

Why this works better than fixed rollout curriculum:
1. Directly addresses the root cause: fast states dominate the gradient signal
2. Prevents the model from "cheating" by learning only easy states
3. The frequency decomposition is interpretable and tunable
4. Can be combined with any rollout length (orthogonal improvement)
"""
import numpy as np
import os
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn

from .config_v9 import STATE_DIM, ACTION_DIM, STATE_NAMES_7D
from .neural_ode_v9 import ODEFunc, NeuralODEConfig


# --- Frequency bands for 7D bicycle state ---
# Indices: [e_y, e_psi, v, theta, theta_dot, delta, delta_dot]
# Physical time constants (approximate, seconds):
#   v: 1-5s (slow speed changes)
#   theta: 0.5-2s (roll dynamics)
#   e_y: 0.3-1s (lateral position)
#   e_psi: 0.2-0.5s (heading angle)
#   delta: 0.1-0.5s (steering angle)
#   theta_dot: 0.05-0.2s (fast roll rate)
#   delta_dot: 0.02-0.1s (fast steering rate)

FREQ_BANDS = {
    'slow': [2, 3],        # v, theta
    'medium': [0, 1, 5],   # e_y, e_psi, delta
    'fast': [4, 6],         # theta_dot, delta_dot
}

# Default frequency weights: which bands to emphasize at each training phase
# Format: (start_weight_slow, start_weight_medium, start_weight_fast)
# Weight is multiplied with per-state loss contribution


@dataclass
class FreqCurriculumConfig(NeuralODEConfig):
    """Config for frequency-domain curriculum."""
    # --- Frequency curriculum parameters ---
    # Frequency band weights schedule as a string: 'slow:1.0,med:0.5,fast:0.1->slow:0.5,med:0.8,fast:1.0'
    # Arrow separates start and end weights
    freq_schedule: str = 'slow:2.0,med:1.0,fast:0.3->slow:0.5,med:0.8,fast:1.5'
    # Minimum per-state weight (prevents any state from being completely ignored)
    min_state_weight: float = 0.1
    # How many epochs to fully transition from start to end weights
    freq_transition_epochs: int = 150
    # Whether to also use mild rollout curriculum (recommended: True)
    use_mild_rollout: bool = True
    # Mild rollout: 1->3->5->10 (conservative, complements frequency curriculum)
    mild_rollout_curriculum: str = '1,3,5,10'
    # Rollout curriculum weight (how much to weight multi-step loss)
    lambda_multi_freq: float = 0.2


def parse_freq_schedule(s: str) -> tuple:
    """Parse 'slow:2.0,med:1.0,fast:0.3->slow:0.5,med:0.8,fast:1.5'

    Returns:
        start_weights: dict mapping band_name -> weight
        end_weights: dict mapping band_name -> weight
    """
    start_str, end_str = s.split('->')

    def _parse_band_weights(ss):
        result = {}
        for part in ss.split(','):
            name, val = part.strip().split(':')
            result[name.strip()] = float(val.strip())
        return result

    return _parse_band_weights(start_str), _parse_band_weights(end_str)


def build_state_weights(epoch, n_epochs, config: FreqCurriculumConfig):
    """Compute per-state loss weights based on frequency curriculum.

    Returns a tensor of shape (STATE_DIM,) with weights for each state.
    """
    start_w, end_w = parse_freq_schedule(config.freq_schedule)

    # Transition progress (0 = start, 1 = end)
    t = min(epoch / max(config.freq_transition_epochs, 1), 1.0)
    # Smooth transition using cosine schedule
    t_smooth = 0.5 * (1 - np.cos(np.pi * t))

    # Compute band weights
    band_weights = {}
    for band_name in ['slow', 'medium', 'fast']:
        s_val = start_w.get(band_name, 1.0)
        e_val = end_w.get(band_name, 1.0)
        band_weights[band_name] = s_val + t_smooth * (e_val - s_val)

    # Map band weights to per-state weights
    state_weights = torch.ones(STATE_DIM)
    for band_name, indices in FREQ_BANDS.items():
        w = band_weights[band_name]
        for idx in indices:
            if idx < STATE_DIM:
                state_weights[idx] = w

    # Apply minimum weight
    state_weights = torch.clamp(state_weights, min=config.min_state_weight)

    return state_weights


class FreqCurriculumNeuralODE:
    """Neural ODE with frequency-domain curriculum training.

    Instead of changing rollout length, this changes which STATE DIMENSIONS
    get emphasized in the loss. Low-frequency states (v, theta) are learned
    first, then medium (e_y, e_psi, delta), then fast (theta_dot, delta_dot).
    """

    def __init__(self, config: FreqCurriculumConfig = None):
        self.config = config or FreqCurriculumConfig()
        self._model = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None
        self._dt = self.config.dt
        self._training_log = []

    def _compute_freq_weighted_loss(self, pred, target, state_weights, device):
        """Compute per-state weighted MSE loss.

        Args:
            pred: predicted derivatives [batch, 7]
            target: target derivatives [batch, 7]
            state_weights: per-state weights [7]
            device: torch device

        Returns:
            weighted loss scalar
        """
        state_weights_t = state_weights.to(device)
        per_state_mse = ((pred - target) ** 2).mean(dim=0)  # [7]
        weighted = per_state_mse * state_weights_t
        return weighted.mean()

    def _compute_freq_weighted_rollout_loss(self, sb, ab, rollout_steps, state_weights, device):
        """Compute frequency-weighted multi-step rollout loss."""
        if rollout_steps <= 1 or len(sb) <= rollout_steps + 1:
            return torch.tensor(0.0, device=device)

        n_roll = min(len(sb) - rollout_steps, 64)
        state_weights_t = state_weights.to(device)

        s_cur = sb[:n_roll].clone()
        total_loss = torch.tensor(0.0, device=device)

        for step in range(rollout_steps):
            a_cur = ab[step:step + n_roll]
            dsdt = self._model(s_cur, a_cur)
            s_cur = s_cur + dsdt * self._dt
            target = sb[step + 1:step + 1 + n_roll]
            if len(target) == 0:
                break
            # Per-state weighted MSE
            per_state_err = ((s_cur - target) ** 2).mean(dim=0)  # [7]
            total_loss = total_loss + (per_state_err * state_weights_t).mean()

        return total_loss / rollout_steps

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

        # Prepare data
        train_s = torch.FloatTensor(states / self._state_std)
        train_a = torch.FloatTensor(actions.reshape(-1, 1) / self._action_std)
        train_dsdot = torch.FloatTensor(deltas / (self._delta_std * self._dt))

        ds = torch.utils.data.TensorDataset(train_s, train_a, train_dsdot)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.config.batch_size, shuffle=True
        )

        opt = torch.optim.Adam(
            self._model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.config.n_epochs
        )

        # Mild rollout curriculum schedule
        if self.config.use_mild_rollout:
            mild_curriculum = [int(x) for x in self.config.mild_rollout_curriculum.split(',')]
        else:
            mild_curriculum = [1]

        # Parse frequency transition
        start_w, end_w = parse_freq_schedule(self.config.freq_schedule)

        self._model.train()
        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            epoch_loss_single = 0.0
            epoch_loss_multi = 0.0
            n_batches = 0

            # Compute current state weights
            state_weights = build_state_weights(epoch, self.config.n_epochs, self.config)

            # Determine mild rollout steps
            rollout_steps = 1
            if self.config.use_mild_rollout:
                for i, threshold in enumerate(mild_curriculum):
                    if epoch >= self.config.n_epochs * (i + 1) / (len(mild_curriculum) + 1):
                        rollout_steps = threshold

            for sb, ab, yb in loader:
                # Single-step loss with frequency weighting
                pred = self._model(sb, ab)
                loss_single = self._compute_freq_weighted_loss(
                    pred, yb, state_weights, sb.device
                )

                # Multi-step rollout loss with frequency weighting
                loss_multi = torch.tensor(0.0, device=sb.device)
                if rollout_steps > 1:
                    loss_multi = self._compute_freq_weighted_rollout_loss(
                        sb, ab, rollout_steps, state_weights, sb.device
                    )

                # Consistency loss (still useful)
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
                        + self.config.lambda_multi_freq * loss_multi
                        + self.config.lambda_consistency * loss_consistency
                        + self.config.lambda_jacobian * loss_jacobian)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()

                epoch_loss += loss.item()
                epoch_loss_single += loss_single.item()
                epoch_loss_multi += loss_multi.item() if isinstance(loss_multi, torch.Tensor) else loss_multi
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_single = epoch_loss_single / max(n_batches, 1)
            avg_multi = epoch_loss_multi / max(n_batches, 1)

            # Log frequency weights
            freq_weights = {
                band: float(state_weights[indices].mean())
                for band, indices in FREQ_BANDS.items()
            }

            self._training_log.append({
                'epoch': epoch,
                'loss': avg_loss,
                'loss_single': avg_single,
                'loss_multi': avg_multi,
                'rollout_steps': rollout_steps,
                'freq_weights': freq_weights,
                'state_weights': state_weights.tolist(),
                'lr': scheduler.get_last_lr()[0],
            })

            if epoch % 50 == 0:
                print(f"    Epoch {epoch}: loss={avg_loss:.4f} "
                      f"(single={avg_single:.4f}, multi={avg_multi:.4f}), "
                      f"rollout={rollout_steps}, "
                      f"slow={freq_weights['slow']:.2f}, "
                      f"med={freq_weights['medium']:.2f}, "
                      f"fast={freq_weights['fast']:.2f}")

        self._model.eval()

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
        return 'freq_curriculum'

    def save(self, path):
        checkpoint = {
            'config': asdict(self.config),
            'state_std': self._state_std,
            'action_std': self._action_std,
            'delta_std': self._delta_std,
            'model_state': self._model.state_dict() if self._model else None,
            'training_log': self._training_log,
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.config = FreqCurriculumConfig(**checkpoint['config'])
        self._state_std = checkpoint['state_std']
        self._action_std = checkpoint['action_std']
        self._delta_std = checkpoint['delta_std']
        self._training_log = checkpoint.get('training_log', [])

        self._model = ODEFunc(
            self.config.hidden, self.config.depth, self.config.activation
        )
        self._model.load_state_dict(checkpoint['model_state'])
        self._model.eval()
