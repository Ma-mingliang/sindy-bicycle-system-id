"""V10 Neural ODE with Learned Feedback Correction.

Architecture: dx/dt = f(x, u) + K(x)
Where K(x) is a learned feedback correction that dampens long-horizon error accumulation.

Inspired by: "Stable Long-Horizon Neural ODE Reduced-Order Models via Learned Feedback" (2026)

Key idea: The feedback term learns to pull predictions back toward a stable manifold,
preventing exponential error growth during long rollouts.
"""
import numpy as np
import os
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from .config_v9 import STATE_DIM, ACTION_DIM
from .neural_ode_v9 import ODEFunc, NeuralODEConfig


@dataclass
class FeedbackODEConfig(NeuralODEConfig):
    """Neural ODE config with feedback correction."""
    feedback_type: str = 'linear'        # 'linear', 'mlp', 'state_dependent'
    feedback_gain_init: float = -0.1     # Initial feedback gain (negative = damping)
    lambda_feedback: float = 0.01        # Regularization on feedback magnitude
    feedback_rank: int = 0               # Low-rank feedback (0 = full rank)


class FeedbackCorrection(nn.Module):
    """Learned feedback correction term K(x).

    Three variants:
    - linear: K is a constant matrix (simplest, most stable)
    - mlp: K(x) is a small MLP that takes state and outputs correction
    - state_dependent: K depends on state magnitude (stronger correction for larger deviations)
    """

    def __init__(self, state_dim, feedback_type='linear', gain_init=-0.1, rank=0):
        super().__init__()
        self.feedback_type = feedback_type
        self.state_dim = state_dim

        if feedback_type == 'linear':
            if rank > 0:
                # Low-rank feedback: K = U @ V^T
                self.U = nn.Parameter(torch.randn(state_dim, rank) * 0.01)
                self.V = nn.Parameter(torch.randn(state_dim, rank) * 0.01)
            else:
                # Full rank feedback matrix
                self.K = nn.Parameter(torch.eye(state_dim) * gain_init)

        elif feedback_type == 'mlp':
            self.net = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.Tanh(),
                nn.Linear(32, state_dim),
            )
            # Initialize small
            nn.init.zeros_(self.net[-1].bias)
            nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

        elif feedback_type == 'state_dependent':
            # Correction magnitude depends on |x|
            self.gain_net = nn.Sequential(
                nn.Linear(state_dim, 16),
                nn.Tanh(),
                nn.Linear(16, state_dim),
                nn.Sigmoid(),  # Output in [0, 1]
            )
            self.base_gain = nn.Parameter(torch.ones(state_dim) * gain_init)
            nn.init.zeros_(self.gain_net[-2].bias)
            nn.init.xavier_uniform_(self.gain_net[-2].weight, gain=0.01)

    def forward(self, x):
        """
        Args:
            x: state [batch, state_dim]
        Returns:
            correction: [batch, state_dim]
        """
        if self.feedback_type == 'linear':
            if hasattr(self, 'U'):
                # Low-rank: x @ V @ U^T
                return x @ self.V @ self.U.T
            else:
                return x @ self.K.T

        elif self.feedback_type == 'mlp':
            return self.net(x)

        elif self.feedback_type == 'state_dependent':
            # Scale correction by state magnitude
            gain = self.base_gain * self.gain_net(x)
            return gain * x


class FeedbackODEFunc(nn.Module):
    """Neural ODE with feedback correction: dx/dt = f(x,u) + K(x)."""

    def __init__(self, hidden, depth, activation='tanh',
                 feedback_type='linear', gain_init=-0.1, rank=0):
        super().__init__()
        # Base dynamics
        self.base_ode = ODEFunc(hidden, depth, activation)
        # Feedback correction
        self.feedback = FeedbackCorrection(
            STATE_DIM, feedback_type, gain_init, rank
        )

    def forward(self, s, a):
        """Compute dx/dt = f(x,u) + K(x)."""
        dx_base = self.base_ode(s, a)
        dx_feedback = self.feedback(s)
        return dx_base + dx_feedback


class FeedbackNeuralODE:
    """Neural ODE with learned feedback correction."""

    def __init__(self, config: FeedbackODEConfig = None):
        self.config = config or FeedbackODEConfig()
        self._model = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None
        self._dt = self.config.dt
        self._training_log = []

    def train(self, states, actions, deltas, state_std, action_std, delta_std,
              val_states=None, val_actions=None, val_deltas=None):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self._state_std = state_std.copy()
        self._action_std = action_std
        self._delta_std = delta_std.copy()
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0

        self._model = FeedbackODEFunc(
            self.config.hidden, self.config.depth, self.config.activation,
            self.config.feedback_type, self.config.feedback_gain_init,
            self.config.feedback_rank,
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

        curriculum = [int(x) for x in self.config.rollout_curriculum.split(',')]

        self._model.train()
        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            epoch_feedback = 0.0
            n_batches = 0

            rollout_steps = 1
            for i, threshold in enumerate(curriculum):
                if epoch >= self.config.n_epochs * (i + 1) / (len(curriculum) + 1):
                    rollout_steps = threshold

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

                # Feedback regularization (penalize large feedback)
                loss_feedback_reg = torch.tensor(0.0)
                if self.config.lambda_feedback > 0:
                    if hasattr(self._model.feedback, 'K'):
                        loss_feedback_reg = self._model.feedback.K.pow(2).sum()
                    elif hasattr(self._model.feedback, 'U'):
                        loss_feedback_reg = (self._model.feedback.U.pow(2).sum()
                                             + self._model.feedback.V.pow(2).sum())

                loss = (loss_single
                        + self.config.lambda_multi * loss_multi
                        + self.config.lambda_consistency * loss_consistency
                        + self.config.lambda_jacobian * loss_jacobian
                        + self.config.lambda_feedback * loss_feedback_reg)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()

                epoch_loss += loss.item()
                epoch_feedback += loss_feedback_reg.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_feedback = epoch_feedback / max(n_batches, 1)
            self._training_log.append({
                'epoch': epoch,
                'loss': avg_loss,
                'feedback_reg': avg_feedback,
                'rollout_steps': rollout_steps,
                'lr': scheduler.get_last_lr()[0],
            })

            if epoch % 50 == 0:
                print(f"    Epoch {epoch}: loss={avg_loss:.6f}, feedback_reg={avg_feedback:.6f}, rollout={rollout_steps}")

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
        return f'feedback_{self.config.feedback_type}'

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
        self.config = FeedbackODEConfig(**checkpoint['config'])
        self._state_std = checkpoint['state_std']
        self._action_std = checkpoint['action_std']
        self._delta_std = checkpoint['delta_std']
        self._training_log = checkpoint.get('training_log', [])

        self._model = FeedbackODEFunc(
            self.config.hidden, self.config.depth, self.config.activation,
            self.config.feedback_type, self.config.feedback_gain_init,
            self.config.feedback_rank,
        )
        self._model.load_state_dict(checkpoint['model_state'])
        self._model.eval()
