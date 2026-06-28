"""V9 Neural ODE with configurable architecture and training."""
import numpy as np
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, List

import torch
import torch.nn as nn

from .config_v9 import STATE_DIM, ACTION_DIM


@dataclass
class NeuralODEConfig:
    hidden: int = 128
    depth: int = 2
    activation: str = 'tanh'
    lr: float = 1e-3
    weight_decay: float = 0.0
    n_epochs: int = 200
    batch_size: int = 256
    dt: float = 1.0 / 30.0
    rollout_curriculum: str = '1,5,10,20'
    lambda_multi: float = 0.3
    lambda_consistency: float = 0.1
    lambda_jacobian: float = 0.01
    seed: int = 42


def _get_activation(name):
    if name == 'tanh':
        return nn.Tanh
    elif name == 'silu':
        return nn.SiLU
    elif name == 'relu':
        return nn.ReLU
    return nn.Tanh


class ODEFunc(nn.Module):
    """Neural ODE dynamics function: dx/dt = f(x, u)."""

    def __init__(self, hidden, depth, activation='tanh'):
        super().__init__()
        act = _get_activation(activation)
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)

        # Initialize last layer small for stability
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)


class NeuralODEV9:
    """V9 Neural ODE with configurable architecture."""

    def __init__(self, config: NeuralODEConfig = None):
        self.config = config or NeuralODEConfig()
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

        # Parse rollout curriculum
        curriculum = [int(x) for x in self.config.rollout_curriculum.split(',')]

        self._model.train()
        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Determine current rollout steps
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

                # Consistency loss: theta_dot should relate to delta
                loss_consistency = torch.tensor(0.0)
                if self.config.lambda_consistency > 0:
                    s_next_norm = sb + pred * self._dt
                    # theta should change proportionally to theta_dot
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
                    # Compute Jacobian norm
                    jac_norm = 0.0
                    for i in range(STATE_DIM):
                        grad = torch.autograd.grad(
                            dsdt[:, i].sum(), s_req, create_graph=True
                        )[0]
                        jac_norm = jac_norm + grad.pow(2).sum()
                    loss_jacobian = jac_norm / (STATE_DIM * min(32, len(sb)))

                loss = (loss_single
                        + self.config.lambda_multi * loss_multi
                        + self.config.lambda_consistency * loss_consistency
                        + self.config.lambda_jacobian * loss_jacobian)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            self._training_log.append({
                'epoch': epoch,
                'loss': avg_loss,
                'rollout_steps': rollout_steps,
                'lr': scheduler.get_last_lr()[0],
            })

        self._model.eval()

    def predict(self, s, tau):
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0)

        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).numpy()[0]

        dsdt = dsdt_norm * self._delta_std * self._dt
        return s + dsdt

    def predict_batch(self, states, actions):
        """Batch prediction for efficiency."""
        s_norm = torch.FloatTensor(states / self._state_std)
        a_norm = torch.FloatTensor(actions.reshape(-1, 1) / self._action_std)

        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).numpy()

        return states + dsdt_norm * self._delta_std * self._dt

    def predict_with_uncertainty(self, s, tau):
        return self.predict(s, tau), None

    def name(self):
        return 'neural_ode_v9'

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
        self.config = NeuralODEConfig(**checkpoint['config'])
        self._state_std = checkpoint['state_std']
        self._action_std = checkpoint['action_std']
        self._delta_std = checkpoint['delta_std']
        self._training_log = checkpoint.get('training_log', [])

        self._model = ODEFunc(
            self.config.hidden, self.config.depth, self.config.activation
        )
        self._model.load_state_dict(checkpoint['model_state'])
        self._model.eval()
