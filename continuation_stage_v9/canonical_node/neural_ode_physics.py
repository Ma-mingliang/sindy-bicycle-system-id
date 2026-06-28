"""V9 Neural ODE with Physics-Informed Loss Enhancement.

Adds physical constraints to the training loss:
1. Lateral velocity: e_y_dot ≈ v * sin(e_psi)
2. Curvature consistency: theta_dot ≈ d(theta)/dt
3. Steering consistency: delta_dot ≈ d(delta)/dt
"""
import numpy as np
import os
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from .config_v9 import STATE_DIM, ACTION_DIM
from .neural_ode_v9 import ODEFunc, NeuralODEConfig


@dataclass
class PhysicsODEConfig(NeuralODEConfig):
    """Neural ODE config with physics constraint weights."""
    lambda_physics_ey: float = 0.1      # e_y_dot ≈ v * sin(e_psi)
    lambda_physics_theta: float = 0.05   # theta consistency
    lambda_physics_delta: float = 0.05   # delta consistency
    lambda_multi: float = 0.3
    lambda_consistency: float = 0.1
    lambda_jacobian: float = 0.01
    rollout_curriculum: str = '1,5,10,20'


class PhysicsNeuralODE:
    """Neural ODE with physics-informed loss."""

    def __init__(self, config: PhysicsODEConfig = None):
        self.config = config or PhysicsODEConfig()
        self._model = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None
        self._dt = self.config.dt
        self._training_log = []

    def _compute_physics_loss(self, sb, pred_dsdt):
        """Compute physics constraint losses.

        Args:
            sb: normalized states [batch, 7] = [e_y, e_psi, v, theta, theta_dot, delta, delta_dot]
            pred_dsdt: predicted derivative [batch, 7]

        Returns:
            dict of physics losses
        """
        # Denormalize to physical units
        # sb is normalized by state_std, pred_dsdt is normalized by delta_std*dt
        dt = self._dt

        # Constraint 1: e_y_dot ≈ v * sin(e_psi)
        # In normalized space: e_y_dot_norm * delta_std[0] * dt ≈ v_norm * std_v * sin(e_psi_norm * std_epsi)
        # Simplified: use the predicted derivative for e_y and compare with v*sin(e_psi)
        e_y_dot_pred = pred_dsdt[:, 0]  # normalized e_y derivative
        v_norm = sb[:, 2]  # normalized v
        e_psi_norm = sb[:, 1]  # normalized e_psi

        # Physical relationship: e_y_dot = v * sin(e_psi)
        # In normalized form: e_y_dot * std_ey / (std_v * v) ≈ sin(e_psi * std_epsi)
        # Simplified loss: penalize when e_y_dot is not proportional to v * sin(e_psi)
        # Use a soft constraint: the ratio should be consistent
        physics_ey = torch.tensor(0.0, device=sb.device)
        if self.config.lambda_physics_ey > 0:
            # e_y_dot should be roughly v * sin(e_psi) in physical units
            # Since we're in normalized space, use the ratio
            e_psi_physical = e_psi_norm * self._state_std_tensor[1]
            v_physical = v_norm * self._state_std_tensor[2]
            expected_ey_dot = v_physical * torch.sin(e_psi_physical)
            # Convert expected to normalized derivative space
            expected_ey_dot_norm = expected_ey_dot / (self._delta_std_tensor[0] * dt)
            physics_ey = nn.functional.mse_loss(e_y_dot_pred, expected_ey_dot_norm)

        # Constraint 2: theta_dot should relate to theta change
        # theta_dot_pred ≈ (theta_next - theta) / dt
        physics_theta = torch.tensor(0.0, device=sb.device)
        if self.config.lambda_physics_theta > 0:
            theta_dot_pred = pred_dsdt[:, 4]  # predicted theta_dot
            theta_dot_target = pred_dsdt[:, 3]  # theta change should match theta_dot
            # Actually: theta_dot IS the derivative of theta
            # So pred_dsdt[:, 3] (theta derivative) should equal sb[:, 4] (theta_dot)
            # In normalized space: pred_dsdt[:, 3] * delta_std[3] * dt ≈ sb[:, 4] * state_std[4]
            theta_change_pred = pred_dsdt[:, 3] * self._delta_std_tensor[3] * dt
            theta_dot_value = sb[:, 4] * self._state_std_tensor[4]
            physics_theta = nn.functional.mse_loss(theta_change_pred, theta_dot_value)

        # Constraint 3: delta_dot should relate to delta change
        physics_delta = torch.tensor(0.0, device=sb.device)
        if self.config.lambda_physics_delta > 0:
            delta_change_pred = pred_dsdt[:, 5] * self._delta_std_tensor[5] * dt
            delta_dot_value = sb[:, 6] * self._state_std_tensor[6]
            physics_delta = nn.functional.mse_loss(delta_change_pred, delta_dot_value)

        return {
            'ey': physics_ey,
            'theta': physics_theta,
            'delta': physics_delta,
        }

    def train(self, states, actions, deltas, state_std, action_std, delta_std,
              val_states=None, val_actions=None, val_deltas=None):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self._state_std = state_std.copy()
        self._action_std = action_std
        self._delta_std = delta_std.copy()
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0

        # Store as tensors for physics loss computation
        device = torch.device('cpu')
        self._state_std_tensor = torch.FloatTensor(self._state_std).to(device)
        self._delta_std_tensor = torch.FloatTensor(self._delta_std).to(device)

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

        curriculum = [int(x) for x in self.config.rollout_curriculum.split(',')]

        self._model.train()
        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            epoch_physics = 0.0
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

                # Physics constraints
                physics_losses = self._compute_physics_loss(sb, pred)
                loss_physics = (self.config.lambda_physics_ey * physics_losses['ey']
                                + self.config.lambda_physics_theta * physics_losses['theta']
                                + self.config.lambda_physics_delta * physics_losses['delta'])

                loss = (loss_single
                        + self.config.lambda_multi * loss_multi
                        + self.config.lambda_consistency * loss_consistency
                        + self.config.lambda_jacobian * loss_jacobian
                        + loss_physics)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()

                epoch_loss += loss.item()
                epoch_physics += loss_physics.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_physics = epoch_physics / max(n_batches, 1)
            self._training_log.append({
                'epoch': epoch,
                'loss': avg_loss,
                'physics_loss': avg_physics,
                'rollout_steps': rollout_steps,
                'lr': scheduler.get_last_lr()[0],
            })

            if epoch % 50 == 0:
                print(f"    Epoch {epoch}: loss={avg_loss:.6f}, physics={avg_physics:.6f}, rollout={rollout_steps}")

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
        return 'physics_neural_ode_v9'

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
        self.config = PhysicsODEConfig(**checkpoint['config'])
        self._state_std = checkpoint['state_std']
        self._action_std = checkpoint['action_std']
        self._delta_std = checkpoint['delta_std']
        self._training_log = checkpoint.get('training_log', [])
        self._state_std_tensor = torch.FloatTensor(self._state_std)
        self._delta_std_tensor = torch.FloatTensor(self._delta_std)

        self._model = ODEFunc(
            self.config.hidden, self.config.depth, self.config.activation
        )
        self._model.load_state_dict(checkpoint['model_state'])
        self._model.eval()
