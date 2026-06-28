"""V10 Neural ODE with Attractor-Based Stabilization.

Architecture: dx/dt = f(x, u) + alpha(x) * attractor(x)

The attractor is fundamentally different from linear feedback:
1. It learns a "stable manifold" - the set of physically reasonable states
2. It only activates when states drift out-of-distribution (adaptive gating)
3. It does NOT fight the base dynamics - it only corrects when the state is "unreasonable"
4. Per-state correction weights let us focus on error-prone states (e_y, e_psi)

Why linear feedback failed (V10 results):
- feedback_linear_m0.1: H=100 NMAE=0.658 (baseline=0.506) - WORSE
- All 8 variants performed worse than the baseline
- Root cause: linear K*x applies uniform damping that fights the learned dynamics

The attractor approach avoids this by:
- Learning WHERE the stable manifold is (not assuming it's at x=0)
- Being state-ADAPTIVE: strong when far from manifold, zero when close
- Including a consistency loss: zero correction on training-data states

Inspired by:
- "Enhancing Robustness of Neural ODEs via Attractor Dynamics" (Zhou et al. 2025)
- "Goal-Conditioned Neural ODEs with Guaranteed Safety and Stability" (Liu et al. 2026)
"""
import numpy as np
import os
from dataclasses import dataclass, asdict
from typing import Optional, List

import torch
import torch.nn as nn

from .config_v9 import STATE_DIM, ACTION_DIM, STATE_NAMES_7D
from .neural_ode_v9 import ODEFunc, NeuralODEConfig


# ============================================================
# Configuration
# ============================================================
@dataclass
class AttractorODEConfig(NeuralODEConfig):
    """Neural ODE config with attractor stabilization."""

    # Attractor architecture
    attractor_type: str = 'residual'       # 'full', 'residual', 'ensemble'
    attractor_hidden: int = 32             # Hidden dim for attractor networks
    attractor_depth: int = 2               # Depth of attractor MLP

    # Per-state correction weights (which states to correct)
    # Default: only e_y (0) and e_psi (1) are error-prone
    correct_mask: str = '1,1,0,0,0,0,0'   # Comma-separated 0/1 for each of 7 states

    # Adaptive gating
    gate_type: str = 'distance'            # 'distance', 'learned', 'sigmoid'
    gate_threshold: float = 0.5            # Threshold for distance-based gating

    # Consistency loss: ensures attractor doesn't change ODE on training data
    lambda_consistency_attractor: float = 0.5

    # Attractor strength regularization
    lambda_attractor_reg: float = 0.01

    # Manifold regularization: keeps manifold encoder close to identity
    lambda_manifold_id: float = 0.1

    # Rollout curriculum (extended to longer horizons)
    rollout_curriculum: str = '1,5,10,20,50'

    # Multi-step loss
    lambda_multi: float = 0.3


# ============================================================
# Attractor Modules
# ============================================================
class LearnedManifold(nn.Module):
    """Learns the stable manifold - the set of physically reasonable states.

    The manifold encoder maps a state to its "ideal" projection on the
    stable manifold. States near the manifold pass through unchanged;
    states far from the manifold get corrected toward it.
    """

    def __init__(self, state_dim, hidden=32, depth=2):
        super().__init__()
        act = nn.Tanh
        layers = [nn.Linear(state_dim, hidden), act()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), act()])
        layers.append(nn.Linear(hidden, state_dim))
        self.net = nn.Sequential(*layers)

        # Initialize near identity: manifold(x) ~ x for states in training distribution
        nn.init.zeros_(self.net[-1].bias)
        nn.init.eye_(self.net[-1].weight)
        # Scale down the last layer to keep initial corrections small
        self.net[-1].weight.data *= 0.1

    def forward(self, x):
        """Map state to its projection on the learned stable manifold.

        Args:
            x: [batch, state_dim] normalized states
        Returns:
            x_manifold: [batch, state_dim] projected states on manifold
        """
        return self.net(x)


class AdaptiveGate(nn.Module):
    """Computes the attractor activation strength.

    The gate outputs a value in [0, 1] for each state dimension:
    - 0 when the state is on the manifold (no correction needed)
    - 1 when the state is far from the manifold (strong correction)

    This is the KEY difference from linear feedback: the gate learns
    to be selective about when and how much to correct.
    """

    def __init__(self, state_dim, gate_type='distance', threshold=0.5):
        super().__init__()
        self.gate_type = gate_type
        self.threshold = threshold

        if gate_type == 'learned':
            # Small MLP that learns to detect out-of-distribution states
            self.net = nn.Sequential(
                nn.Linear(state_dim, 16),
                nn.Tanh(),
                nn.Linear(16, state_dim),
                nn.Sigmoid(),
            )
            # Initialize to output ~0.5 (neutral)
            nn.init.zeros_(self.net[-1].bias)
            nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

        elif gate_type == 'distance':
            # Distance-based: gate = sigmoid(|x - x_manifold| - threshold)
            self.scale = nn.Parameter(torch.ones(1) * 3.0)
            self.threshold_val = threshold

        elif gate_type == 'sigmoid':
            # Simple sigmoid on state magnitude
            self.scale = nn.Parameter(torch.ones(state_dim) * 2.0)
            self.offset = nn.Parameter(torch.zeros(state_dim))

    def forward(self, x, x_manifold=None):
        """Compute gate values in [0, 1].

        Args:
            x: [batch, state_dim] current state
            x_manifold: [batch, state_dim] manifold projection (for distance gate)
        Returns:
            gate: [batch, state_dim] activation strengths in [0, 1]
        """
        if self.gate_type == 'learned':
            return self.net(x)

        elif self.gate_type == 'distance' and x_manifold is not None:
            # Distance from manifold per state dimension
            dist = torch.abs(x - x_manifold)
            # Sigmoid: activates when distance > threshold
            gate = torch.sigmoid(self.scale * (dist - self.threshold_val))
            return gate

        elif self.gate_type == 'sigmoid':
            # Simple magnitude-based gating
            gate = torch.sigmoid(self.scale * torch.abs(x) + self.offset)
            return gate

        # Fallback: uniform gate
        return torch.ones_like(x) * 0.5


class AttractorHead(nn.Module):
    """Single attractor that pulls states toward the learned manifold.

    output = -(x - x_manifold) * gate
    This is a pull toward the manifold, with adaptive strength.
    """

    def __init__(self, state_dim, hidden=32, depth=2, correct_mask=None):
        super().__init__()
        self.state_dim = state_dim

        # The manifold encoder
        self.manifold = LearnedManifold(state_dim, hidden, depth)

        # The adaptive gate
        self.gate = AdaptiveGate(state_dim, gate_type='distance')

        # Per-state correction weights (learnable)
        if correct_mask is not None:
            # Initialize correction weights: 1 for states to correct, 0 for others
            self.correction_weight = nn.Parameter(
                torch.FloatTensor(correct_mask).unsqueeze(0)
            )
        else:
            self.correction_weight = nn.Parameter(torch.ones(1, state_dim))

    def forward(self, x):
        """Compute attractor correction.

        Args:
            x: [batch, state_dim] normalized state
        Returns:
            correction: [batch, state_dim] force pulling toward manifold
            info: dict with diagnostic information
        """
        # Project to manifold
        x_manifold = self.manifold(x)

        # Compute distance from manifold
        distance = x - x_manifold

        # Compute adaptive gate
        gate = self.gate(x, x_manifold)

        # Apply per-state correction weights
        weight = torch.sigmoid(self.correction_weight)  # [0, 1]

        # Final correction: pull toward manifold with adaptive strength
        correction = -distance * gate * weight

        info = {
            'x_manifold': x_manifold,
            'distance': distance,
            'gate': gate,
            'correction_weight': weight,
            'correction_norm': correction.pow(2).sum(dim=-1).mean(),
        }

        return correction, info


class EnsembleAttractor(nn.Module):
    """Multiple attractors for different operating regions.

    Different attractors specialize in different parts of the state space
    (e.g., one for small perturbations, one for large roll angles).
    A gating network smoothly blends their outputs.
    """

    def __init__(self, state_dim, n_attractors=3, hidden=32, depth=2,
                 correct_mask=None):
        super().__init__()
        self.n_attractors = n_attractors

        # Multiple attractor heads
        self.attractors = nn.ModuleList([
            AttractorHead(state_dim, hidden, depth, correct_mask)
            for _ in range(n_attractors)
        ])

        # Gating network to blend attractor outputs
        self.gate_net = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.Tanh(),
            nn.Linear(16, n_attractors),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        """Compute blended attractor correction.

        Args:
            x: [batch, state_dim] normalized state
        Returns:
            correction: [batch, state_dim] blended attractor force
            info: dict with per-attractor diagnostics
        """
        # Compute per-attractor corrections
        corrections = []
        all_info = []
        for attractor in self.attractors:
            c, info = attractor(x)
            corrections.append(c)
            all_info.append(info)

        # Stack: [batch, state_dim, n_attractors]
        corrections_stacked = torch.stack(corrections, dim=-1)

        # Gating weights: [batch, n_attractors]
        weights = self.gate_net(x)

        # Weighted blend: [batch, state_dim]
        correction = (corrections_stacked * weights.unsqueeze(1)).sum(dim=-1)

        info = {
            'weights': weights,
            'per_attractor_norm': [c['correction_norm'] for c in all_info],
        }

        return correction, info


# ============================================================
# ODEFunc with Attractor
# ============================================================
class AttractorODEFunc(nn.Module):
    """Neural ODE with attractor stabilization: dx/dt = f(x,u) + alpha * attractor(x).

    The base ODE f(x,u) learns the dynamics from data.
    The attractor term learns to pull states back to the stable manifold
    when they drift out-of-distribution during long rollouts.
    """

    def __init__(self, hidden, depth, activation='tanh',
                 attractor_type='residual', attractor_hidden=32,
                 attractor_depth=2, correct_mask=None):
        super().__init__()

        # Base dynamics (same as V9)
        self.base_ode = ODEFunc(hidden, depth, activation)

        # Attractor
        if attractor_type == 'ensemble':
            self.attractor = EnsembleAttractor(
                STATE_DIM, n_attractors=3, hidden=attractor_hidden,
                depth=attractor_depth, correct_mask=correct_mask
            )
        else:
            self.attractor = AttractorHead(
                STATE_DIM, hidden=attractor_hidden,
                depth=attractor_depth, correct_mask=correct_mask
            )

    def forward(self, s, a):
        """Compute dx/dt = f(x,u) + attractor(x).

        Args:
            s: [batch, state_dim] normalized state
            a: [batch, 1] normalized action
        Returns:
            dsdt: [batch, state_dim] total derivative
        """
        # Base dynamics
        dx_base = self.base_ode(s, a)

        # Attractor correction
        dx_attractor, _ = self.attractor(s)

        return dx_base + dx_attractor


# ============================================================
# Training Class
# ============================================================
class AttractorNeuralODE:
    """Neural ODE with attractor-based stabilization.

    Training strategy:
    1. Standard single-step loss (fitted to data)
    2. Multi-step rollout loss (reduces error accumulation)
    3. Consistency loss: attractor should be zero on training-data states
    4. Attractor regularization: penalize large corrections
    5. Manifold identity regularization: manifold(x) ~ x near training data
    """

    def __init__(self, config: AttractorODEConfig = None):
        self.config = config or AttractorODEConfig()
        self._model = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None
        self._dt = self.config.dt
        self._training_log = []

    def _parse_correct_mask(self):
        """Parse correct_mask string to list of floats."""
        mask_str = self.config.correct_mask.split(',')
        return [float(x.strip()) for x in mask_str]

    def train(self, states, actions, deltas, state_std, action_std, delta_std,
              val_states=None, val_actions=None, val_deltas=None):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self._state_std = state_std.copy()
        self._action_std = action_std
        self._delta_std = delta_std.copy()
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0

        correct_mask = self._parse_correct_mask()

        self._model = AttractorODEFunc(
            self.config.hidden, self.config.depth, self.config.activation,
            attractor_type=self.config.attractor_type,
            attractor_hidden=self.config.attractor_hidden,
            attractor_depth=self.config.attractor_depth,
            correct_mask=correct_mask,
        )

        # Prepare data
        train_s = torch.FloatTensor(states / self._state_std)
        train_a = torch.FloatTensor(actions.reshape(-1, 1) / self._action_std)
        train_dsdot = torch.FloatTensor(deltas / (self._delta_std * self._dt))

        ds = torch.utils.data.TensorDataset(train_s, train_a, train_dsdot)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.config.batch_size, shuffle=True
        )

        # Two-parameter groups: base ODE and attractor with separate LR
        base_params = list(self._model.base_ode.parameters())
        attractor_params = list(self._model.attractor.parameters())

        opt = torch.optim.Adam([
            {'params': base_params, 'lr': self.config.lr},
            {'params': attractor_params, 'lr': self.config.lr * 0.5},  # Slower for attractor
        ], weight_decay=self.config.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.config.n_epochs
        )

        curriculum = [int(x) for x in self.config.rollout_curriculum.split(',')]

        self._model.train()
        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            epoch_attractor = 0.0
            epoch_consistency = 0.0
            n_batches = 0

            # Determine current rollout steps
            rollout_steps = 1
            for i, threshold in enumerate(curriculum):
                if epoch >= self.config.n_epochs * (i + 1) / (len(curriculum) + 1):
                    rollout_steps = threshold

            for sb, ab, yb in loader:
                # ========================================
                # 1. Single-step loss (primary)
                # ========================================
                pred = self._model(sb, ab)
                loss_single = nn.functional.mse_loss(pred, yb)

                # ========================================
                # 2. Multi-step rollout loss
                # ========================================
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

                # ========================================
                # 3. Consistency loss: attractor near zero on training data
                # ========================================
                # On training data, the attractor should not change predictions
                # This ensures the attractor only activates for OOD states
                loss_consistency = torch.tensor(0.0)
                if self.config.lambda_consistency_attractor > 0:
                    # Compute attractor correction on training batch
                    if hasattr(self._model.attractor, 'attractors'):
                        # Ensemble: sum of corrections
                        total_corr = torch.zeros_like(sb)
                        for att in self._model.attractor.attractors:
                            corr, _ = att(sb)
                            total_corr = total_corr + corr.abs()
                        avg_correction = total_corr.mean()
                    else:
                        corr, _ = self._model.attractor(sb)
                        avg_correction = corr.abs().mean()

                    # The attractor should produce small corrections on training data
                    loss_consistency = avg_correction

                # ========================================
                # 4. Attractor regularization
                # ========================================
                loss_attractor_reg = torch.tensor(0.0)
                if self.config.lambda_attractor_reg > 0:
                    if hasattr(self._model.attractor, 'attractors'):
                        for att in self._model.attractor.attractors:
                            loss_attractor_reg = loss_attractor_reg + \
                                att.correction_weight.pow(2).sum()
                    else:
                        loss_attractor_reg = self._model.attractor.correction_weight.pow(2).sum()

                # ========================================
                # 5. Manifold identity regularization
                # ========================================
                loss_manifold_id = torch.tensor(0.0)
                if self.config.lambda_manifold_id > 0:
                    if hasattr(self._model.attractor, 'attractors'):
                        for att in self._model.attractor.attractors:
                            x_recon = att.manifold(sb)
                            loss_manifold_id = loss_manifold_id + \
                                nn.functional.mse_loss(x_recon, sb)
                    else:
                        x_recon = self._model.attractor.manifold(sb)
                        loss_manifold_id = nn.functional.mse_loss(x_recon, sb)

                # ========================================
                # Total loss
                # ========================================
                loss = (loss_single
                        + self.config.lambda_multi * loss_multi
                        + self.config.lambda_consistency_attractor * loss_consistency
                        + self.config.lambda_attractor_reg * loss_attractor_reg
                        + self.config.lambda_manifold_id * loss_manifold_id)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()

                epoch_loss += loss.item()
                epoch_attractor += loss_attractor_reg.item()
                epoch_consistency += loss_consistency.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_attractor = epoch_attractor / max(n_batches, 1)
            avg_consistency = epoch_consistency / max(n_batches, 1)
            self._training_log.append({
                'epoch': epoch,
                'loss': avg_loss,
                'attractor_reg': avg_attractor,
                'consistency': avg_consistency,
                'rollout_steps': rollout_steps,
                'lr': scheduler.get_last_lr()[0],
            })

            if epoch % 50 == 0:
                print(f"    Epoch {epoch}: loss={avg_loss:.6f}, "
                      f"attractor={avg_attractor:.6f}, "
                      f"consistency={avg_consistency:.6f}, "
                      f"rollout={rollout_steps}")

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

    def predict_with_attractor_info(self, s, tau):
        """Predict and return attractor diagnostic info."""
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0)
        with torch.no_grad():
            # Get base dynamics
            dx_base = self._model.base_ode(s_norm, a_norm)
            # Get attractor correction
            if hasattr(self._model.attractor, 'attractors'):
                corr = torch.zeros_like(s_norm)
                for att in self._model.attractor.attractors:
                    c, _ = att(s_norm)
                    corr = corr + c
                dx_attractor = corr
            else:
                dx_attractor, info = self._model.attractor(s_norm)

            dsdt_total = dx_base + dx_attractor

        dsdt = dsdt_total.numpy()[0] * self._delta_std * self._dt
        base_dsdt = dx_base.numpy()[0] * self._delta_std * self._dt
        attr_dsdt = dx_attractor.numpy()[0] * self._delta_std * self._dt

        return s + dsdt, {
            'base_dsdt': base_dsdt,
            'attractor_dsdt': attr_dsdt,
            'attractor_magnitude': float(np.abs(attr_dsdt).mean()),
        }

    def name(self):
        return f'attractor_{self.config.attractor_type}'

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
        self.config = AttractorODEConfig(**checkpoint['config'])
        self._state_std = checkpoint['state_std']
        self._action_std = checkpoint['action_std']
        self._delta_std = checkpoint['delta_std']
        self._training_log = checkpoint.get('training_log', [])

        correct_mask = self._parse_correct_mask()
        self._model = AttractorODEFunc(
            self.config.hidden, self.config.depth, self.config.activation,
            attractor_type=self.config.attractor_type,
            attractor_hidden=self.config.attractor_hidden,
            attractor_depth=self.config.attractor_depth,
            correct_mask=correct_mask,
        )
        self._model.load_state_dict(checkpoint['model_state'])
        self._model.eval()
