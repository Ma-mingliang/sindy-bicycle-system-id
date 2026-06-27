"""V11 Neural ODE with Contractivity-Promoting Regularization.

Architecture: same as V9 baseline (dx/dt = f(x,u)), NO architecture changes.
Regularization: loss_term = lambda_contractive * ReLU(max_eigenvalue(J_sym))

Where J_sym = (J + J^T) / 2 is the symmetric part of the Jacobian, and
max_eigenvalue(J_sym) is the spectral abscissa (logarithmic norm).

Key insight from Zakwan et al. 2025, Guglielmi et al. 2025:
- Penalizing positive eigenvalues of the Jacobian's symmetric part
  encourages the ODE to be locally contractive (nearby trajectories converge)
- This is fundamentally different from feedback correction:
  feedback ADDS a correction term (fighting base dynamics),
  contractivity REGULARIZES the base dynamics (guiding learning)
- No extra parameters, no architecture change, just an additional loss term

Why this should work better than feedback correction:
1. Feedback adds K*x that competes with learned dynamics -> unstable training
2. Contractivity regularizes f(x,u) itself -> base dynamics become stable
3. The Jacobian penalty is smooth and differentiable -> clean gradients
4. It works across the full state space -> no need to pick which states to feedback

Spectral abscissa computation:
- J = df/dx is STATE_DIM x STATE_DIM (7x7) for each sample
- J_sym = (J + J^T) / 2 is symmetric, so eigenvalues are real
- For a 7x7 matrix, direct eigendecomposition is cheap (7x7 = 49 elements)
- We use torch.linalg.eigvalsh for the symmetric part (optimized for symmetric)
- Alternatively, use power iteration for max eigenvalue only (cheaper for large dims)

For 7D: direct eigendecomposition is fine (7 eigenvalues, negligible cost).
For higher dimensions: use power iteration (1 power step = 1 matvec).
"""
import numpy as np
import os
from dataclasses import dataclass, asdict
from typing import Optional, Literal

import torch
import torch.nn as nn

from .config_v9 import STATE_DIM, ACTION_DIM
from .neural_ode_v9 import ODEFunc, NeuralODEConfig


@dataclass
class ContractiveODEConfig(NeuralODEConfig):
    """Neural ODE config with contractivity regularization.

    Contractivity regularization adds a loss term that penalizes positive
    eigenvalues of the Jacobian's symmetric part, encouraging the ODE to be
    locally contracting (nearby trajectories converge).
    """
    # Contractivity regularization
    lambda_contractive: float = 0.05           # Strength of contractivity loss
    contractive_method: str = 'symmetric_eig'  # 'symmetric_eig' or 'power_iter'
    contractive_power_steps: int = 5           # Power iteration steps (if method='power_iter')
    contractive_warmup_epochs: int = 20        # Epochs before enabling contractivity loss
    contractive_ramp_epochs: int = 50          # Epochs over which lambda ramps from 0 to full

    # Whether to apply contractivity to all states or only sensitive ones
    contractive_states: str = 'all'            # 'all' or 'sensitive' (e_y, e_psi only)

    # Optional: reduce existing Jacobian norm loss when contractive is enabled
    # (they overlap significantly; double-counting wastes compute)
    lambda_jacobian: float = 0.01

    # Diagnostic: log max eigenvalue during training
    log_eigenvalue: bool = True


def compute_jacobian(model, s, a):
    """Compute the Jacobian df/dx for a batch of states.

    Args:
        model: ODEFunc (or compatible) that takes (s, a) -> dsdt
        s: state tensor [batch, STATE_DIM], requires_grad=True
        a: action tensor [batch, ACTION_DIM]

    Returns:
        J: Jacobian tensor [batch, STATE_DIM, STATE_DIM]
           J[b, i, j] = d(dsdt[b,i]) / d(s[b,j])
    """
    batch_size = s.shape[0]
    dsdt = model(s, a)  # [batch, STATE_DIM]

    # Initialize Jacobian: J[b, i, j] = d(dsdt[b,i]) / d(s[b,j])
    J = torch.zeros(batch_size, STATE_DIM, STATE_DIM, device=s.device, dtype=s.dtype)

    for i in range(STATE_DIM):
        grad = torch.autograd.grad(
            dsdt[:, i].sum(), s, create_graph=True, retain_graph=True
        )[0]  # [batch, STATE_DIM]
        J[:, i, :] = grad

    return J


def max_eigenvalue_symmetric(J):
    """Compute max eigenvalue of the symmetric part (J + J^T) / 2.

    For a 7x7 Jacobian, this is cheap: eigendecomposition of a 7x7 matrix.
    Returns the maximum eigenvalue (spectral abscissa of the symmetric part).

    The symmetric part's eigenvalues are real (by symmetry), and its largest
    eigenvalue is the logarithmic norm (matrix measure) of J.
    """
    J_sym = 0.5 * (J + J.transpose(-1, -2))  # [batch, STATE_DIM, STATE_DIM]

    # For small matrices (7x7), eigvalsh is efficient
    # eigvalsh returns eigenvalues in ascending order
    eigenvalues = torch.linalg.eigvalsh(J_sym)  # [batch, STATE_DIM]

    # Max eigenvalue (last one, since ascending order)
    max_eig = eigenvalues[:, -1]  # [batch]

    return max_eig, eigenvalues


def max_eigenvalue_power_iter(J, n_steps=5):
    """Compute max eigenvalue of symmetric part via power iteration.

    More efficient for large STATE_DIM (but 7x7 is already cheap).
    Power iteration: v_{k+1} = J_sym @ v_k / ||J_sym @ v_k||
    lambda_max approx = v^T @ J_sym @ v

    Returns:
        max_eig: approximate max eigenvalue [batch]
    """
    batch_size = J.shape[0]
    J_sym = 0.5 * (J + J.transpose(-1, -2))  # [batch, STATE_DIM, STATE_DIM]

    # Random initialization
    v = torch.randn(batch_size, STATE_DIM, device=J.device, dtype=J.dtype)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    for _ in range(n_steps):
        # J_sym @ v: [batch, STATE_DIM, STATE_DIM] @ [batch, STATE_DIM, 1]
        Jv = torch.bmm(J_sym, v.unsqueeze(-1)).squeeze(-1)  # [batch, STATE_DIM]
        v = Jv / (Jv.norm(dim=-1, keepdim=True) + 1e-8)

    # Rayleigh quotient: v^T J_sym v
    Jv = torch.bmm(J_sym, v.unsqueeze(-1)).squeeze(-1)
    max_eig = (v * Jv).sum(dim=-1)  # [batch]

    return max_eig


def contractivity_loss(model, s, a, method='symmetric_eig', n_power_steps=5):
    """Compute the contractivity-promoting loss.

    Loss = mean(ReLU(max_eigenvalue(J_sym)))

    This penalizes positive eigenvalues of the Jacobian's symmetric part,
    encouraging the ODE to be locally contractive.

    Args:
        model: ODEFunc
        s: state tensor [batch, STATE_DIM], requires_grad=True
        a: action tensor [batch, ACTION_DIM]
        method: 'symmetric_eig' (exact for small dims) or 'power_iter'
        n_power_steps: number of power iteration steps (if method='power_iter')

    Returns:
        loss: scalar contractivity loss
        max_eigs: max eigenvalue per sample [batch] (for diagnostics)
        all_eigs: all eigenvalues [batch, STATE_DIM] (for diagnostics, None if power_iter)
    """
    J = compute_jacobian(model, s, a)  # [batch, STATE_DIM, STATE_DIM]

    if method == 'symmetric_eig':
        max_eigs, all_eigs = max_eigenvalue_symmetric(J)
    elif method == 'power_iter':
        max_eigs = max_eigenvalue_power_iter(J, n_power_steps)
        all_eigs = None
    else:
        raise ValueError(f"Unknown method: {method}")

    # Contractivity loss: penalize positive eigenvalues
    # ReLU(max_eig) = max(0, max_eig) -- only penalize expanding directions
    loss = torch.relu(max_eigs).mean()

    return loss, max_eigs.detach(), all_eigs


class NeuralODEContractive:
    """V11 Neural ODE with contractivity-promoting regularization.

    Same architecture as V9 baseline, but with an additional loss term
    that encourages the Jacobian to have non-positive eigenvalues.

    This is the "contractivity-promoting regularization" approach from:
    - Zakwan et al. 2025: "Robust Convolution Neural ODEs via Contractivity"
    - Guglielmi et al. 2025: Contractivity analysis for Neural ODEs
    """

    def __init__(self, config: ContractiveODEConfig = None):
        self.config = config or ContractiveODEConfig()
        self._model = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None
        self._dt = self.config.dt
        self._training_log = []

    def _get_contractive_lambda(self, epoch):
        """Compute effective lambda with warmup and linear ramp.

        Schedule:
        - epoch < warmup_epochs: lambda = 0 (let base dynamics learn first)
        - warmup_epochs <= epoch < warmup + ramp: lambda linearly ramps to full
        - epoch >= warmup + ramp: lambda = lambda_contractive (full strength)
        """
        if epoch < self.config.contractive_warmup_epochs:
            return 0.0
        elif epoch < self.config.contractive_warmup_epochs + self.config.contractive_ramp_epochs:
            progress = (epoch - self.config.contractive_warmup_epochs) / self.config.contractive_ramp_epochs
            return self.config.lambda_contractive * progress
        else:
            return self.config.lambda_contractive

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
            epoch_contractive = 0.0
            epoch_max_eig = []
            n_batches = 0

            # Effective contractivity lambda for this epoch
            eff_lambda = self._get_contractive_lambda(epoch)

            # Determine current rollout steps
            rollout_steps = 1
            for i, threshold in enumerate(curriculum):
                if epoch >= self.config.n_epochs * (i + 1) / (len(curriculum) + 1):
                    rollout_steps = threshold

            for sb, ab, yb in loader:
                # === Single-step loss ===
                pred = self._model(sb, ab)
                loss_single = nn.functional.mse_loss(pred, yb)

                # === Multi-step rollout loss ===
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

                # === Consistency loss ===
                loss_consistency = torch.tensor(0.0)
                if self.config.lambda_consistency > 0:
                    s_next_norm = sb + pred * self._dt
                    theta_pred = s_next_norm[:, 3]
                    theta_dot = sb[:, 4]
                    theta_gt = sb[:, 3] + theta_dot * self._dt
                    loss_consistency = nn.functional.mse_loss(theta_pred, theta_gt)

                # === Jacobian Frobenius norm (existing) ===
                loss_jacobian = torch.tensor(0.0)
                if self.config.lambda_jacobian > 0:
                    s_req = sb[:min(32, len(sb))].requires_grad_(True)
                    a_req = ab[:min(32, len(sb))].requires_grad_(True)
                    dsdt_j = self._model(s_req, a_req)
                    jac_norm = 0.0
                    for i in range(STATE_DIM):
                        grad = torch.autograd.grad(
                            dsdt_j[:, i].sum(), s_req, create_graph=True
                        )[0]
                        jac_norm = jac_norm + grad.pow(2).sum()
                    loss_jacobian = jac_norm / (STATE_DIM * min(32, len(sb)))

                # === Contractivity loss (NEW) ===
                loss_contractive = torch.tensor(0.0)
                max_eigs_batch = None
                all_eigs_batch = None
                if eff_lambda > 0:
                    # Use a subset for Jacobian computation (memory efficiency)
                    n_jac = min(32, len(sb))
                    s_con = sb[:n_jac].requires_grad_(True)
                    a_con = ab[:n_jac].requires_grad_(True)

                    loss_contractive, max_eigs_batch, all_eigs_batch = contractivity_loss(
                        self._model, s_con, a_con,
                        method=self.config.contractive_method,
                        n_power_steps=self.config.contractive_power_steps,
                    )

                # === Total loss ===
                loss = (loss_single
                        + self.config.lambda_multi * loss_multi
                        + self.config.lambda_consistency * loss_consistency
                        + self.config.lambda_jacobian * loss_jacobian
                        + eff_lambda * loss_contractive)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()

                epoch_loss += loss.item()
                epoch_contractive += loss_contractive.item()
                if max_eigs_batch is not None:
                    epoch_max_eig.extend(max_eigs_batch.cpu().tolist())
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_contractive = epoch_contractive / max(n_batches, 1)
            avg_max_eig = float(np.mean(epoch_max_eig)) if epoch_max_eig else 0.0
            n_positive = sum(1 for e in epoch_max_eig if e > 0) if epoch_max_eig else 0
            frac_positive = n_positive / max(len(epoch_max_eig), 1)

            self._training_log.append({
                'epoch': epoch,
                'loss': avg_loss,
                'contractive_loss': avg_contractive,
                'eff_lambda': eff_lambda,
                'max_eigenvalue_mean': avg_max_eig,
                'frac_positive_eigenvalues': frac_positive,
                'rollout_steps': rollout_steps,
                'lr': scheduler.get_last_lr()[0],
            })

            if epoch % 50 == 0 or epoch == self.config.n_epochs - 1:
                print(f"    Epoch {epoch}: loss={avg_loss:.6f}, "
                      f"contractive={avg_contractive:.6f}, "
                      f"lambda_eff={eff_lambda:.4f}, "
                      f"max_eig={avg_max_eig:.4f}, "
                      f"frac_pos={frac_positive:.3f}, "
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
        """Batch prediction for efficiency."""
        s_norm = torch.FloatTensor(states / self._state_std)
        a_norm = torch.FloatTensor(actions.reshape(-1, 1) / self._action_std)
        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).numpy()
        return states + dsdt_norm * self._delta_std * self._dt

    def predict_with_uncertainty(self, s, tau):
        return self.predict(s, tau), None

    def name(self):
        return 'neural_ode_contractive'

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
        self.config = ContractiveODEConfig(**checkpoint['config'])
        self._state_std = checkpoint['state_std']
        self._action_std = checkpoint['action_std']
        self._delta_std = checkpoint['delta_std']
        self._training_log = checkpoint.get('training_log', [])

        self._model = ODEFunc(
            self.config.hidden, self.config.depth, self.config.activation
        )
        self._model.load_state_dict(checkpoint['model_state'])
        self._model.eval()
