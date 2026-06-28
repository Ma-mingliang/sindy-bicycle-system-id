"""Conservative Ensemble Residual-MPPI with SINDy prior.

Key conservative features:
  - Ensemble world models (3 or 5 members) for uncertainty estimation
  - Small epsilon_max (0.1)
  - Strong residual penalty (lambda_res=2.0, lambda_smooth=5.0)
  - Multi-level safety checks before executing non-zero residual
  - Fallback to zero-residual unless clearly beneficial
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from reward_fn import compute_tracking_reward_torch


def _build_sindy_library(state_norm, action_norm):
    """Build SINDy polynomial library row (PyTorch)."""
    x = torch.cat([state_norm, action_norm], dim=-1)
    n = x.shape[-1]
    terms = [torch.ones(*x.shape[:-1], 1, device=x.device)]
    terms.append(x)
    for i in range(n):
        for j in range(i, n):
            terms.append((x[..., i] * x[..., j]).unsqueeze(-1))
    return torch.cat(terms, dim=-1)


class WorldModel(nn.Module):
    """SINDy-prior world model with NN residual."""

    def __init__(self, obs_dim, action_dim, mlp_dim, sindy_Xi, action_scale):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.register_buffer('sindy_Xi', torch.tensor(sindy_Xi, dtype=torch.float32))
        self.register_buffer('action_scale', torch.tensor(action_scale, dtype=torch.float32))

        self.nn_residual = nn.Sequential(
            nn.Linear(obs_dim + action_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, obs_dim),
        )

    def sindy_predict(self, s_norm, a):
        a_norm = a / (self.action_scale + 1e-8)
        theta_row = _build_sindy_library(s_norm, a_norm)
        return theta_row @ self.sindy_Xi

    def nn_predict(self, s_norm, a):
        return self.nn_residual(torch.cat([s_norm, a], dim=-1))

    def predict_next_state(self, s_norm, a):
        delta_sindy = self.sindy_predict(s_norm, a)
        delta_nn = self.nn_predict(s_norm, a)
        return s_norm + delta_sindy + delta_nn, delta_sindy, delta_nn


class EnsembleWorldModel:
    """Ensemble of WorldModel for uncertainty estimation."""

    def __init__(self, obs_dim, action_dim, mlp_dim, sindy_Xi, action_scale,
                 ensemble_size=3, device=None):
        self.ensemble_size = ensemble_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.models = nn.ModuleList([
            WorldModel(obs_dim, action_dim, mlp_dim, sindy_Xi, action_scale)
            for _ in range(ensemble_size)
        ]).to(self.device)

        self.optimizers = [
            torch.optim.Adam(m.parameters(), lr=3e-4) for m in self.models
        ]

    def predict_next_state(self, s_norm, a):
        """Predict next state with all ensemble members.

        Returns:
            mean_pred: mean prediction (batch, obs_dim)
            var_pred: prediction variance (batch, obs_dim)
            all_preds: list of per-member predictions
        """
        all_preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred, _, _ = model.predict_next_state(s_norm, a)
                all_preds.append(pred)

        stacked = torch.stack(all_preds, dim=0)  # (ensemble, batch, obs_dim)
        mean_pred = stacked.mean(dim=0)
        var_pred = stacked.var(dim=0)

        return mean_pred, var_pred, all_preds

    def train_member(self, member_idx, train_data, val_data, epochs=50,
                     batch_size=256, patience=10, grad_clip=10.0):
        """Train a single ensemble member with early stopping."""
        model = self.models[member_idx]
        optimizer = self.optimizers[member_idx]

        train_obs, train_action, train_next_obs = train_data
        val_obs, val_action, val_next_obs = val_data

        n_train = len(train_obs)
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        train_losses, val_losses = [], []

        model.train()
        for epoch in range(epochs):
            model.train()
            idx = torch.randperm(n_train)
            epoch_loss = 0
            n_batches = 0

            for i in range(0, n_train, batch_size):
                batch_idx = idx[i:i+batch_size]
                s = train_obs[batch_idx]
                a = train_action[batch_idx]
                s_next_real = train_next_obs[batch_idx]

                s_next_pred, _, _ = model.predict_next_state(s, a)
                loss = F.mse_loss(s_next_pred, s_next_real)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            train_losses.append(epoch_loss / max(n_batches, 1))

            model.eval()
            with torch.no_grad():
                s_next_pred, _, _ = model.predict_next_state(val_obs, val_action)
                val_loss = F.mse_loss(s_next_pred, val_next_obs).item()
                val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses),
        }

    def train(self, buffer, epochs=50, batch_size=256, val_ratio=0.2):
        """Train all ensemble members on bootstrap samples (with replacement)."""
        n = buffer.size
        all_obs = torch.tensor(buffer.obs[:n], device=self.device)
        all_action = torch.tensor(buffer.action[:n], device=self.device)
        all_next_obs = torch.tensor(buffer.next_obs[:n], device=self.device)

        n_val = int(n * val_ratio)
        n_train = n - n_val

        val_obs = all_obs[n_train:]
        val_action = all_action[n_train:]
        val_next_obs = all_next_obs[n_train:]

        results = []
        for i in range(self.ensemble_size):
            # True bootstrap: sample with replacement, different seed per member
            gen = torch.Generator()
            gen.manual_seed(42 + i * 1000)
            idx = torch.randint(0, n_train, (n_train,), generator=gen)
            train_obs = all_obs[idx]
            train_action = all_action[idx]
            train_next_obs = all_next_obs[idx]

            result = self.train_member(
                i,
                (train_obs, train_action, train_next_obs),
                (val_obs, val_action, val_next_obs),
                epochs=epochs, batch_size=batch_size,
            )
            results.append(result)
            print(f"    Member {i}: val_loss={result['best_val_loss']:.6f}, "
                  f"epochs={result['epochs_trained']}")

        return results

    def compute_ensemble_uncertainty(self, buffer):
        """Compute ensemble prediction uncertainty on buffer data."""
        n = buffer.size
        obs = torch.tensor(buffer.obs[:n], device=self.device)
        action = torch.tensor(buffer.action[:n], device=self.device)
        next_obs = torch.tensor(buffer.next_obs[:n], device=self.device)

        mean_pred, var_pred, _ = self.predict_next_state(obs, action)

        mse_mean = ((mean_pred - next_obs) ** 2).mean().item()
        avg_var = var_pred.mean().item()
        per_dim_var = var_pred.mean(dim=0).cpu().numpy()

        return {
            'mse_mean': mse_mean,
            'avg_variance': avg_var,
            'per_dim_variance': per_dim_var,
        }

    def save(self, path, extra_meta=None):
        meta = {
            'checkpoint_type': 'ensemble',
            'ensemble_size': self.ensemble_size,
            'state_dim': self.models[0].obs_dim,
            'action_dim': self.models[0].action_dim,
            'uses_sindy': True,
            'model_version': 2,
            'ensemble_state': self.models.state_dict(),
        }
        if extra_meta:
            meta.update(extra_meta)
        torch.save(meta, path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device, weights_only=False)

        # Schema validation
        required = ['checkpoint_type', 'ensemble_size', 'state_dim',
                     'action_dim', 'uses_sindy', 'model_version']
        missing = [k for k in required if k not in state_dict]
        if missing:
            raise ValueError(
                f"Checkpoint missing required fields: {missing}. "
                f"Found keys: {list(state_dict.keys())}. "
                f"This is likely an old single-model checkpoint. "
                f"Please retrain with the ensemble pipeline."
            )

        if state_dict['checkpoint_type'] != 'ensemble':
            raise ValueError(
                f"Checkpoint type is '{state_dict['checkpoint_type']}', "
                f"expected 'ensemble'. Retrain required."
            )
        if state_dict['ensemble_size'] != self.ensemble_size:
            raise ValueError(
                f"Checkpoint ensemble_size={state_dict['ensemble_size']}, "
                f"expected {self.ensemble_size}. Retrain required."
            )
        if state_dict['state_dim'] != self.models[0].obs_dim:
            raise ValueError(
                f"Checkpoint state_dim={state_dict['state_dim']}, "
                f"expected {self.models[0].obs_dim}."
            )

        self.models.load_state_dict(state_dict['ensemble_state'])
        print(f"  Loaded ensemble checkpoint: {state_dict['ensemble_size']} members, "
              f"v{state_dict['model_version']}")


def denormalize_state(s_norm):
    """Convert normalized 8D state to raw state."""
    scales = torch.tensor([10.0, 1.57, 5.0, 1.57, 10.0, 1.0/8.0, 0.785, 3.0],
                          device=s_norm.device)
    return s_norm * scales


def compute_analytic_reward(s_raw, u_total):
    """Legacy: kept for backward compat. Use compute_tracking_reward_torch instead."""
    ey = s_raw[:, 0]
    epsi = s_raw[:, 1]
    theta = s_raw[:, 3]
    theta_dot = s_raw[:, 4]
    reward = -(ey**2 + 0.5 * epsi**2 + 0.3 * theta**2 + 0.1 * theta_dot**2 + 0.01 * u_total**2)
    return reward


class ConservativeEnsembleResidualMPPI:
    """Conservative Ensemble Residual-MPPI agent.

    Multi-level safety:
      1. Ensemble uncertainty threshold
      2. Small epsilon_max (0.1)
      3. Strong residual penalty
      4. Fallback unless best_G > zero_G + margin
      5. One-step safety check
      6. Risk check: |ey| and |theta| must not worsen
    """

    def __init__(self, cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mlp_dim = cfg.get('mlp_dim', 256)

        self.horizon = cfg.get('horizon', 5)
        self.num_samples = cfg.get('num_samples', 128)
        self.num_elites = cfg.get('num_elites', 16)
        self.iterations = cfg.get('iterations', 6)
        self.temperature = cfg.get('temperature', 0.5)
        self.epsilon_max = cfg.get('epsilon_max', 0.1)
        self.epsilon_std = cfg.get('epsilon_std', 0.08)
        self.gamma = cfg.get('gamma', 0.95)
        self.lambda_res = cfg.get('lambda_res', 2.0)
        self.lambda_smooth = cfg.get('lambda_smooth', 5.0)
        self.lambda_uncertainty = cfg.get('lambda_uncertainty', 5.0)
        self.uncertainty_threshold = cfg.get('uncertainty_threshold', None)  # None = use val-based
        self.margin = cfg.get('margin', 0.1)
        self.margin_gate_off = cfg.get('margin_gate_off', False)
        self.forced_epsilon = cfg.get('forced_epsilon', None)  # None = use MPPI
        self.action_dim = action_dim

        # Risk gate parameters
        # Modes: 'strict' (no tolerance), 'tolerance' (additive tolerance), 'hard_limit_only'
        self.risk_mode = cfg.get('risk_mode', 'strict')
        self.ey_tol = cfg.get('ey_tol', 0.0)
        self.theta_tol = cfg.get('theta_tol', 0.0)
        # Hard limits: fallback if predicted |ey| or |theta| exceeds these
        self.ey_hard_limit = cfg.get('ey_hard_limit', 4.0)   # termination is 5.0
        self.theta_hard_limit = cfg.get('theta_hard_limit', 1.2)  # termination is pi/2 ≈ 1.57

        # Fallback reason counters (reset each episode)
        self._fb_reason = {
            'one_step_reward': 0,  # diagnostic only (not a hard gate)
            'risk': 0,
            'uncertainty': 0,
            'margin': 0,
            'total_steps': 0,
            'nonfallback': 0,
            'eps_abs_sum': 0.0,
            'eps_abs_max': 0.0,
            'clip_count': 0,  # eps hitting epsilon_max
            'predicted_adv_sum': 0.0,
            # Per-step diagnostic (summed, divided by total_steps for mean)
            'step_adv_raw_sum': 0.0,
            'step_adv_pen_sum': 0.0,
            'step_zero_G_sum': 0.0,
            'step_best_G_raw_sum': 0.0,
            'step_best_G_pen_sum': 0.0,
        }

        ensemble_size = cfg.get('ensemble_size', 3)
        self.ensemble = EnsembleWorldModel(
            obs_dim, action_dim, mlp_dim, sindy_Xi, action_scale,
            ensemble_size=ensemble_size, device=self.device,
        )

        self._prev_mean = torch.zeros(self.horizon, action_dim, device=self.device)

    @torch.no_grad()
    def act(self, obs_norm, eval_mode=False):
        """Conservative MPPI planning with ensemble uncertainty and multi-level safety.

        Returns:
            epsilon: residual action (may be zero if safety checks fail)
            best_G: predicted return of best action
            zero_G: predicted return of zero action
        """
        obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        s_norm = obs_t

        # Forced epsilon mode: skip MPPI, return fixed residual
        if self.forced_epsilon is not None:
            eps = np.array([self.forced_epsilon])
            return eps, 0.0, 0.0, 0.0

        # === Evaluate zero-residual first ===
        zero_G, zero_next_ey, zero_next_theta, zero_uncertainty = \
            self._evaluate_zero_residual(s_norm)
        zero_next_reward = self._compute_zero_next_reward(s_norm)

        # === MPPI planning ===
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.full((self.horizon, self.action_dim), self.epsilon_std, device=self.device)
        mean[:-1] = self._prev_mean[1:]

        for _iter in range(self.iterations):
            r = torch.randn(self.num_samples, self.horizon, self.action_dim, device=self.device)
            epsilon = mean.unsqueeze(0) + std.unsqueeze(0) * r
            epsilon = epsilon.clamp(-self.epsilon_max, self.epsilon_max)

            G = torch.zeros(self.num_samples, device=self.device)
            U = torch.zeros(self.num_samples, device=self.device)  # uncertainty sum
            _s_norm = s_norm.repeat(self.num_samples, 1)
            discount = 1.0

            for t in range(self.horizon):
                a_t = epsilon[:, t, :]
                prev_raw = denormalize_state(_s_norm)
                s_next_mean, s_next_var, _ = self.ensemble.predict_next_state(_s_norm, a_t)
                next_raw = denormalize_state(s_next_mean)
                reward = compute_tracking_reward_torch(prev_raw, next_raw)

                step_uncertainty = s_next_var.sum(dim=-1)  # (num_samples,)
                G = G + discount * reward
                U = U + discount * step_uncertainty

                discount *= self.gamma
                _s_norm = s_next_mean

            # Total objective with uncertainty penalty
            penalty_res = self.lambda_res * (epsilon ** 2).sum(dim=1).squeeze(-1)
            penalty_smooth = self.lambda_smooth * ((epsilon[:, 1:] - epsilon[:, :-1]) ** 2).sum(dim=(1, 2))
            penalty_unc = self.lambda_uncertainty * U
            G_total = G - penalty_res - penalty_smooth - penalty_unc
            # Keep raw G for fair margin gate comparison
            G_raw = G

            elite_idx = torch.topk(G_total, self.num_elites, dim=0).indices
            elite_actions = epsilon[elite_idx]
            elite_value = G_total[elite_idx]

            max_val = elite_value.max()
            score = torch.exp(self.temperature * (elite_value - max_val))
            score = score / (score.sum() + 1e-9)

            mean = (score.unsqueeze(-1).unsqueeze(-1) * elite_actions).sum(dim=0)
            mean = mean / (score.sum() + 1e-9)
            mean = mean.clamp(-self.epsilon_max, self.epsilon_max)

            var = (score.unsqueeze(-1).unsqueeze(-1) * (elite_actions - mean.unsqueeze(0)) ** 2).sum(dim=0)
            var = var / (score.sum() + 1e-9)
            std = var.sqrt().clamp(0.01, self.epsilon_std)

        self._prev_mean.copy_(mean)

        # === Safety checks ===
        best_eps = mean[0:1, :]
        self._fb_reason['total_steps'] += 1

        # Pre-compute best G values (needed by all gates)
        best_G_total_val = G_total[elite_idx[0]].item()
        best_G_raw_val = G_raw[elite_idx[0]].item()
        predicted_adv_raw = best_G_raw_val - zero_G
        predicted_adv_penalized = best_G_total_val - zero_G

        # Check 1: one-step safety (DIAGNOSTIC ONLY — does not trigger fallback)
        s_next_mean, s_next_var, _ = self.ensemble.predict_next_state(s_norm, best_eps)
        prev_raw = denormalize_state(s_norm)
        s_next_raw = denormalize_state(s_next_mean)
        eps_next_reward = compute_tracking_reward_torch(prev_raw, s_next_raw)
        eps_uncertainty = s_next_var.mean().item()

        if eps_next_reward.item() < zero_next_reward:
            self._fb_reason['one_step_reward'] += 1
            # NOT a hard gate — continue to next checks

        # Check 2: risk gate (HARD)
        eps_ey = abs(s_next_raw[0, 0].item())
        eps_theta = abs(s_next_raw[0, 3].item())

        risk_triggered = False
        if self.risk_mode == 'strict':
            # Original: any worsening triggers fallback
            if eps_ey > zero_next_ey or eps_theta > zero_next_theta:
                risk_triggered = True
        elif self.risk_mode == 'tolerance':
            # Allow small worsening within tolerance
            if eps_ey > zero_next_ey + self.ey_tol:
                risk_triggered = True
            if eps_theta > zero_next_theta + self.theta_tol:
                risk_triggered = True
        elif self.risk_mode == 'hard_limit_only':
            # Only check if approaching termination threshold
            if eps_ey > self.ey_hard_limit or eps_theta > self.theta_hard_limit:
                risk_triggered = True

        if risk_triggered:
            self._fb_reason['risk'] += 1
            return np.zeros(self.action_dim), best_G_raw_val, zero_G, eps_uncertainty

        # Check 3: uncertainty gate — ensemble uncertainty > threshold (HARD)
        threshold = self.uncertainty_threshold if self.uncertainty_threshold is not None else 1e6
        if eps_uncertainty > threshold:
            self._fb_reason['uncertainty'] += 1
            return np.zeros(self.action_dim), best_G_raw_val, zero_G, eps_uncertainty

        # Check 4: margin gate — best raw G must exceed zero raw G + margin (HARD)
        if not self.margin_gate_off and best_G_raw_val <= zero_G + self.margin:
            self._fb_reason['margin'] += 1
            # Log per-step diagnostic even on fallback
            self._fb_reason['step_adv_raw_sum'] += predicted_adv_raw
            self._fb_reason['step_adv_pen_sum'] += predicted_adv_penalized
            self._fb_reason['step_zero_G_sum'] += zero_G
            self._fb_reason['step_best_G_raw_sum'] += best_G_raw_val
            self._fb_reason['step_best_G_pen_sum'] += best_G_total_val
            return np.zeros(self.action_dim), best_G_raw_val, zero_G, eps_uncertainty

        # All hard checks passed — execute epsilon
        if eval_mode:
            eps = mean[0]
        else:
            eps = mean[0] + std[0] * torch.randn(self.action_dim, device=self.device)
        eps = eps.clamp(-self.epsilon_max, self.epsilon_max)

        # Track non-fallback statistics
        self._fb_reason['nonfallback'] += 1
        eps_abs = eps.abs().item()
        self._fb_reason['eps_abs_sum'] += eps_abs
        self._fb_reason['eps_abs_max'] = max(self._fb_reason['eps_abs_max'], eps_abs)
        if eps_abs >= self.epsilon_max - 0.001:
            self._fb_reason['clip_count'] += 1
        self._fb_reason['predicted_adv_sum'] += predicted_adv_raw
        # Per-step diagnostic
        self._fb_reason['step_adv_raw_sum'] += predicted_adv_raw
        self._fb_reason['step_adv_pen_sum'] += predicted_adv_penalized
        self._fb_reason['step_zero_G_sum'] += zero_G
        self._fb_reason['step_best_G_raw_sum'] += best_G_raw_val
        self._fb_reason['step_best_G_pen_sum'] += best_G_total_val

        return eps.cpu().numpy(), best_G_raw_val, zero_G, eps_uncertainty

    @torch.no_grad()
    def _evaluate_zero_residual(self, s_norm):
        """Evaluate zero-residual rollout, return G, next |ey|, |theta|, uncertainty."""
        zero_a = torch.zeros(1, self.action_dim, device=self.device)
        G = torch.zeros(1, device=self.device)
        U = torch.zeros(1, device=self.device)
        _s = s_norm.clone()
        discount = 1.0
        next_ey, next_theta = 0.0, 0.0

        for t in range(self.horizon):
            prev_raw = denormalize_state(_s)
            s_next_mean, s_next_var, _ = self.ensemble.predict_next_state(_s, zero_a)
            next_raw = denormalize_state(s_next_mean)
            reward = compute_tracking_reward_torch(prev_raw, next_raw)
            step_unc = s_next_var.sum(dim=-1)
            G = G + discount * reward
            U = U + discount * step_unc
            if t == 0:
                next_ey = abs(next_raw[0, 0].item())
                next_theta = abs(next_raw[0, 3].item())
            discount *= self.gamma
            _s = s_next_mean

        return G.item(), next_ey, next_theta, U.item()

    @torch.no_grad()
    def _compute_zero_next_reward(self, s_norm):
        """Compute one-step reward for zero action."""
        zero_a = torch.zeros(1, self.action_dim, device=self.device)
        prev_raw = denormalize_state(s_norm)
        s_next_mean, _, _ = self.ensemble.predict_next_state(s_norm, zero_a)
        next_raw = denormalize_state(s_next_mean)
        return compute_tracking_reward_torch(prev_raw, next_raw).item()

    def reset_planning(self):
        self._prev_mean.zero_()
        self._fb_reason = {k: 0 for k in self._fb_reason}

    def get_fb_reasons(self):
        """Return fallback reason counts and diagnostics for current episode."""
        total = self._fb_reason['total_steps']
        nf = self._fb_reason['nonfallback']
        if total == 0:
            return {k: 0 for k in self._fb_reason}
        result = {}
        for k, v in self._fb_reason.items():
            if k in ('total_steps', 'nonfallback', 'eps_abs_sum', 'eps_abs_max',
                      'clip_count', 'predicted_adv_sum',
                      'step_adv_raw_sum', 'step_adv_pen_sum',
                      'step_zero_G_sum', 'step_best_G_raw_sum', 'step_best_G_pen_sum'):
                result[k] = v
            else:
                result[k] = v / total
        result['eps_abs_mean'] = self._fb_reason['eps_abs_sum'] / max(nf, 1)
        result['clip_rate'] = self._fb_reason['clip_count'] / max(nf, 1)
        result['predicted_adv_mean'] = (self._fb_reason['predicted_adv_sum'] / max(nf, 1)
                                         if nf > 0 else 0.0)
        # Per-step means (over ALL steps, not just non-fallback)
        result['step_adv_raw_mean'] = self._fb_reason['step_adv_raw_sum'] / max(total, 1)
        result['step_adv_pen_mean'] = self._fb_reason['step_adv_pen_sum'] / max(total, 1)
        result['step_zero_G_mean'] = self._fb_reason['step_zero_G_sum'] / max(total, 1)
        result['step_best_G_raw_mean'] = self._fb_reason['step_best_G_raw_sum'] / max(total, 1)
        result['step_best_G_pen_mean'] = self._fb_reason['step_best_G_pen_sum'] / max(total, 1)
        return result

    def train_world_model(self, buffer, epochs=50, batch_size=256, val_ratio=0.2):
        """Train ensemble world models."""
        results = self.ensemble.train(buffer, epochs=epochs, batch_size=batch_size,
                                       val_ratio=val_ratio)
        return results

    def compute_model_errors(self, buffer):
        """Compute model errors using ensemble mean."""
        n = buffer.size
        obs = torch.tensor(buffer.obs[:n], device=self.device)
        action = torch.tensor(buffer.action[:n], device=self.device)
        next_obs = torch.tensor(buffer.next_obs[:n], device=self.device)

        mean_pred, var_pred, _ = self.ensemble.predict_next_state(obs, action)
        mse_mean = ((mean_pred - next_obs) ** 2).mean().item()
        avg_var = var_pred.mean().item()
        per_dim_mse = ((mean_pred - next_obs) ** 2).mean(dim=0).cpu().numpy()
        per_dim_var = var_pred.mean(dim=0).cpu().numpy()

        # SINDy-only error (from first member)
        sindy_pred = obs + self.ensemble.models[0].sindy_predict(obs, action)
        sindy_error = ((sindy_pred - next_obs) ** 2).mean().item()
        sindy_per_dim = ((sindy_pred - next_obs) ** 2).mean(dim=0).cpu().numpy()

        return {
            'sindy_error': sindy_error,
            'model_error': mse_mean,
            'avg_uncertainty': avg_var,
            'sindy_per_dim': sindy_per_dim,
            'model_per_dim': per_dim_mse,
            'uncertainty_per_dim': per_dim_var,
        }

    def compute_val_uncertainty_percentiles(self, buffer, val_ratio=0.2):
        """Compute uncertainty percentiles on validation set for threshold calibration."""
        n = buffer.size
        n_val = int(n * val_ratio)
        obs = torch.tensor(buffer.obs[n - n_val:], device=self.device)
        action = torch.tensor(buffer.action[n - n_val:], device=self.device)

        _, var_pred, _ = self.ensemble.predict_next_state(obs, action)
        # Per-sample mean uncertainty across state dims
        unc_per_sample = var_pred.mean(dim=-1).cpu().numpy()

        p50 = float(np.percentile(unc_per_sample, 50))
        p90 = float(np.percentile(unc_per_sample, 90))
        p95 = float(np.percentile(unc_per_sample, 95))
        p99 = float(np.percentile(unc_per_sample, 99))

        return {'p50': p50, 'p90': p90, 'p95': p95, 'p99': p99,
                'mean': float(unc_per_sample.mean()),
                'std': float(unc_per_sample.std())}

    def compute_train_uncertainty_percentiles(self, buffer, val_ratio=0.2):
        """Compute uncertainty percentiles on training set."""
        n = buffer.size
        n_val = int(n * val_ratio)
        n_train = n - n_val
        obs = torch.tensor(buffer.obs[:n_train], device=self.device)
        action = torch.tensor(buffer.action[:n_train], device=self.device)

        _, var_pred, _ = self.ensemble.predict_next_state(obs, action)
        unc_per_sample = var_pred.mean(dim=-1).cpu().numpy()

        p50 = float(np.percentile(unc_per_sample, 50))
        p90 = float(np.percentile(unc_per_sample, 90))
        p95 = float(np.percentile(unc_per_sample, 95))
        p99 = float(np.percentile(unc_per_sample, 99))

        return {'p50': p50, 'p90': p90, 'p95': p95, 'p99': p99,
                'mean': float(unc_per_sample.mean()),
                'std': float(unc_per_sample.std())}

    def validate_rollout(self, buffer, horizon=5, n_samples=100):
        """Validate rollout with ensemble uncertainty."""
        n = buffer.size
        idx = np.random.randint(0, n - horizon, size=n_samples)

        one_step_errors = []
        multi_step_errors = []
        multi_step_uncertainties = []

        for i in idx:
            s = torch.tensor(buffer.obs[i:i+1], device=self.device)
            s_real_seq = torch.tensor(buffer.next_obs[i:i+horizon], device=self.device)
            actions = torch.tensor(buffer.action[i:i+horizon], device=self.device)

            s_next_mean, s_next_var, _ = self.ensemble.predict_next_state(s, actions[0:1])
            one_step_err = ((s_next_mean - s_real_seq[0:1]) ** 2).mean().item()
            one_step_errors.append(one_step_err)

            _s = s.clone()
            rollout_errors = []
            rollout_uncertainties = []
            for t in range(horizon):
                _s_mean, _s_var, _ = self.ensemble.predict_next_state(
                    _s, actions[t:t+1])
                err = ((_s_mean - s_real_seq[t:t+1]) ** 2).mean().item()
                unc = _s_var.mean().item()
                rollout_errors.append(err)
                rollout_uncertainties.append(unc)
                _s = _s_mean

            multi_step_errors.append(rollout_errors)
            multi_step_uncertainties.append(rollout_uncertainties)

        one_step_mean = np.mean(one_step_errors)
        multi_step_mean = np.mean(multi_step_errors, axis=0)
        multi_step_unc_mean = np.mean(multi_step_uncertainties, axis=0)

        diverges = False
        if len(multi_step_mean) > 2:
            if multi_step_mean[-1] > 0.5:
                diverges = True
            ratios = [multi_step_mean[i+1] / max(multi_step_mean[i], 1e-8)
                      for i in range(len(multi_step_mean)-1)]
            if len(ratios) >= 3 and ratios[-1] > 5.0 * max(ratios[0], 0.1):
                diverges = True

        return {
            'one_step_error': one_step_mean,
            'multi_step_error': multi_step_mean,
            'multi_step_uncertainty': multi_step_unc_mean,
            'diverges': diverges,
        }

    def save(self, path, extra_meta=None):
        self.ensemble.save(path, extra_meta=extra_meta)

    def load(self, path):
        self.ensemble.load(path)
