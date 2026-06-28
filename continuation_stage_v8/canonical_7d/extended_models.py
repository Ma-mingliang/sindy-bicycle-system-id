"""Extended models for 7D: GP Ensemble, Adaptive Residual, NeuralODE.

These are methods from earlier versions that showed promise but were
not carried into V6/V8 canonical packages. Retested here on 7D real data.
"""
import numpy as np
import math
from typing import Optional, Tuple
from .models import BaseModel, GPModel
from .config import STATE_DIM

# Lazy torch import
_torch = None

def _get_torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch


# ============================================================
# 1. GP Bootstrap Ensemble
# ============================================================
class GPEnsemble(BaseModel):
    """Bootstrap GP Ensemble: multiple GPs on bootstrap samples.

    Reduces variance by averaging predictions from N GPs,
    each trained on a different bootstrap sample of the data.
    """

    def __init__(self, max_samples: int = 2000, n_models: int = 5, seed: int = 42):
        self.max_samples = max_samples
        self.n_models = n_models
        self.seed = seed
        self._ensembles = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        self._state_std = state_std.copy()
        self._action_std = action_std
        self._delta_std = delta_std.copy()
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0

        rng = np.random.RandomState(self.seed)
        n = len(states)
        self._ensembles = []

        for i in range(self.n_models):
            if n > self.max_samples:
                idx = rng.choice(n, self.max_samples, replace=True)
            else:
                idx = rng.choice(n, n, replace=True)

            X = np.column_stack([states[idx] / self._state_std,
                                 actions[idx].reshape(-1, 1) / action_std])
            Y = deltas[idx] / self._delta_std

            gps = []
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            for col in range(STATE_DIM):
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
                gp.fit(X, Y[:, col])
                gps.append(gp)
            self._ensembles.append(gps)

    def predict(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)

        preds = []
        for gps in self._ensembles:
            delta_norm = np.array([gp.predict(x)[0] for gp in gps])
            preds.append(delta_norm)
        return s + np.mean(preds, axis=0) * self._delta_std

    def predict_with_uncertainty(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)

        preds = []
        for gps in self._ensembles:
            delta_norm = np.array([gp.predict(x)[0] for gp in gps])
            preds.append(delta_norm)
        preds = np.array(preds)
        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)
        s_next = s + mean * self._delta_std
        uncertainty = std * self._delta_std
        return s_next, uncertainty

    def name(self):
        return f'gp_ensemble_{self.n_models}'


# ============================================================
# 2. Adaptive Residual GP (OOD-aware scaling)
# ============================================================
class AdaptiveResidualGP(BaseModel):
    """GP + NN residual with OOD-aware adaptive scaling.

    From V4 Round 1 improvement: uses Mahalanobis-like distance
    to detect OOD inputs and reduce residual weight accordingly.

    sigmoid scale: scale = base_scale / (1 + exp(k * (ood_dist - threshold)))
    """

    def __init__(self, n_models: int = 5, n_epochs: int = 50,
                 base_scale: float = 0.5, ood_threshold: float = 2.0,
                 ood_k: float = 3.0):
        self.n_models = n_models
        self.n_epochs = n_epochs
        self.base_scale = base_scale
        self.ood_threshold = ood_threshold
        self.ood_k = ood_k
        self._baseline = None
        self._ensemble = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None
        self._train_mean = None
        self._train_cov_inv = None

    def _compute_ood_distance(self, s_norm, a_norm):
        """Compute Mahalanobis-like distance from training distribution."""
        x = np.concatenate([s_norm, [a_norm]])
        diff = x - self._train_mean
        dist = np.sqrt(diff @ self._train_cov_inv @ diff)
        return dist

    def _adaptive_scale(self, ood_dist):
        """Sigmoid scaling: large OOD → small scale."""
        clipped = max(-10, min(10, self.ood_k * (ood_dist - self.ood_threshold)))
        return self.base_scale / (1.0 + math.exp(clipped))

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        from .residual_models import NNEnsemble7D

        self._state_std = state_std.copy()
        self._action_std = action_std
        self._delta_std = delta_std.copy()
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0

        self._baseline = GPModel(max_samples=2000, n_restarts=2)
        self._baseline.train(states, actions, deltas, state_std, action_std, delta_std)

        # Compute training distribution statistics
        train_inputs = np.column_stack([
            states / self._state_std,
            actions.reshape(-1, 1) / action_std
        ])
        self._train_mean = np.mean(train_inputs, axis=0)
        cov = np.cov(train_inputs.T) + np.eye(train_inputs.shape[1]) * 1e-6
        self._train_cov_inv = np.linalg.inv(cov)

        # Train NN residual
        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_base = self._baseline.predict(states[i], actions[i])
            residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std

        self._ensemble = NNEnsemble7D(self.n_models, self.n_epochs)
        self._ensemble.train(train_inputs, residuals)

    def predict(self, s, tau):
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        delta_nn = self._ensemble.predict_mean(s_norm, a_norm)

        ood_dist = self._compute_ood_distance(s_norm, a_norm)
        scale = self._adaptive_scale(ood_dist)

        return s_next_base + delta_nn * self._delta_std * scale

    def predict_with_uncertainty(self, s, tau):
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        mean_nn, std_nn = self._ensemble.predict_mean_std(s_norm, a_norm)

        ood_dist = self._compute_ood_distance(s_norm, a_norm)
        scale = self._adaptive_scale(ood_dist)

        s_next = s_next_base + mean_nn * self._delta_std * scale
        uncertainty = std_nn * self._delta_std * scale
        return s_next, uncertainty

    def name(self):
        return 'adaptive_residual_gp'


# ============================================================
# 3. NeuralODE (Euler integration)
# ============================================================
class NeuralODEModel(BaseModel):
    """Neural ODE: NN learns ds/dt, integrated via RK4.

    Instead of predicting delta_s directly, the NN learns the
    continuous-time derivative, which is then integrated.
    """

    def __init__(self, n_epochs: int = 100, lr: float = 1e-3, hidden: int = 128):
        self.n_epochs = n_epochs
        self.lr = lr
        self.hidden = hidden
        self._model = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None
        self._dt = 1.0 / 30.0

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        torch = _get_torch()
        import torch.nn as nn

        self._state_std = state_std.copy()
        self._action_std = action_std
        self._delta_std = delta_std.copy()
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0

        class ODEFunc(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(STATE_DIM + 1, hidden), nn.Tanh(),
                    nn.Linear(hidden, hidden), nn.Tanh(),
                    nn.Linear(hidden, STATE_DIM),
                )

            def forward(self, s, a):
                return self.net(torch.cat([s, a], dim=-1))

        self._model = ODEFunc(self.hidden)

        # Train: input (s, tau), target = ds/dt = delta / dt
        train_s = torch.FloatTensor(states / self._state_std)
        train_a = torch.FloatTensor(actions.reshape(-1, 1) / self._action_std)
        # Target: ds/dt in normalized space
        train_dsdot = torch.FloatTensor(deltas / (self._delta_std * self._dt))

        ds = torch.utils.data.TensorDataset(train_s, train_a, train_dsdot)
        loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        crit = torch.nn.MSELoss()

        self._model.train()
        for epoch in range(self.n_epochs):
            for sb, ab, yb in loader:
                pred = self._model(sb, ab)
                loss = crit(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._model.eval()

    def predict(self, s, tau):
        torch = _get_torch()
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0)

        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).numpy()[0]

        # Integrate: s_next = s + dsdt * dt * delta_std (in original space)
        dsdt = dsdt_norm * self._delta_std * self._dt
        return s + dsdt

    def predict_with_uncertainty(self, s, tau):
        return self.predict(s, tau), None

    def name(self):
        return 'neural_ode'


# ============================================================
# 4. NeuralODE with Multi-step Loss
# ============================================================
class NeuralODEMultiStep(BaseModel):
    """Neural ODE with multi-step rollout loss training.

    From V4 Round 3 improvement: adds multi-step rollout loss
    to single-step loss, forcing the NN to learn long-term dynamics.
    """

    def __init__(self, n_epochs: int = 100, lr: float = 1e-3, hidden: int = 128,
                 rollout_steps: int = 5, rollout_weight: float = 0.5):
        self.n_epochs = n_epochs
        self.lr = lr
        self.hidden = hidden
        self.rollout_steps = rollout_steps
        self.rollout_weight = rollout_weight
        self._model = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None
        self._dt = 1.0 / 30.0

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        torch = _get_torch()
        import torch.nn as nn

        self._state_std = state_std.copy()
        self._action_std = action_std
        self._delta_std = delta_std.copy()
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0

        class ODEFunc(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(STATE_DIM + 1, hidden), nn.Tanh(),
                    nn.Linear(hidden, hidden), nn.Tanh(),
                    nn.Linear(hidden, STATE_DIM),
                )

            def forward(self, s, a):
                return self.net(torch.cat([s, a], dim=-1))

        self._model = ODEFunc(self.hidden)

        train_s = torch.FloatTensor(states / self._state_std)
        train_a = torch.FloatTensor(actions.reshape(-1, 1) / self._action_std)
        train_dsdot = torch.FloatTensor(deltas / (self._delta_std * self._dt))

        ds = torch.utils.data.TensorDataset(train_s, train_a, train_dsdot)
        loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        crit = torch.nn.MSELoss()

        self._model.train()
        for epoch in range(self.n_epochs):
            for sb, ab, yb in loader:
                # Single-step loss
                pred = self._model(sb, ab)
                loss_single = crit(pred, yb)

                # Multi-step rollout loss
                loss_multi = torch.tensor(0.0)
                if self.rollout_steps > 1 and len(sb) > self.rollout_steps:
                    s_cur = sb[:len(sb) - self.rollout_steps].clone()
                    for step in range(self.rollout_steps):
                        idx = torch.arange(len(s_cur))
                        a_cur = ab[idx + step]
                        dsdt = self._model(s_cur, a_cur)
                        s_cur = s_cur + dsdt * self._dt
                        # Compare with target
                        target_idx = idx + step + 1
                        if target_idx.max() < len(sb):
                            target_s = sb[target_idx]
                            loss_multi = loss_multi + crit(s_cur, target_s)

                loss = loss_single + self.rollout_weight * loss_multi
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._model.eval()

    def predict(self, s, tau):
        torch = _get_torch()
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0)

        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).numpy()[0]

        dsdt = dsdt_norm * self._delta_std * self._dt
        return s + dsdt

    def predict_with_uncertainty(self, s, tau):
        return self.predict(s, tau), None

    def name(self):
        return 'neural_ode_multistep'
