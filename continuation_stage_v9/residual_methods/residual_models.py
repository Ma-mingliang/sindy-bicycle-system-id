"""V9 Residual methods for Neural ODE.

All residual methods wrap a frozen Neural ODE base model and add
a learned residual correction on top.
"""
import numpy as np
import os
from typing import Optional, List

import torch
import torch.nn as nn

STATE_DIM = 7
ACTION_DIM = 1


class _ResidualNet(nn.Module):
    """Simple residual network."""

    def __init__(self, input_dim, output_dim, hidden=64, depth=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, output_dim))
        nn.init.zeros_(layers[-1].bias)
        nn.init.xavier_uniform_(layers[-1].weight, gain=0.01)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _GatingNet(nn.Module):
    """Gating network: outputs alpha in [0, 1]."""

    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def _train_residual_net(model, X, Y, n_epochs=80, lr=1e-3, batch_size=256, seed=42):
    """Train a residual network."""
    torch.manual_seed(seed)
    Xt = torch.FloatTensor(X)
    Yt = torch.FloatTensor(Y)
    ds = torch.utils.data.TensorDataset(Xt, Yt)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()


class StateResidual:
    """Residual on state: r = x_{t+1}^true - x_{t+1}^NODE."""

    def __init__(self, base_model, state_std, action_std, delta_std,
                 residual_scale=0.1, n_epochs=80, hidden=64, seed=42):
        self.base = base_model
        self.state_std = state_std.copy()
        self.action_std = action_std
        self.delta_std = delta_std.copy()
        self.state_std[self.state_std < 1e-10] = 1.0
        self.delta_std[self.delta_std < 1e-10] = 1.0
        self.scale = residual_scale
        self.seed = seed
        self._net = _ResidualNet(STATE_DIM + ACTION_DIM, STATE_DIM, hidden)
        self._n_epochs = n_epochs
        self._step_count = 0

    def train(self, states, actions, deltas):
        # Compute residuals: true_next - ode_next
        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_ode = self.base.predict(states[i], actions[i])
            s_next_true = states[i] + deltas[i]
            residuals[i] = (s_next_true - s_next_ode) / self.delta_std

        X = np.column_stack([states / self.state_std,
                             actions.reshape(-1, 1) / self.action_std])
        _train_residual_net(self._net, X, residuals,
                            n_epochs=self._n_epochs, seed=self.seed)

    def predict(self, s, tau):
        s_next_ode = self.base.predict(s, tau)
        x = np.concatenate([s / self.state_std, [tau / self.action_std]]).reshape(1, -1)
        with torch.no_grad():
            r = self._net(torch.FloatTensor(x)).numpy()[0]
        return s_next_ode + r * self.delta_std * self.scale

    def reset(self):
        self._step_count = 0


class DerivativeResidual:
    """Residual on derivative: r = (x_{t+1}-x_t)/dt - f_NODE(x_t, u_t)."""

    def __init__(self, base_model, state_std, action_std, delta_std,
                 residual_scale=0.1, n_epochs=80, hidden=64, seed=42, dt=1/30):
        self.base = base_model
        self.state_std = state_std.copy()
        self.action_std = action_std
        self.delta_std = delta_std.copy()
        self.state_std[self.state_std < 1e-10] = 1.0
        self.delta_std[self.delta_std < 1e-10] = 1.0
        self.scale = residual_scale
        self.dt = dt
        self.seed = seed
        self._net = _ResidualNet(STATE_DIM + ACTION_DIM, STATE_DIM, hidden)
        self._n_epochs = n_epochs

    def train(self, states, actions, deltas):
        # Compute derivative residuals
        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_ode = self.base.predict(states[i], actions[i])
            delta_ode = s_next_ode - states[i]
            dsdt_true = deltas[i] / self.dt
            dsdt_ode = delta_ode / self.dt
            residuals[i] = (dsdt_true - dsdt_ode) / (self.delta_std / self.dt)

        X = np.column_stack([states / self.state_std,
                             actions.reshape(-1, 1) / self.action_std])
        _train_residual_net(self._net, X, residuals,
                            n_epochs=self._n_epochs, seed=self.seed)

    def predict(self, s, tau):
        s_next_ode = self.base.predict(s, tau)
        x = np.concatenate([s / self.state_std, [tau / self.action_std]]).reshape(1, -1)
        with torch.no_grad():
            r = self._net(torch.FloatTensor(x)).numpy()[0]
        correction = r * (self.delta_std / self.dt) * self.dt * self.scale
        return s_next_ode + correction

    def reset(self):
        pass


class MultiStepEndpointResidual:
    """Residual trained on multi-step endpoint error."""

    def __init__(self, base_model, state_std, action_std, delta_std,
                 H=10, residual_scale=0.1, n_epochs=80, hidden=64, seed=42):
        self.base = base_model
        self.state_std = state_std.copy()
        self.action_std = action_std
        self.delta_std = delta_std.copy()
        self.state_std[self.state_std < 1e-10] = 1.0
        self.delta_std[self.delta_std < 1e-10] = 1.0
        self.H = H
        self.scale = residual_scale
        self.seed = seed
        self._net = _ResidualNet(STATE_DIM + ACTION_DIM, STATE_DIM, hidden)
        self._n_epochs = n_epochs

    def train(self, states, actions, deltas):
        residuals = []
        X_list = []
        n = len(states)
        for i in range(0, n - self.H, self.H):
            s_cur = states[i].copy()
            for step in range(self.H):
                if i + step >= n:
                    break
                s_cur = self.base.predict(s_cur, actions[i + step])
            s_true_end = states[min(i + self.H, n - 1)]
            r = (s_true_end - s_cur) / self.delta_std
            residuals.append(r)
            X_list.append(np.concatenate([
                states[i] / self.state_std, [actions[i] / self.action_std]
            ]))

        X = np.array(X_list)
        Y = np.array(residuals)
        _train_residual_net(self._net, X, Y,
                            n_epochs=self._n_epochs, seed=self.seed)

    def predict(self, s, tau):
        s_next_ode = self.base.predict(s, tau)
        x = np.concatenate([s / self.state_std, [tau / self.action_std]]).reshape(1, -1)
        with torch.no_grad():
            r = self._net(torch.FloatTensor(x)).numpy()[0]
        return s_next_ode + r * self.delta_std * self.scale / self.H

    def reset(self):
        pass


class PerStateResidual:
    """Residual only on selected states."""

    def __init__(self, base_model, state_std, action_std, delta_std,
                 active_states=None, residual_scale=0.1, n_epochs=80, hidden=64, seed=42):
        self.base = base_model
        self.state_std = state_std.copy()
        self.action_std = action_std
        self.delta_std = delta_std.copy()
        self.state_std[self.state_std < 1e-10] = 1.0
        self.delta_std[self.delta_std < 1e-10] = 1.0
        self.scale = residual_scale
        self.seed = seed
        # Active states: indices to apply residual
        self.active_states = active_states or list(range(STATE_DIM))
        self._n_active = len(self.active_states)
        self._net = _ResidualNet(STATE_DIM + ACTION_DIM, self._n_active, hidden)
        self._n_epochs = n_epochs

    def train(self, states, actions, deltas):
        residuals = np.empty((len(states), self._n_active))
        for i in range(len(states)):
            s_next_ode = self.base.predict(states[i], actions[i])
            s_next_true = states[i] + deltas[i]
            full_r = (s_next_true - s_next_ode) / self.delta_std
            residuals[i] = full_r[self.active_states]

        X = np.column_stack([states / self.state_std,
                             actions.reshape(-1, 1) / self.action_std])
        _train_residual_net(self._net, X, residuals,
                            n_epochs=self._n_epochs, seed=self.seed)

    def predict(self, s, tau):
        s_next_ode = self.base.predict(s, tau)
        x = np.concatenate([s / self.state_std, [tau / self.action_std]]).reshape(1, -1)
        with torch.no_grad():
            r = self._net(torch.FloatTensor(x)).numpy()[0]
        result = s_next_ode.copy()
        for j, idx in enumerate(self.active_states):
            result[idx] += r[j] * self.delta_std[idx] * self.scale
        return result

    def reset(self):
        pass


class ShortTimeResidual:
    """Residual only for first K steps, then pure NODE."""

    def __init__(self, base_model, state_std, action_std, delta_std,
                 K=10, residual_scale=0.1, n_epochs=80, hidden=64, seed=42):
        self.base = base_model
        self.state_std = state_std.copy()
        self.action_std = action_std
        self.delta_std = delta_std.copy()
        self.state_std[self.state_std < 1e-10] = 1.0
        self.delta_std[self.delta_std < 1e-10] = 1.0
        self.K = K
        self.scale = residual_scale
        self.seed = seed
        self._net = _ResidualNet(STATE_DIM + ACTION_DIM, STATE_DIM, hidden)
        self._n_epochs = n_epochs
        self._step_count = 0

    def train(self, states, actions, deltas):
        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_ode = self.base.predict(states[i], actions[i])
            s_next_true = states[i] + deltas[i]
            residuals[i] = (s_next_true - s_next_ode) / self.delta_std

        X = np.column_stack([states / self.state_std,
                             actions.reshape(-1, 1) / self.action_std])
        _train_residual_net(self._net, X, residuals,
                            n_epochs=self._n_epochs, seed=self.seed)

    def predict(self, s, tau):
        s_next_ode = self.base.predict(s, tau)
        if self._step_count >= self.K:
            self._step_count += 1
            return s_next_ode

        x = np.concatenate([s / self.state_std, [tau / self.action_std]]).reshape(1, -1)
        with torch.no_grad():
            r = self._net(torch.FloatTensor(x)).numpy()[0]
        self._step_count += 1
        return s_next_ode + r * self.delta_std * self.scale

    def reset(self):
        self._step_count = 0


class DecayResidual:
    """Residual with time-domain exponential decay: alpha = alpha_0 * gamma^h."""

    def __init__(self, base_model, state_std, action_std, delta_std,
                 gamma=0.95, residual_scale=0.1, n_epochs=80, hidden=64, seed=42):
        self.base = base_model
        self.state_std = state_std.copy()
        self.action_std = action_std
        self.delta_std = delta_std.copy()
        self.state_std[self.state_std < 1e-10] = 1.0
        self.delta_std[self.delta_std < 1e-10] = 1.0
        self.gamma = gamma
        self.scale = residual_scale
        self.seed = seed
        self._net = _ResidualNet(STATE_DIM + ACTION_DIM, STATE_DIM, hidden)
        self._n_epochs = n_epochs
        self._step_count = 0

    def train(self, states, actions, deltas):
        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_ode = self.base.predict(states[i], actions[i])
            s_next_true = states[i] + deltas[i]
            residuals[i] = (s_next_true - s_next_ode) / self.delta_std

        X = np.column_stack([states / self.state_std,
                             actions.reshape(-1, 1) / self.action_std])
        _train_residual_net(self._net, X, residuals,
                            n_epochs=self._n_epochs, seed=self.seed)

    def predict(self, s, tau):
        s_next_ode = self.base.predict(s, tau)
        alpha = self.scale * (self.gamma ** self._step_count)
        self._step_count += 1

        if alpha < 0.01:
            return s_next_ode

        x = np.concatenate([s / self.state_std, [tau / self.action_std]]).reshape(1, -1)
        with torch.no_grad():
            r = self._net(torch.FloatTensor(x)).numpy()[0]
        return s_next_ode + r * self.delta_std * alpha

    def reset(self):
        self._step_count = 0


class StateGatedResidual:
    """Residual with learned state-dependent gating: alpha = sigma(h(x, u))."""

    def __init__(self, base_model, state_std, action_std, delta_std,
                 residual_scale=0.1, n_epochs=80, hidden=64, seed=42):
        self.base = base_model
        self.state_std = state_std.copy()
        self.action_std = action_std
        self.delta_std = delta_std.copy()
        self.state_std[self.state_std < 1e-10] = 1.0
        self.delta_std[self.delta_std < 1e-10] = 1.0
        self.scale = residual_scale
        self.seed = seed
        self._res_net = _ResidualNet(STATE_DIM + ACTION_DIM, STATE_DIM, hidden)
        self._gate_net = _GatingNet(STATE_DIM + ACTION_DIM, hidden=32)
        self._n_epochs = n_epochs

    def train(self, states, actions, deltas):
        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_ode = self.base.predict(states[i], actions[i])
            s_next_true = states[i] + deltas[i]
            residuals[i] = (s_next_true - s_next_ode) / self.delta_std

        X = np.column_stack([states / self.state_std,
                             actions.reshape(-1, 1) / self.action_std])
        Xt = torch.FloatTensor(X)
        Yt = torch.FloatTensor(residuals)
        ds = torch.utils.data.TensorDataset(Xt, Yt)
        loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

        params = list(self._res_net.parameters()) + list(self._gate_net.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)

        self._res_net.train()
        self._gate_net.train()
        for epoch in range(self._n_epochs):
            for xb, yb in loader:
                r = self._res_net(xb)
                alpha = self._gate_net(xb)
                pred = r * alpha
                loss = nn.functional.mse_loss(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._res_net.eval()
        self._gate_net.eval()

    def predict(self, s, tau):
        s_next_ode = self.base.predict(s, tau)
        x = np.concatenate([s / self.state_std, [tau / self.action_std]]).reshape(1, -1)
        xt = torch.FloatTensor(x)
        with torch.no_grad():
            r = self._res_net(xt).numpy()[0]
            alpha = self._gate_net(xt).numpy()[0, 0]
        return s_next_ode + r * self.delta_std * self.scale * alpha

    def reset(self):
        pass


class UncertaintyGatedResidual:
    """Residual gated by prediction uncertainty: alpha = exp(-beta * U)."""

    def __init__(self, base_model, state_std, action_std, delta_std,
                 beta=1.0, residual_scale=0.1, n_epochs=80, hidden=64, seed=42,
                 ensemble_models=None):
        self.base = base_model
        self.state_std = state_std.copy()
        self.action_std = action_std
        self.delta_std = delta_std.copy()
        self.state_std[self.state_std < 1e-10] = 1.0
        self.delta_std[self.delta_std < 1e-10] = 1.0
        self.beta = beta
        self.scale = residual_scale
        self.seed = seed
        self._net = _ResidualNet(STATE_DIM + ACTION_DIM, STATE_DIM, hidden)
        self._n_epochs = n_epochs
        self._ensemble_models = ensemble_models or []

    def train(self, states, actions, deltas):
        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_ode = self.base.predict(states[i], actions[i])
            s_next_true = states[i] + deltas[i]
            residuals[i] = (s_next_true - s_next_ode) / self.delta_std

        X = np.column_stack([states / self.state_std,
                             actions.reshape(-1, 1) / self.action_std])
        _train_residual_net(self._net, X, residuals,
                            n_epochs=self._n_epochs, seed=self.seed)

    def _compute_uncertainty(self, s, tau):
        """Compute prediction uncertainty from ensemble."""
        if not self._ensemble_models:
            return 0.0
        preds = []
        for m in self._ensemble_models:
            preds.append(m.predict(s, tau))
        preds = np.array(preds)
        return float(np.mean(np.std(preds, axis=0) / self.delta_std))

    def predict(self, s, tau):
        s_next_ode = self.base.predict(s, tau)
        U = self._compute_uncertainty(s, tau)
        alpha = self.scale * np.exp(-self.beta * U)

        if alpha < 0.01:
            return s_next_ode

        x = np.concatenate([s / self.state_std, [tau / self.action_std]]).reshape(1, -1)
        with torch.no_grad():
            r = self._net(torch.FloatTensor(x)).numpy()[0]
        return s_next_ode + r * self.delta_std * alpha

    def reset(self):
        pass


class EnsembleResidual:
    """5-member residual ensemble with mean/median/trimmed aggregation."""

    def __init__(self, base_model, state_std, action_std, delta_std,
                 n_members=5, residual_scale=0.1, n_epochs=80, hidden=64, seed=42,
                 aggregation='mean'):
        self.base = base_model
        self.state_std = state_std.copy()
        self.action_std = action_std
        self.delta_std = delta_std.copy()
        self.state_std[self.state_std < 1e-10] = 1.0
        self.delta_std[self.delta_std < 1e-10] = 1.0
        self.n_members = n_members
        self.scale = residual_scale
        self.seed = seed
        self._nets = []
        self._n_epochs = n_epochs
        self.aggregation = aggregation

    def train(self, states, actions, deltas):
        X = np.column_stack([states / self.state_std,
                             actions.reshape(-1, 1) / self.action_std])

        # Compute residuals
        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_ode = self.base.predict(states[i], actions[i])
            s_next_true = states[i] + deltas[i]
            residuals[i] = (s_next_true - s_next_ode) / self.delta_std

        # Train each member on bootstrap sample
        rng = np.random.RandomState(self.seed)
        self._nets = []
        for m in range(self.n_members):
            n = len(states)
            idx = rng.choice(n, n, replace=True)
            net = _ResidualNet(STATE_DIM + ACTION_DIM, STATE_DIM, hidden)
            _train_residual_net(net, X[idx], residuals[idx],
                                n_epochs=self._n_epochs, seed=self.seed + m)
            self._nets.append(net)

    def predict(self, s, tau):
        s_next_ode = self.base.predict(s, tau)
        x = np.concatenate([s / self.state_std, [tau / self.action_std]]).reshape(1, -1)
        xt = torch.FloatTensor(x)

        preds = []
        with torch.no_grad():
            for net in self._nets:
                preds.append(net(xt).numpy()[0])
        preds = np.array(preds)

        if self.aggregation == 'mean':
            r = np.mean(preds, axis=0)
        elif self.aggregation == 'median':
            r = np.median(preds, axis=0)
        elif self.aggregation == 'trimmed':
            # Trim top and bottom
            sorted_preds = np.sort(preds, axis=0)
            r = np.mean(sorted_preds[1:-1], axis=0) if self.n_members > 2 else np.mean(preds, axis=0)
        else:
            r = np.mean(preds, axis=0)

        return s_next_ode + r * self.delta_std * self.scale

    def predict_with_uncertainty(self, s, tau):
        s_next_ode = self.base.predict(s, tau)
        x = np.concatenate([s / self.state_std, [tau / self.action_std]]).reshape(1, -1)
        xt = torch.FloatTensor(x)

        preds = []
        with torch.no_grad():
            for net in self._nets:
                preds.append(net(xt).numpy()[0])
        preds = np.array(preds)

        mean_r = np.mean(preds, axis=0)
        std_r = np.std(preds, axis=0)
        s_next = s_next_ode + mean_r * self.delta_std * self.scale
        uncertainty = std_r * self.delta_std * self.scale
        return s_next, uncertainty

    def reset(self):
        pass

    def get_member_predictions(self, s, tau):
        """Get individual member predictions for risk-aware planning."""
        x = np.concatenate([s / self.state_std, [tau / self.action_std]]).reshape(1, -1)
        xt = torch.FloatTensor(x)
        s_next_ode = self.base.predict(s, tau)

        preds = []
        with torch.no_grad():
            for net in self._nets:
                r = net(xt).numpy()[0]
                preds.append(s_next_ode + r * self.delta_std * self.scale)
        return np.array(preds)
