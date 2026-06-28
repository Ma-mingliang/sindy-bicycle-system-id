"""Residual Dynamics Ensemble models for 7D - E1, RDE-L, RDE-T, RDE-M."""
import numpy as np
from typing import Optional, Tuple, List
from .models import BaseModel, GPModel
from .config import TrainingConfig, RDEConfig, STATE_DIM

# Lazy torch import
_torch = None

def _get_torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch


class NNEnsemble7D:
    """NN ensemble for 7D residual prediction."""

    def __init__(self, n_models: int = 5, n_epochs: int = 50, lr: float = 1e-3):
        self.n_models = n_models
        self.n_epochs = n_epochs
        self.lr = lr
        self.models = []

    def train(self, train_inputs: np.ndarray, train_residuals: np.ndarray):
        torch = _get_torch()
        import torch.nn as nn

        class ResidualNet7D(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(STATE_DIM + 1, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, STATE_DIM),
                )

            def forward(self, s, a):
                x = torch.cat([s, a], dim=-1)
                return self.net(x)

        self.models = []
        for i in range(self.n_models):
            torch.manual_seed(i * 42)
            np.random.seed(i * 42)
            model = ResidualNet7D()
            train_x = torch.FloatTensor(train_inputs[:, :STATE_DIM])
            train_a = torch.FloatTensor(train_inputs[:, STATE_DIM:])
            train_y = torch.FloatTensor(train_residuals)
            ds = torch.utils.data.TensorDataset(train_x, train_a, train_y)
            loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            crit = torch.nn.MSELoss()
            model.train()
            for epoch in range(self.n_epochs):
                for xb, ab, yb in loader:
                    loss = crit(model(xb, ab), yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            model.eval()
            self.models.append(model)

    def predict_mean(self, s_norm, a_norm):
        torch = _get_torch()
        s_t = torch.FloatTensor(s_norm).unsqueeze(0)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0)
        preds = []
        for model in self.models:
            with torch.no_grad():
                pred = model(s_t, a_t).numpy()[0]
            preds.append(pred)
        return np.mean(preds, axis=0)

    def predict_mean_std(self, s_norm, a_norm):
        torch = _get_torch()
        s_t = torch.FloatTensor(s_norm).unsqueeze(0)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0)
        preds = []
        for model in self.models:
            with torch.no_grad():
                pred = model(s_t, a_t).numpy()[0]
            preds.append(pred)
        preds = np.array(preds)
        return np.mean(preds, axis=0), np.std(preds, axis=0)


class E1OfflineNN(BaseModel):
    """E1: GP + offline NN residual (no DAgger) for 7D."""

    def __init__(self, n_models=5, n_epochs=50, residual_scale=0.3):
        self.n_models = n_models
        self.n_epochs = n_epochs
        self.residual_scale = residual_scale
        self._baseline = None
        self._ensemble = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._baseline = GPModel(max_samples=2000, n_restarts=2)
        self._baseline.train(states, actions, deltas, state_std, action_std, delta_std)

        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_base = self._baseline.predict(states[i], actions[i])
            residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std

        train_inputs = np.column_stack([
            states / state_std,
            actions.reshape(-1, 1) / action_std
        ])
        self._ensemble = NNEnsemble7D(self.n_models, self.n_epochs)
        self._ensemble.train(train_inputs, residuals)

    def predict(self, s, tau):
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        delta_nn = self._ensemble.predict_mean(s_norm, a_norm)
        return s_next_base + delta_nn * self._delta_std * self.residual_scale

    def predict_with_uncertainty(self, s, tau):
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        mean_nn, std_nn = self._ensemble.predict_mean_std(s_norm, a_norm)
        s_next = s_next_base + mean_nn * self._delta_std * self.residual_scale
        uncertainty = std_nn * self._delta_std * self.residual_scale
        return s_next, uncertainty

    def name(self):
        return 'e1_offline_nn'


class RDELocal(BaseModel):
    """RDE-L: Local dynamics residual for 7D."""

    def __init__(self, n_models=5, n_epochs=50, residual_scale=0.3):
        self.n_models = n_models
        self.n_epochs = n_epochs
        self.residual_scale = residual_scale
        self._baseline = None
        self._ensemble = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._baseline = GPModel(max_samples=2000, n_restarts=2)
        self._baseline.train(states, actions, deltas, state_std, action_std, delta_std)

        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_base = self._baseline.predict(states[i], actions[i])
            residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std

        train_inputs = np.column_stack([
            states / state_std,
            actions.reshape(-1, 1) / action_std
        ])
        self._ensemble = NNEnsemble7D(self.n_models, self.n_epochs)
        self._ensemble.train(train_inputs, residuals)

    def predict(self, s, tau):
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        delta_nn = self._ensemble.predict_mean(s_norm, a_norm)
        return s_next_base + delta_nn * self._delta_std * self.residual_scale

    def predict_with_uncertainty(self, s, tau):
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        mean_nn, std_nn = self._ensemble.predict_mean_std(s_norm, a_norm)
        s_next = s_next_base + mean_nn * self._delta_std * self.residual_scale
        uncertainty = std_nn * self._delta_std * self.residual_scale
        return s_next, uncertainty

    def name(self):
        return 'rde_local'


class RDETrajectory(BaseModel):
    """RDE-T: Trajectory sync residual for 7D."""

    def __init__(self, n_models=5, n_epochs=50, residual_scale=0.3,
                 dagger_rounds=1, n_segments=3, segment_length=500):
        self.n_models = n_models
        self.n_epochs = n_epochs
        self.residual_scale = residual_scale
        self.dagger_rounds = dagger_rounds
        self.n_segments = n_segments
        self.segment_length = segment_length
        self._baseline = None
        self._ensemble = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def _collect_trajectory_data(self, data_segments, dynamics_fn=None):
        """Collect training data from trajectory sync."""
        all_inputs = []
        all_residuals = []
        for seg in data_segments:
            states_seg = seg['states']
            actions_seg = seg['actions']
            s_model = states_seg[0].copy()

            for step in range(min(self.segment_length, len(actions_seg))):
                tau = actions_seg[step]
                s_real_next = states_seg[step + 1]

                s_model_norm = s_model / self._state_std
                a_norm = tau / self._action_std

                gp_next = self._baseline.predict(s_model, tau)
                gp_delta = gp_next - s_model
                real_delta = s_real_next - states_seg[step]

                label = (real_delta - gp_delta) / self._delta_std

                all_inputs.append(np.concatenate([s_model_norm, [a_norm]]))
                all_residuals.append(label)

                if self._ensemble and self._ensemble.models:
                    delta_nn = self._ensemble.predict_mean(s_model_norm, a_norm)
                    s_model = gp_next + delta_nn * self._delta_std * self.residual_scale
                else:
                    s_model = gp_next.copy()

        return np.array(all_inputs), np.array(all_residuals)

    def train(self, states, actions, deltas, state_std, action_std, delta_std,
              data_segments=None, dynamics=None, make_tau_func=None):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        self._baseline = GPModel(max_samples=2000, n_restarts=2)
        self._baseline.train(states, actions, deltas, state_std, action_std, delta_std)

        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_base = self._baseline.predict(states[i], actions[i])
            residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std

        train_inputs = np.column_stack([
            states / state_std,
            actions.reshape(-1, 1) / action_std
        ])
        self._ensemble = NNEnsemble7D(self.n_models, self.n_epochs)
        self._ensemble.train(train_inputs, residuals)

        for rd in range(self.dagger_rounds):
            if data_segments:
                traj_inputs, traj_residuals = self._collect_trajectory_data(data_segments)
                if len(traj_inputs) > 0:
                    self._ensemble.train(traj_inputs, traj_residuals)

    def predict(self, s, tau):
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        delta_nn = self._ensemble.predict_mean(s_norm, a_norm)
        return s_next_base + delta_nn * self._delta_std * self.residual_scale

    def predict_with_uncertainty(self, s, tau):
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        mean_nn, std_nn = self._ensemble.predict_mean_std(s_norm, a_norm)
        s_next = s_next_base + mean_nn * self._delta_std * self.residual_scale
        uncertainty = std_nn * self._delta_std * self.residual_scale
        return s_next, uncertainty

    def name(self):
        return 'rde_trajectory'


class RDEHybrid(BaseModel):
    """RDE-M: Hybrid DAgger for 7D."""

    def __init__(self, n_models=5, n_epochs=50, residual_scale=0.3,
                 dagger_rounds=2, n_segments=3, segment_length=500):
        self.n_models = n_models
        self.n_epochs = n_epochs
        self.residual_scale = residual_scale
        self.dagger_rounds = dagger_rounds
        self.n_segments = n_segments
        self.segment_length = segment_length
        self._baseline = None
        self._ensemble = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def _collect_dagger_data(self, data_segments):
        """Collect DAgger data with local residual labels at model state."""
        all_inputs = []
        all_residuals = []
        for seg in data_segments:
            states_seg = seg['states']
            actions_seg = seg['actions']
            s_model = states_seg[0].copy()

            for step in range(min(self.segment_length, len(actions_seg))):
                tau = actions_seg[step]
                s_model_norm = s_model / self._state_std
                a_norm = tau / self._action_std

                gp_next = self._baseline.predict(s_model, tau)
                base_delta = gp_next - s_model

                # Local residual: use next real state as proxy for F(s_model, u)
                # In real data setting, we approximate with the observed transition
                real_next = states_seg[step + 1]
                real_delta = real_next - states_seg[step]
                new_residual = (real_delta - base_delta) / self._delta_std

                all_inputs.append(np.concatenate([s_model_norm, [a_norm]]))
                all_residuals.append(new_residual)

                if self._ensemble and self._ensemble.models:
                    delta_nn = self._ensemble.predict_mean(s_model_norm, a_norm)
                    s_model = gp_next + delta_nn * self._delta_std * self.residual_scale
                else:
                    s_model = gp_next.copy()

        return np.array(all_inputs), np.array(all_residuals)

    def train(self, states, actions, deltas, state_std, action_std, delta_std,
              data_segments=None, dynamics=None, make_tau_func=None):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        self._baseline = GPModel(max_samples=2000, n_restarts=2)
        self._baseline.train(states, actions, deltas, state_std, action_std, delta_std)

        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_base = self._baseline.predict(states[i], actions[i])
            residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std

        train_inputs = np.column_stack([
            states / state_std,
            actions.reshape(-1, 1) / action_std
        ])
        self._ensemble = NNEnsemble7D(self.n_models, self.n_epochs)
        self._ensemble.train(train_inputs, residuals)

        for rd in range(self.dagger_rounds):
            if data_segments:
                dagger_inputs, dagger_residuals = self._collect_dagger_data(data_segments)
                if len(dagger_inputs) > 0:
                    self._ensemble.train(dagger_inputs, dagger_residuals)

    def predict(self, s, tau):
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        delta_nn = self._ensemble.predict_mean(s_norm, a_norm)
        return s_next_base + delta_nn * self._delta_std * self.residual_scale

    def predict_with_uncertainty(self, s, tau):
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        mean_nn, std_nn = self._ensemble.predict_mean_std(s_norm, a_norm)
        s_next = s_next_base + mean_nn * self._delta_std * self.residual_scale
        uncertainty = std_nn * self._delta_std * self.residual_scale
        return s_next, uncertainty

    def name(self):
        return 'rde_hybrid'
