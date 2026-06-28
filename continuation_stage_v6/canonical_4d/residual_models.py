"""Residual Dynamics Ensemble models - E1, RDE-L, RDE-T, RDE-M.

CRITICAL: Each model generates its own training labels independently.
RDE-L uses oracle simulation states (local residual).
RDE-T uses trajectory sync (real states vs model rollout).
RDE-M is the hybrid (historical DAgger).
"""
import numpy as np
import math
from typing import Optional, Tuple, List
from .models import BaseModel, GPModel
from .config import TrainingConfig, RDEConfig

# Lazy torch import: only load when NNEnsemble is actually used
_torch = None

def _get_torch():
    global _torch
    if _torch is None:
        import torch as _t
        _torch = _t
    return _torch


def _get_nn():
    import methods_nn
    return methods_nn


class NNEnsemble:
    """Small NN ensemble for residual prediction."""

    def __init__(self, n_models: int = 5, n_epochs: int = 50, lr: float = 1e-3):
        self.n_models = n_models
        self.n_epochs = n_epochs
        self.lr = lr
        self.models = []

    def train(self, train_inputs: np.ndarray, train_residuals: np.ndarray):
        torch = _get_torch()
        nn_mod = _get_nn()
        ResidualNet = nn_mod.ResidualNet
        DEVICE = nn_mod.DEVICE
        self.models = []
        for i in range(self.n_models):
            torch.manual_seed(i * 42)
            np.random.seed(i * 42)
            model = ResidualNet().to(DEVICE)
            train_x = torch.FloatTensor(train_inputs).to(DEVICE)
            train_y = torch.FloatTensor(train_residuals).to(DEVICE)
            ds = torch.utils.data.TensorDataset(train_x, train_y)
            loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            crit = torch.nn.MSELoss()
            model.train()
            for epoch in range(self.n_epochs):
                for xb, yb in loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    loss = crit(model(xb), yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            model.eval()
            self.models.append(model)

    def predict_mean(self, s_norm, a_norm):
        torch = _get_torch()
        nn_mod = _get_nn()
        DEVICE = nn_mod.DEVICE
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
        preds = []
        for model in self.models:
            with torch.no_grad():
                pred = model(s_t, a_t).cpu().numpy()[0]
            preds.append(pred)
        return np.mean(preds, axis=0)

    def predict_mean_std(self, s_norm, a_norm):
        torch = _get_torch()
        nn_mod = _get_nn()
        DEVICE = nn_mod.DEVICE
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
        preds = []
        for model in self.models:
            with torch.no_grad():
                pred = model(s_t, a_t).cpu().numpy()[0]
            preds.append(pred)
        preds = np.array(preds)
        return np.mean(preds, axis=0), np.std(preds, axis=0)


class E1OfflineNN(BaseModel):
    """E1: GP + offline NN residual (no DAgger)."""

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

        train_inputs = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
        self._ensemble = NNEnsemble(self.n_models, self.n_epochs)
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
    """RDE-L: Local dynamics residual.

    Label: r^L = F(oracle_state, u) - GP(oracle_state, u)
    where oracle_state is the real simulated state at query time.
    Uses oracle (real dynamics) to get the state for residual computation.
    """

    def __init__(self, n_models=5, n_epochs=50, residual_scale=0.3):
        self.n_models = n_models
        self.n_epochs = n_epochs
        self.residual_scale = residual_scale
        self._baseline = None
        self._ensemble = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None
        self._dynamics = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._baseline = GPModel(max_samples=2000, n_restarts=2)
        self._baseline.train(states, actions, deltas, state_std, action_std, delta_std)

        # RDE-L label: residual at the REAL state
        # r^L_i = F(s_i, u_i) - GP(s_i, u_i)
        # Here s_i are the real states from training data, so this is:
        # r^L_i = delta_real_i - (GP(s_i, u_i) - s_i)
        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_base = self._baseline.predict(states[i], actions[i])
            residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std

        train_inputs = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
        self._ensemble = NNEnsemble(self.n_models, self.n_epochs)
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
    """RDE-T: Trajectory sync residual.

    CRITICAL DISTINCTION from RDE-L:
    RDE-T trains on residuals collected during model rollout (DAgger style),
    but the LABEL is computed by comparing real trajectory vs model trajectory
    at each step, NOT just local residual at the training data state.

    Label: r^T_k = (s_real[k+1] - s_real[k]) - (GP(s_model[k], u_k) - s_model[k])
    This captures the cumulative effect of model state drift.
    """

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
        self._dynamics = None

    def _collect_trajectory_data(self, dynamics, make_tau_func):
        """Collect training data from trajectory sync.

        For each step k:
        - s_real[k] = real state
        - s_model[k] = model predicted state (from previous step)
        - u_k = action from real controller at s_real[k]
        - GP_pred = GP(s_model[k], u_k) - s_model[k]  (what GP predicts from model state)
        - real_delta = s_real[k+1] - s_real[k]  (what actually happened)
        - label = (real_delta - GP_pred) / delta_std
        """
        all_inputs = []
        all_residuals = []
        for seg in range(self.n_segments):
            tau_func = make_tau_func(100 + seg, self.segment_length)
            phi0 = np.random.uniform(-0.25, 0.25)
            s0 = np.array([phi0, 0.0, 0.0, 0.0])
            s_real = s0.copy()
            s_model = s0.copy()

            for step in range(self.segment_length):
                tau = tau_func(step, s_real)
                s_real_next = dynamics.step(s_real, tau)

                if dynamics.is_diverged(s_real_next):
                    break

                # Input: model state (which may have drifted)
                s_model_norm = s_model / self._state_std
                a_norm = tau / self._action_std

                # GP prediction from model state
                gp_next = self._baseline.predict(s_model, tau)
                gp_delta = gp_next - s_model

                # Real delta
                real_delta = s_real_next - s_real

                # Trajectory sync label
                label = (real_delta - gp_delta) / self._delta_std

                all_inputs.append(np.concatenate([s_model_norm, [a_norm]]))
                all_residuals.append(label)

                # Update model state with GP+NN
                delta_nn = self._ensemble.predict_mean(s_model_norm, a_norm) if self._ensemble.models else np.zeros(4)
                s_model = gp_next + delta_nn * self._delta_std * self.residual_scale
                s_real = s_real_next

        return np.array(all_inputs), np.array(all_residuals)

    def train(self, states, actions, deltas, state_std, action_std, delta_std,
              dynamics=None, make_tau_func=None):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._dynamics = dynamics

        # Round 0: offline label (same as E1 for initialization)
        self._baseline = GPModel(max_samples=2000, n_restarts=2)
        self._baseline.train(states, actions, deltas, state_std, action_std, delta_std)

        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_base = self._baseline.predict(states[i], actions[i])
            residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std

        train_inputs = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
        self._ensemble = NNEnsemble(self.n_models, self.n_epochs)
        self._ensemble.train(train_inputs, residuals)

        # DAgger rounds with trajectory sync labels
        for rd in range(self.dagger_rounds):
            traj_inputs, traj_residuals = self._collect_trajectory_data(dynamics, make_tau_func)
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
    """RDE-M: Hybrid (historical DAgger method, for comparison only).

    Round 0: offline local residual label.
    Round 1+: DAgger with model rollout, but label is LOCAL residual
    at model state (not trajectory sync).
    This is the ORIGINAL V4/V5 implementation.
    """

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

    def _collect_dagger_data(self, dynamics, make_tau_func):
        """Collect DAgger data with LOCAL residual labels (not trajectory sync).

        KEY DISTINCTION from RDE-T:
        RDE-T label = (real_delta - GP(s_model, u)) where real_delta = s_real_next - s_real
        RDE-M label = (true_delta_from_model - GP(s_model, u)) where true_delta_from_model = dynamics.step(s_model, u) - s_model

        RDE-M approximates F(s_model, u) by running the true dynamics simulator FROM the model state.
        This gives the LOCAL residual at the model state, not the trajectory comparison.
        """
        all_inputs = []
        all_residuals = []
        for seg in range(self.n_segments):
            tau_func = make_tau_func(100 + seg, self.segment_length)
            phi0 = np.random.uniform(-0.25, 0.25)
            s0 = np.array([phi0, 0.0, 0.0, 0.0])
            s_model = s0.copy()
            s_real = s0.copy()

            for step in range(self.segment_length):
                tau = tau_func(step, s_real)
                s_real_next = dynamics.step(s_real, tau)

                if dynamics.is_diverged(s_real_next):
                    break

                s_model_norm = s_model / self._state_std
                a_norm = tau / self._action_std

                # Local residual label at MODEL state
                gp_next = self._baseline.predict(s_model, tau)
                base_delta = gp_next - s_model

                # KEY FIX: Use true dynamics FROM model state (not real trajectory delta)
                # This approximates F(s_model, u) by stepping the simulator from s_model
                true_next_from_model = dynamics.step(s_model, tau)
                true_delta_from_model = true_next_from_model - s_model

                # Label: local residual at model state
                new_residual = (true_delta_from_model - base_delta) / self._delta_std

                all_inputs.append(np.concatenate([s_model_norm, [a_norm]]))
                all_residuals.append(new_residual)

                delta_nn = self._ensemble.predict_mean(s_model_norm, a_norm) if self._ensemble.models else np.zeros(4)
                s_model = gp_next + delta_nn * self._delta_std * self.residual_scale
                s_real = s_real_next

        return np.array(all_inputs), np.array(all_residuals)

    def train(self, states, actions, deltas, state_std, action_std, delta_std,
              dynamics=None, make_tau_func=None):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        self._baseline = GPModel(max_samples=2000, n_restarts=2)
        self._baseline.train(states, actions, deltas, state_std, action_std, delta_std)

        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_base = self._baseline.predict(states[i], actions[i])
            residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std

        train_inputs = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
        self._ensemble = NNEnsemble(self.n_models, self.n_epochs)
        self._ensemble.train(train_inputs, residuals)

        for rd in range(self.dagger_rounds):
            dagger_inputs, dagger_residuals = self._collect_dagger_data(dynamics, make_tau_func)
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
