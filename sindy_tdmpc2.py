"""
SINDy-enhanced TD-MPC2 agent for bicycle balancing.

Hybrid architecture:
  World Model (latent): encoder, dynamics, reward — with SINDy consistency
  Q-networks (obs-space): (obs, a) → Q — directly on raw observations
  Policy (obs-space): obs → a — directly on raw observations
  Planning: MPPI using world model's reward prediction

The Q-network and policy operate on raw observations (like HRRL's TD3),
while the world model provides SINDy-regularized dynamics for planning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def _build_theta_torch(state_norm, action_norm):
    """Build polynomial library row for SINDy prediction (PyTorch)."""
    x = torch.cat([state_norm, action_norm.unsqueeze(-1) if action_norm.dim() == 0 else action_norm], dim=-1)
    n = x.shape[-1]
    terms = [torch.ones(*x.shape[:-1], 1, device=x.device)]
    terms.append(x)
    for i in range(n):
        for j in range(i, n):
            terms.append((x[..., i] * x[..., j]).unsqueeze(-1))
    return torch.cat(terms, dim=-1)


class WorldModel(nn.Module):
    """Latent world model with SINDy prior (for planning only)."""

    def __init__(self, obs_dim, action_dim, latent_dim, mlp_dim, sindy_Xi, action_scale):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: obs → latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, latent_dim),
        )

        # Dynamics: (z, a) → delta_z
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, latent_dim),
        )

        # Reward: (z, a) → scalar (for MPPI planning)
        self.reward = nn.Sequential(
            nn.Linear(latent_dim + action_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim), nn.SiLU(),
            nn.Linear(mlp_dim, 1),
        )

        # SINDy prior (fixed, not trained)
        self.register_buffer('sindy_Xi', torch.tensor(sindy_Xi, dtype=torch.float32))
        self.register_buffer('action_scale', torch.tensor(action_scale, dtype=torch.float32))

    def encode(self, obs):
        return self.encoder(obs)

    def next_z(self, z, a):
        return z + self.dynamics(torch.cat([z, a], dim=-1))

    def predict_reward(self, z, a):
        return self.reward(torch.cat([z, a], dim=-1))

    def sindy_predict_delta(self, obs_norm, action_norm):
        """SINDy forward prediction in normalized space."""
        a_norm = action_norm / self.action_scale if self.action_scale > 0 else action_norm
        theta_row = _build_theta_torch(obs_norm, a_norm)
        return theta_row @ self.sindy_Xi


class QNetwork(nn.Module):
    """Q-network operating directly on observations (like HRRL's TD3)."""

    def __init__(self, obs_dim, action_dim, mlp_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, mlp_dim), nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim), nn.ReLU(),
            nn.Linear(mlp_dim, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


class Policy(nn.Module):
    """Policy operating directly on observations (like HRRL's TD3)."""

    def __init__(self, obs_dim, action_dim, mlp_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, mlp_dim), nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim), nn.ReLU(),
            nn.Linear(mlp_dim, 2 * action_dim),
        )

    def forward(self, obs):
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -2, 2)
        return mean, log_std


class SINDyTDMPC2:
    """SINDy-enhanced TD-MPC2 agent with obs-space Q-network and policy."""

    def __init__(self, cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        latent_dim = cfg.get('latent_dim', 64)
        mlp_dim = cfg.get('mlp_dim', 256)
        lr = cfg.get('lr', 3e-4)
        self.horizon = cfg.get('horizon', 3)
        self.num_samples = cfg.get('num_samples', 64)
        self.num_elites = cfg.get('num_elites', 8)
        self.num_pi_trajs = cfg.get('num_pi_trajs', 8)
        self.iterations = cfg.get('iterations', 6)
        self.temperature = cfg.get('temperature', 0.5)
        self.max_std = cfg.get('max_std', 0.5)
        self.min_std = cfg.get('min_std', 0.05)
        self.sindy_coef = cfg.get('sindy_coef', 5.0)
        self.tau = cfg.get('tau', 0.005)
        self.gamma = cfg.get('gamma', 0.99)
        self.grad_clip = cfg.get('grad_clip', 10.0)
        self.action_dim = action_dim

        # World model (latent space, for planning + SINDy regularization)
        self.world_model = WorldModel(obs_dim, action_dim, latent_dim, mlp_dim, sindy_Xi, action_scale).to(self.device)
        self.wm_optim = torch.optim.Adam(self.world_model.parameters(), lr=lr)

        # Q-networks (obs space, like HRRL's TD3)
        self.Q1 = QNetwork(obs_dim, action_dim, mlp_dim).to(self.device)
        self.Q2 = QNetwork(obs_dim, action_dim, mlp_dim).to(self.device)
        self.target_Q1 = deepcopy(self.Q1)
        self.target_Q2 = deepcopy(self.Q2)
        for p in self.target_Q1.parameters():
            p.requires_grad = False
        for p in self.target_Q2.parameters():
            p.requires_grad = False
        self.q_optim = torch.optim.Adam(
            list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=lr
        )

        # Policy (obs space, like HRRL's TD3)
        self.pi = Policy(obs_dim, action_dim, mlp_dim).to(self.device)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=lr)

        self._prev_mean = torch.zeros(self.horizon, action_dim, device=self.device)

    @torch.no_grad()
    def act_policy(self, obs, eval_mode=False):
        """Policy action in obs space (for training and fast inference)."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, log_std = self.pi(obs_t)
        if eval_mode:
            action = mean
        else:
            eps = torch.randn_like(mean)
            action = mean + eps * log_std.exp()
        return action.clamp(-1, 1).cpu().numpy().flatten()

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False):
        """MPPI planning using world model's reward prediction."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        z = self.world_model.encode(obs_t)

        if t0:
            self._prev_mean.zero_()

        z = z.repeat(self.num_samples, 1)
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.full((self.horizon, self.action_dim), self.max_std, device=self.device)
        mean[:-1] = self._prev_mean[1:]

        actions = torch.empty(self.horizon, self.num_samples, self.action_dim, device=self.device)

        for _iter in range(self.iterations):
            r = torch.randn(self.horizon, self.num_samples - self.num_pi_trajs, self.action_dim, device=self.device)
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)

            # Policy-guided samples (policy operates on obs, not latent)
            _z_pi = z[:self.num_pi_trajs].clone()
            for t in range(self.horizon):
                m, ls = self.pi(obs_t.repeat(self.num_pi_trajs, 1))
                a_pi = m + ls.exp() * torch.randn_like(m)
                a_pi = a_pi.clamp(-1, 1)
                actions[t, :self.num_pi_trajs] = a_pi
                if t < self.horizon - 1:
                    _z_pi = self.world_model.next_z(_z_pi, a_pi)

            actions[:, self.num_pi_trajs:] = actions_sample

            # Evaluate trajectories using world model reward
            G = torch.zeros(self.num_samples, self.action_dim, device=self.device)
            _z = z.clone()
            discount = 1.0
            for t in range(self.horizon):
                r_pred = self.world_model.predict_reward(_z, actions[t])
                _z = self.world_model.next_z(_z, actions[t])
                G = G + discount * r_pred
                discount *= self.gamma

            # Terminal value from obs-space Q-network
            a_term_mean, _ = self.pi(obs_t)
            q1_term, q2_term = self.Q1(obs_t, a_term_mean), self.Q2(obs_t, a_term_mean)
            G = G + discount * torch.min(q1_term, q2_term).expand_as(G)

            # Elite selection
            elite_idx = torch.topk(G.squeeze(-1), self.num_elites, dim=0).indices
            elite_actions = actions[:, elite_idx]
            elite_value = G[elite_idx]

            # Update mean/std
            max_val = elite_value.max(0).values
            score = torch.exp(self.temperature * (elite_value - max_val))
            score = score / (score.sum(0) + 1e-9)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(self.min_std, self.max_std)

        self._prev_mean.copy_(mean)
        a = mean[0] + std[0] * torch.randn(self.action_dim, device=self.device)
        if eval_mode:
            a = mean[0]
        return a.clamp(-1, 1).cpu().numpy()

    def update(self, batch, step=0):
        """Train all components."""
        obs, action, reward, next_obs, terminated = [b.to(self.device) for b in batch]

        # === 1. World model update (SINDy-regularized) ===
        z = self.world_model.encode(obs)
        with torch.no_grad():
            next_z = self.world_model.encode(next_obs)

        z_pred = self.world_model.next_z(z.detach(), action)
        consistency_loss = F.mse_loss(z_pred, next_z)

        r_pred = self.world_model.predict_reward(z.detach(), action)
        reward_loss = F.mse_loss(r_pred, reward)

        # SINDy prior
        if self.sindy_coef > 0:
            with torch.no_grad():
                sindy_delta = self.world_model.sindy_predict_delta(obs, action)
            sindy_delta_norm = sindy_delta.norm(dim=-1, keepdim=True)
            learned_delta_norm = (z_pred - z.detach()).norm(dim=-1, keepdim=True)
            sindy_loss = F.mse_loss(learned_delta_norm, sindy_delta_norm.detach() * 0.1)
        else:
            sindy_loss = torch.tensor(0.0, device=obs.device)

        wm_loss = consistency_loss + reward_loss + self.sindy_coef * sindy_loss
        self.wm_optim.zero_grad()
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.grad_clip)
        self.wm_optim.step()

        # === 2. Q-network update (TD3-style, obs space) ===
        with torch.no_grad():
            # Target policy smoothing
            a_next_mean, a_next_logstd = self.pi(next_obs)
            a_next = a_next_mean + torch.clamp(
                torch.randn_like(a_next_mean) * 0.2, -0.5, 0.5
            ).clamp(-1, 1)
            q1_tgt, q2_tgt = self.target_Q1(next_obs, a_next), self.target_Q2(next_obs, a_next)
            q1_tgt = q1_tgt.clamp(-50, 50)
            q2_tgt = q2_tgt.clamp(-50, 50)
            td_target = reward + self.gamma * (1 - terminated) * torch.min(q1_tgt, q2_tgt)
            td_target = torch.clamp(td_target, -50, 50)

        q1 = self.Q1(obs, action).clamp(-50, 50)
        q2 = self.Q2(obs, action).clamp(-50, 50)
        q_loss = F.smooth_l1_loss(q1, td_target) + F.smooth_l1_loss(q2, td_target)

        self.q_optim.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.Q1.parameters()) + list(self.Q2.parameters()), self.grad_clip
        )
        self.q_optim.step()

        # === 3. Policy update (TD3 delayed, obs space) ===
        pi_loss_val = 0.0
        if step % 2 == 0:
            a_pi_mean, _ = self.pi(obs)
            a_pi_mean = a_pi_mean.clamp(-1, 1)  # Clamp before Q evaluation
            q1_pi, q2_pi = self.Q1(obs, a_pi_mean).clamp(-50, 50), self.Q2(obs, a_pi_mean).clamp(-50, 50)
            pi_loss = -torch.min(q1_pi, q2_pi).mean()
            self.pi_optim.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pi.parameters(), self.grad_clip)
            self.pi_optim.step()
            pi_loss_val = pi_loss.item()

        # Update target Q
        for p, tp in zip(self.Q1.parameters(), self.target_Q1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.Q2.parameters(), self.target_Q2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {
            'total_loss': (wm_loss + q_loss).item(),
            'consistency_loss': consistency_loss.item(),
            'reward_loss': reward_loss.item(),
            'q_loss': q_loss.item(),
            'sindy_loss': sindy_loss.item(),
            'pi_loss': pi_loss_val,
        }

    def save(self, path):
        torch.save({
            'world_model': self.world_model.state_dict(),
            'Q1': self.Q1.state_dict(),
            'Q2': self.Q2.state_dict(),
            'pi': self.pi.state_dict(),
        }, path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.world_model.load_state_dict(state_dict['world_model'])
        self.Q1.load_state_dict(state_dict['Q1'])
        self.Q2.load_state_dict(state_dict['Q2'])
        self.pi.load_state_dict(state_dict['pi'])
