"""MBPO-SAC v4: Suboptimal controller + large residual.

Key insight: LQR is near-optimal for linear Meijaard model.
RL residual can only improve a SUBOPTIMAL controller.

Approach:
1. Use weakened LQR (Q = diag(100, 10, 1, 0.1)) as suboptimal base
2. Large ALPHA=0.5 for big residual range
3. Process noise + actuator delay
4. 1200 episodes training
"""

import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random
from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix
from scipy.linalg import solve_continuous_are


# ============================================================
# Constants
# ============================================================

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6
ALPHA = 0.5  # Large residual range


# ============================================================
# SAC Networks
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity=300000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean


class SACAgent:
    def __init__(self, state_dim, action_dim, device,
                 lr=3e-4, gamma=0.99, tau=0.005, auto_alpha=True):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.critic1 = QNetwork(state_dim, action_dim).to(device)
        self.critic2 = QNetwork(state_dim, action_dim).to(device)
        self.critic1_target = QNetwork(state_dim, action_dim).to(device)
        self.critic2_target = QNetwork(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.policy = GaussianPolicy(state_dim, action_dim).to(device)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        if auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.target_entropy = -action_dim
        self.auto_alpha = auto_alpha

    @property
    def alpha_val(self):
        if self.auto_alpha:
            return min(self.log_alpha.exp().item(), 1.0)
        return 0.2

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            _, _, action = self.policy.sample(state_t)
        else:
            action, _, _ = self.policy.sample(state_t)
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:
            return {}
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha_val * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        new_actions, log_probs, _ = self.policy.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha_val * log_probs - q_new).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.log_alpha.data.clamp_(-5, 0)
        for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        return {'critic1_loss': critic1_loss.item(), 'alpha': self.alpha_val}


class SINDyWorldModel:
    def __init__(self, A_d, B_d, K_lqr):
        self.A_d = A_d
        self.B_d = B_d
        self.K_lqr = K_lqr

    def predict_next(self, theta, delta, theta_dot, delta_dot, action, target_theta):
        epsilon = action * ALPHA
        x_lqr = np.array([theta - target_theta, theta_dot, delta, delta_dot])
        u_lqr = float(-self.K_lqr @ x_lqr)
        tau = u_lqr + epsilon
        s = np.array([theta, delta, theta_dot, delta_dot])
        s_next = self.A_d @ s + self.B_d.flatten() * tau
        return s_next[0], s_next[1], s_next[2], s_next[3]


def compute_sigma(ep):
    return min(0.3, (1.08 ** ep) * 0.01)


def compute_reward(prev_dis, new_dis, w0, terminated):
    if terminated:
        return -1.0 - 0.01 * abs(w0)
    elif abs(new_dis) < 0.002:
        return 0.1 - abs(w0)
    else:
        return abs(prev_dis) - abs(new_dis)


def normalize_state(dis_angle, theta, theta_dot, v):
    return np.array([
        np.clip(dis_angle, -1.57, 1.57) / 1.57,
        np.clip(theta, -1.57, 1.57) / 1.57,
        np.clip(theta_dot, -10., 10.) / 10.,
        np.clip(v, -5., 5.) / 5.,
    ], dtype=np.float32)


class SuboptimalEnv:
    """Environment with SUBOPTIMAL LQR + process noise.

    Uses weakened Q matrix so LQR is not optimal.
    RL can learn to compensate for the suboptimality.
    """

    def __init__(self, K_sub, v0=3.5, dt=1/30, max_steps=1000):
        self.K_sub = K_sub
        self.v0 = v0
        self.dt = dt
        self.max_steps = max_steps
        self.g = 9.81
        p = {
            'IBxx': 11.3557360401, 'IBxz': -1.96756380745,
            'IByy': 12.2177848012, 'IBzz': 3.12354397008,
            'IFxx': 0.0904106601579, 'IFyy': 0.149389340425,
            'IHxx': 0.253379594731, 'IHxz': -0.0720452391817,
            'IHyy': 0.246138810935, 'IHzz': 0.0955770796289,
            'IRxx': 0.0883819364527, 'IRyy': 0.152467620286,
            'c': 0.0685808540382, 'g': 9.81, 'lam': 0.399680398707,
            'mB': 81.86, 'mF': 2.02, 'mH': 3.22, 'mR': 3.11,
            'rF': 0.34352982332, 'rR': 0.340958858855, 'w': 1.121,
            'xB': 0.289099434117, 'xH': 0.866949640247,
            'zB': -1.04029228321, 'zH': -0.748236400835,
        }
        self.M, self.C1, self.K0, self.K2 = benchmark_par_to_canonical(p)
        self.sub_steps = 5
        self.base_noise = 0.001
        self.prev_action = 0.0

    def reset(self, episode_count, seed=None):
        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.theta = rng.uniform(-0.05, 0.05)
        self.theta_dot = 0.0
        self.delta = 0.0
        self.delta_dot = 0.0
        self.step_count = 0
        self.episode_count = episode_count
        self.sigma = compute_sigma(episode_count)
        self.target_theta = np.clip(rng.normal(0, self.sigma), -math.pi / 12, math.pi / 12)
        self.rng = rng
        self.prev_action = 0.0
        return self._get_obs()

    def _get_obs(self):
        dis_angle = self.target_theta - self.theta
        return normalize_state(dis_angle, self.theta, self.theta_dot, self.v0)

    def step(self, action):
        action = float(np.clip(action[0], -1.0, 1.0))
        self.step_count += 1
        if self.step_count <= 100:
            self.target_theta = 0.0
        elif self.step_count % 100 == 1:
            self.target_theta = np.clip(self.rng.normal(0, self.sigma),
                                        -math.pi / 12, math.pi / 12)
        # Actuator delay
        delayed_action = 0.7 * action + 0.3 * self.prev_action
        self.prev_action = action

        # Suboptimal LQR + RL residual
        epsilon = delayed_action * ALPHA
        x_lqr = np.array([self.theta - self.target_theta, self.theta_dot,
                          self.delta, self.delta_dot])
        u_sub = float(-self.K_sub @ x_lqr)
        tau = u_sub + epsilon

        prev_dis = self.target_theta - self.theta
        dt_sub = self.dt / self.sub_steps
        A_m, B_m = ab_matrix(self.M, self.C1, self.K0, self.K2,
                             max(self.v0, 0.5), self.g)
        B_s = B_m[:, 1:2]
        noise_scale = self.base_noise * (1 + self.sigma * 10)
        for _ in range(self.sub_steps):
            x = np.array([[self.theta], [self.delta],
                          [self.theta_dot], [self.delta_dot]])
            noise = self.rng.normal(0, noise_scale, size=(4, 1))
            xd = A_m @ x + B_s * tau + noise
            self.theta += dt_sub * xd[0, 0]
            self.delta += dt_sub * xd[1, 0]
            self.theta_dot += dt_sub * xd[2, 0]
            self.delta_dot += dt_sub * xd[3, 0]
        new_dis = self.target_theta - self.theta
        terminated = abs(self.theta) > math.pi / 3
        reward = compute_reward(prev_dis, new_dis, self.theta_dot, terminated)
        done = terminated or (self.step_count >= self.max_steps)
        return self._get_obs(), reward, done, {'tracking_err': abs(new_dis)}


def sindy_virtual_rollout(model, buffer, agent, n_rollouts=10, horizon=8):
    data = []
    for _ in range(n_rollouts):
        if len(buffer) < 32:
            break
        states, _, _, _, _ = buffer.sample(1)
        state = states[0]
        dis = state[0] * 1.57
        theta = state[1] * 1.57
        w = state[2] * 10.0
        v = state[3] * 5.0
        target = dis + theta
        for h in range(horizon):
            action = agent.select_action(state)
            tn, dn, wn, vn = model.predict_next(theta, 0.0, w, v, action[0], target)
            new_dis = target - tn
            term = abs(tn) > math.pi / 3
            r = compute_reward(dis, new_dis, wn, term)
            ns = normalize_state(new_dis, tn, wn, vn)
            data.append((state, action, r, ns, float(term)))
            state = ns
            theta = tn
            w = wn
            dis = new_dis
            if term:
                break
    return data


def train():
    print("=" * 70)
    print("MBPO-SAC v4: Suboptimal controller + large residual")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"ALPHA: {ALPHA}")

    # Optimal LQR (for comparison)
    v0 = 3.5
    g = 9.81
    p = {
        'IBxx': 11.3557360401, 'IBxz': -1.96756380745,
        'IByy': 12.2177848012, 'IBzz': 3.12354397008,
        'IFxx': 0.0904106601579, 'IFyy': 0.149389340425,
        'IHxx': 0.253379594731, 'IHxz': -0.0720452391817,
        'IHyy': 0.246138810935, 'IHzz': 0.0955770796289,
        'IRxx': 0.0883819364527, 'IRyy': 0.152467620286,
        'c': 0.0685808540382, 'g': 9.81, 'lam': 0.399680398707,
        'mB': 81.86, 'mF': 2.02, 'mH': 3.22, 'mR': 3.11,
        'rF': 0.34352982332, 'rR': 0.340958858855, 'w': 1.121,
        'xB': 0.289099434117, 'xH': 0.866949640247,
        'zB': -1.04029228321, 'zH': -0.748236400835,
    }
    M, C1, K0, K2 = benchmark_par_to_canonical(p)
    A35, B35 = ab_matrix(M, C1, K0, K2, v0, g)
    reorder = [0, 2, 1, 3]
    A_lqr = A35[np.ix_(reorder, reorder)]
    B_lqr = B35[reorder, 1:2]

    # Optimal LQR
    Q_opt = np.diag([1000.0, 100.0, 10.0, 1.0])
    R = np.array([[0.2]])
    P_opt = solve_continuous_are(A_lqr, B_lqr, Q_opt, R)
    K_opt = (np.linalg.inv(R) @ B_lqr.T @ P_opt).flatten()
    print(f"Optimal LQR: {K_opt}")

    # SUBOPTIMAL LQR (weakened Q - less penalty on angular velocity)
    Q_sub = np.diag([100.0, 10.0, 1.0, 0.1])
    P_sub = solve_continuous_are(A_lqr, B_lqr, Q_sub, R)
    K_sub = (np.linalg.inv(R) @ B_lqr.T @ P_sub).flatten()
    print(f"Suboptimal LQR: {K_sub}")

    # SINDy model
    sindy_data = np.load('D:/系统辨识作业/sindy_bicycle/meijaard_sindy_v35.npz')
    Xi = sindy_data['coefficients']
    A_d = np.eye(4) + Xi[1:5, :].T
    B_d = Xi[5:6, :].T
    sindy_model = SINDyWorldModel(A_d, B_d, K_sub)

    env = SuboptimalEnv(K_sub, v0=v0, max_steps=1000)
    sac = SACAgent(4, 1, device, lr=3e-4, gamma=0.99, tau=0.005)

    real_buffer = ReplayBuffer(300000)
    virtual_buffer = ReplayBuffer(300000)

    num_episodes = 1200
    batch_size = 256

    print(f"\nTraining {num_episodes} episodes, max {env.max_steps} steps")
    print(f"Base: Suboptimal LQR (Q=diag(100,10,1,0.1))")
    print(f"{'Ep':>4} {'sigma':>6} {'sig°':>5} | {'Reward':>8} {'Steps':>6} "
          f"{'Track':>7} {'EPS':>6} | {'Alpha':>6} {'Buf':>6}")

    for ep in range(num_episodes):
        sigma = compute_sigma(ep)
        state = env.reset(ep, seed=ep)
        ep_reward = 0.0
        ep_track = 0.0
        ep_eps = 0.0
        steps = 0
        while True:
            action = sac.select_action(state)
            next_state, reward, done, info = env.step(action)
            real_buffer.push(state, action, reward, next_state, float(done))
            ep_reward += reward
            ep_track += info['tracking_err']
            ep_eps += abs(action[0]) * ALPHA
            steps += 1
            state = next_state
            if steps % 10 == 0 and len(real_buffer) >= batch_size:
                vdata = sindy_virtual_rollout(sindy_model, real_buffer, sac, 10, 8)
                for (s, a, r, ns, d) in vdata:
                    virtual_buffer.push(s, a, r, ns, d)
            if len(real_buffer) >= batch_size:
                if len(virtual_buffer) >= batch_size // 2:
                    rs, ra, rr, rns, rd = real_buffer.sample(batch_size // 2)
                    vs, va, vr, vns, vd = virtual_buffer.sample(batch_size // 2)
                    ms = np.concatenate([rs, vs])
                    ma = np.concatenate([ra, va])
                    mr = np.concatenate([rr, vr])
                    mns = np.concatenate([rns, vns])
                    md = np.concatenate([rd, vd])
                    tb = ReplayBuffer(batch_size)
                    for i in range(batch_size):
                        tb.push(ms[i], ma[i], mr[i], mns[i], md[i])
                    sac.update(tb, batch_size)
                else:
                    sac.update(real_buffer, batch_size)
            if done:
                break
        avg_track = ep_track / max(steps, 1)
        avg_eps = ep_eps / max(steps, 1)
        if (ep + 1) % 5 == 0:
            alpha = sac.alpha_val
            buf = len(real_buffer) + len(virtual_buffer)
            print(f"{ep:4d} {sigma:6.3f} {math.degrees(sigma):4.1f}° | "
                  f"{ep_reward:8.1f} {steps:6d} {avg_track:7.4f} {avg_eps:6.3f} | "
                  f"{alpha:6.3f} {buf:6d}")
            sys.stdout.flush()

    torch.save({
        'policy': sac.policy.state_dict(),
        'critic1': sac.critic1.state_dict(),
        'critic2': sac.critic2.state_dict(),
        'K_sub': K_sub,
    }, 'D:/系统辨识作业/sindy_bicycle/mbpo_sac_v4.pt')
    print("\nModel saved to mbpo_sac_v4.pt")

    # Evaluation: Suboptimal vs Suboptimal+SAC vs Optimal
    print(f"\n{'='*80}")
    print("Evaluation: Optimal LQR vs Suboptimal LQR vs Suboptimal+SAC v4")
    print(f"{'='*80}")
    print(f"{'Ep':>4} {'sig°':>5} | {'Opt track':>9} {'Sub track':>9} {'SAC track':>9} | "
          f"{'Sub→Opt':>8} {'SAC→Opt':>8}")

    for ep in [1, 10, 20, 30, 40, 53, 80]:
        sigma = compute_sigma(ep)
        n = 10
        opt_track = []
        sub_track = []
        sac_track = []

        for seed in range(n):
            # Optimal LQR (ground truth)
            env_opt = SuboptimalEnv(K_opt, v0=v0, max_steps=1000)
            state = env_opt.reset(ep, seed=seed)
            total = 0.0
            for step in range(1000):
                action = np.array([0.0])
                obs, _, done, info = env_opt.step(action)
                total += info['tracking_err']
                state = obs
                if done:
                    break
            opt_track.append(total / (step + 1))

            # Suboptimal LQR
            state = env.reset(ep, seed=seed)
            total = 0.0
            for step in range(1000):
                action = np.array([0.0])
                obs, _, done, info = env.step(action)
                total += info['tracking_err']
                state = obs
                if done:
                    break
            sub_track.append(total / (step + 1))

            # Suboptimal + SAC
            state = env.reset(ep, seed=seed)
            total = 0.0
            for step in range(1000):
                action = sac.select_action(state, evaluate=True)
                obs, _, done, info = env.step(action)
                total += info['tracking_err']
                state = obs
                if done:
                    break
            sac_track.append(total / (step + 1))

        opt_m = np.mean(opt_track)
        sub_m = np.mean(sub_track)
        sac_m = np.mean(sac_track)
        sub_gap = (sub_m / opt_m - 1) * 100
        sac_gap = (sac_m / opt_m - 1) * 100

        print(f"{ep:4d} {math.degrees(sigma):4.1f}° | {opt_m:9.4f} {sub_m:9.4f} {sac_m:9.4f} | "
              f"{sub_gap:+7.1f}% {sac_gap:+7.1f}%")


if __name__ == '__main__':
    train()
