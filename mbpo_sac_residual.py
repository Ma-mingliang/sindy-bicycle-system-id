"""Model-Based SAC Residual Controller (MBPO-style).

Follows HRRL Attitude_control residual injection pattern:
  target_handlebar = 0.1 * action + lqr_control
  action ∈ [-1, 1], scaled by alpha=0.1

Uses SINDy world model (v=3.5) for virtual rollouts to accelerate SAC training.

State (4D, normalized): [dis_angle, theta, theta_dot, v]
  - matches HRRL env.py:181 observation space
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
ALPHA = 0.15  # HRRL residual scaling (env.py:145)


# ============================================================
# SAC Networks
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def sample_batch(self, batch_size):
        return self.sample(batch_size)

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


# ============================================================
# SAC Agent
# ============================================================

class SACAgent:
    def __init__(self, state_dim, action_dim, device,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha_init=0.2,
                 auto_alpha=True):
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

        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.target_entropy = -action_dim
        else:
            self.alpha = alpha_init

    @property
    def alpha_val(self):
        if self.auto_alpha:
            return min(self.log_alpha.exp().item(), 1.0)  # cap alpha
        return self.alpha

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

        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # Cap log_alpha to prevent explosion
            self.log_alpha.data.clamp_(-5, 0)

        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha_val,
        }


# ============================================================
# SINDy World Model (v=3.5)
# ============================================================

class SINDyWorldModel:
    """SINDy discrete-time model for virtual rollouts.

    State in SINDy order: [phi, delta, phi_dot, delta_dot]
    s_next = A_d @ s + B_d * tau
    """

    def __init__(self, A_d, B_d, K_lqr):
        self.A_d = A_d
        self.B_d = B_d
        self.K_lqr = K_lqr
        # SINDy order: [phi, delta, phi_dot, delta_dot]
        # LQR order: [phi, phi_dot, delta, delta_dot]
        self.sindy_to_lqr = [0, 2, 1, 3]
        self.lqr_to_sindy = [0, 2, 1, 3]

    def predict_next(self, theta, delta, theta_dot, delta_dot,
                     action, target_theta, dt=1/30):
        """Predict next state given current state and RL action.

        HRRL residual injection:
          epsilon = action * ALPHA
          u_total = u_lqr + epsilon
        where u_lqr = -K @ [theta-target_theta, theta_dot, delta, delta_dot]
        """
        epsilon = action * ALPHA
        x_lqr = np.array([theta - target_theta, theta_dot, delta, delta_dot])
        u_lqr = float(-self.K_lqr @ x_lqr)
        tau = u_lqr + epsilon

        # Convert to SINDy order
        s_sindy = np.array([theta, delta, theta_dot, delta_dot])
        s_next_sindy = self.A_d @ s_sindy + self.B_d.flatten() * tau

        return s_next_sindy[0], s_next_sindy[1], s_next_sindy[2], s_next_sindy[3]


# ============================================================
# HRRL-style Environment (Meijaard dynamics)
# ============================================================

def compute_sigma(ep):
    return min(0.3, (1.08 ** ep) * 0.01)


def compute_reward(prev_dis_angle, new_dis_angle, theta_dot, terminated):
    """HRRL attitude reward (env.py:190-202)."""
    if terminated:
        return -1.0 - 0.01 * abs(theta_dot)
    elif abs(new_dis_angle) < 0.002:
        return 0.1 - abs(theta_dot)
    else:
        return abs(prev_dis_angle) - abs(new_dis_angle)


def normalize_state(dis_angle, theta, theta_dot, v):
    """HRRL-style state normalization (env.py:172-178)."""
    return np.array([
        np.clip(dis_angle, -1.57, 1.57) / 1.57,
        np.clip(theta, -1.57, 1.57) / 1.57,
        np.clip(theta_dot, -10., 10.) / 10.,
        np.clip(v, -5., 5.) / 5.,
    ], dtype=np.float32)


class MeijaardAttitudeEnv:
    """Meijaard bicycle with HRRL-style residual injection."""

    def __init__(self, K_lqr, v0=3.5, dt=1/30, max_steps=2000):
        self.K_lqr = K_lqr
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
        self.theta_old = self.theta
        return self._get_obs()

    def _get_obs(self):
        dis_angle = self.target_theta - self.theta
        return normalize_state(dis_angle, self.theta, self.theta_dot, self.v0)

    def step(self, action):
        """HRRL-style step: action ∈ [-1,1], residual on steering angle."""
        action = float(np.clip(action[0], -1.0, 1.0))
        self.step_count += 1

        # Progressive disturbance (env.py:249-253)
        if self.step_count % 100 == 1:
            self.target_theta = np.clip(self.rng.normal(0, self.sigma),
                                        -math.pi / 12, math.pi / 12)

        # HRRL residual injection (env.py:269-301)
        # u_lqr = -K @ [theta-target_theta, theta_dot, delta, delta_dot]
        # epsilon = action * ALPHA
        # u_total = u_lqr + epsilon
        epsilon = action * ALPHA
        x_lqr = np.array([self.theta - self.target_theta, self.theta_dot,
                          self.delta, self.delta_dot])
        u_lqr = float(-self.K_lqr @ x_lqr)
        tau = u_lqr + epsilon

        # Reward before step
        prev_dis_angle = self.target_theta - self.theta

        # Dynamics (precompute A_m, B_m once per step)
        dt_sub = self.dt / self.sub_steps
        A_m, B_m = ab_matrix(self.M, self.C1, self.K0, self.K2,
                             max(self.v0, 0.5), self.g)
        B_s = B_m[:, 1:2]
        for _ in range(self.sub_steps):
            x = np.array([[self.theta], [self.delta],
                          [self.theta_dot], [self.delta_dot]])
            xd = A_m @ x + B_s * tau
            self.theta += dt_sub * xd[0, 0]
            self.delta += dt_sub * xd[1, 0]
            self.theta_dot += dt_sub * xd[2, 0]
            self.delta_dot += dt_sub * xd[3, 0]

        # Reward after step
        new_dis_angle = self.target_theta - self.theta
        terminated = abs(self.theta) > math.pi / 3
        reward = compute_reward(prev_dis_angle, new_dis_angle,
                                self.theta_dot, terminated)

        done = terminated or (self.step_count >= self.max_steps)
        return self._get_obs(), reward, done, {'tracking_err': abs(new_dis_angle)}


# ============================================================
# MBPO Virtual Rollout
# ============================================================

def sindy_virtual_rollout(sindy_model, replay_buffer, sac_agent,
                          n_rollouts=10, horizon=5):
    """Generate virtual transitions using SINDy world model.

    Follows MBPO: sample initial states from real buffer, roll out
    with current SAC policy using SINDy dynamics.
    """
    virtual_data = []

    for _ in range(n_rollouts):
        # Sample initial state from real buffer
        if len(replay_buffer) < 32:
            break
        states, _, _, _, _ = replay_buffer.sample(1)
        state = states[0]

        # Denormalize: [dis_angle, theta, theta_dot, v]
        dis_angle = state[0] * 1.57
        theta = state[1] * 1.57
        theta_dot = state[2] * 10.0
        v = state[3] * 5.0
        # Approximate target_theta from dis_angle
        target_theta = dis_angle + theta

        for h in range(horizon):
            # SAC selects action
            action = sac_agent.select_action(state)

            # SINDy predicts next state
            theta_n, delta_n, theta_dot_n, v_n = sindy_model.predict_next(
                theta, 0.0, theta_dot, v, action[0], target_theta)

            # Compute reward
            new_dis_angle = target_theta - theta_n
            terminated = abs(theta_n) > math.pi / 3
            reward = compute_reward(dis_angle, new_dis_angle,
                                    theta_dot_n, terminated)

            next_state = normalize_state(new_dis_angle, theta_n, theta_dot_n, v_n)

            virtual_data.append((state, action, reward, next_state, float(terminated)))

            state = next_state
            theta = theta_n
            theta_dot = theta_dot_n
            dis_angle = new_dis_angle

            if terminated:
                break

    return virtual_data


# ============================================================
# Training
# ============================================================

def train_mbpo_sac():
    print("=" * 70)
    print("MBPO-SAC Residual Controller at v=3.5 m/s")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # LQR at v=3.5
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
    Q = np.diag([1000.0, 100.0, 10.0, 1.0])
    R = np.array([[0.2]])
    P = solve_continuous_are(A_lqr, B_lqr, Q, R)
    K_true = (np.linalg.inv(R) @ B_lqr.T @ P).flatten()
    print(f"True LQR: {K_true}")

    # SINDy model
    sindy_data = np.load('D:/系统辨识作业/sindy_bicycle/meijaard_sindy_v35.npz')
    Xi = sindy_data['coefficients']
    A_lin = Xi[1:5, :].T
    B_lin = Xi[5:6, :].T
    A_d = np.eye(4) + A_lin
    B_d = B_lin
    sindy_model = SINDyWorldModel(A_d, B_d, K_true)
    print(f"SINDy eigenvalues: {np.linalg.eigvals(A_d)}")

    # Environment
    env = MeijaardAttitudeEnv(K_true, v0=v0)

    # SAC
    state_dim = 4  # [dis_angle, theta, theta_dot, v]
    action_dim = 1
    sac = SACAgent(state_dim, action_dim, device, lr=3e-4,
                   gamma=0.99, tau=0.005, auto_alpha=True)

    real_buffer = ReplayBuffer(100000)
    virtual_buffer = ReplayBuffer(100000)

    num_episodes = 300
    batch_size = 128
    updates_per_step = 1
    rollout_interval = 20  # virtual rollout every N steps
    n_virtual_rollouts = 5
    rollout_horizon = 5

    print(f"\nTraining {num_episodes} episodes...")
    print(f"MBPO: {n_virtual_rollouts} rollouts x {rollout_horizon} steps "
          f"every {rollout_interval} real steps")
    print(f"{'Ep':>4} {'sigma':>6} {'sig°':>5} | {'Reward':>8} {'Steps':>6} "
          f"{'Track':>7} {'EPS':>6} | {'Alpha':>6} {'Buf':>6}")

    for ep in range(num_episodes):
        sigma = compute_sigma(ep)
        state = env.reset(ep, seed=ep)
        episode_reward = 0.0
        episode_track = 0.0
        episode_eps = 0.0
        steps = 0

        while True:
            action = sac.select_action(state)
            next_state, reward, done, info = env.step(action)
            real_buffer.push(state, action, reward, next_state, float(done))

            episode_reward += reward
            episode_track += info['tracking_err']
            episode_eps += abs(action[0]) * ALPHA
            steps += 1
            state = next_state

            # MBPO: virtual rollouts
            if steps % rollout_interval == 0 and len(real_buffer) >= batch_size:
                virtual_data = sindy_virtual_rollout(
                    sindy_model, real_buffer, sac,
                    n_rollouts=n_virtual_rollouts, horizon=rollout_horizon)
                for (s, a, r, ns, d) in virtual_data:
                    virtual_buffer.push(s, a, r, ns, d)

            # Update SAC on mixed data
            if len(real_buffer) >= batch_size:
                for _ in range(updates_per_step):
                    # 50/50 mix of real and virtual
                    if len(virtual_buffer) >= batch_size // 2:
                        # Sample from both buffers
                        real_s, real_a, real_r, real_ns, real_d = real_buffer.sample(batch_size // 2)
                        vir_s, vir_a, vir_r, vir_ns, vir_d = virtual_buffer.sample(batch_size // 2)
                        mixed_s = np.concatenate([real_s, vir_s])
                        mixed_a = np.concatenate([real_a, vir_a])
                        mixed_r = np.concatenate([real_r, vir_r])
                        mixed_ns = np.concatenate([real_ns, vir_ns])
                        mixed_d = np.concatenate([real_d, vir_d])

                        # Create temp buffer with mixed data
                        temp_buffer = ReplayBuffer(batch_size)
                        for i in range(batch_size):
                            temp_buffer.push(mixed_s[i], mixed_a[i], mixed_r[i],
                                             mixed_ns[i], mixed_d[i])
                        sac.update(temp_buffer, batch_size)
                    else:
                        sac.update(real_buffer, batch_size)

            if done:
                break

        avg_track = episode_track / max(steps, 1)
        avg_eps = episode_eps / max(steps, 1)

        if (ep + 1) % 5 == 0:
            alpha = sac.alpha_val
            total_buf = len(real_buffer) + len(virtual_buffer)
            print(f"{ep:4d} {sigma:6.3f} {math.degrees(sigma):4.1f}° | "
                  f"{episode_reward:8.1f} {steps:6d} {avg_track:7.4f} {avg_eps:6.3f} | "
                  f"{alpha:6.3f} {total_buf:6d}")
            sys.stdout.flush()

    # Save
    torch.save({
        'policy': sac.policy.state_dict(),
        'critic1': sac.critic1.state_dict(),
        'critic2': sac.critic2.state_dict(),
    }, 'D:/系统辨识作业/sindy_bicycle/mbpo_sac_residual_v35.pt')
    print("\nModel saved to mbpo_sac_residual_v35.pt")

    # ============================================================
    # Evaluation
    # ============================================================
    print(f"\n{'='*70}")
    print("Evaluation: LQR vs LQR+MBPO-SAC")
    print(f"{'='*70}")
    print(f"{'Ep':>4} {'sigma':>6} {'sig°':>5} | "
          f"{'LQR fell%':>9} {'steps':>6} {'track':>7} | "
          f"{'SAC fell%':>10} {'steps':>6} {'track':>7} {'eps':>6} | "
          f"{'Improve':>8}")

    for ep in [1, 10, 20, 30, 40, 53, 80]:
        sigma = compute_sigma(ep)
        num_seeds = 10

        lqr_fell = 0; lqr_steps = []; lqr_track = []
        sac_fell = 0; sac_steps = []; sac_track = []; sac_eps = []

        for seed in range(num_seeds):
            # LQR only
            state = env.reset(ep, seed=seed)
            total_track = 0.0
            fell = False
            for step in range(2000):
                x_err = np.array([env.theta - env.target_theta, env.theta_dot,
                                  env.delta, env.delta_dot])
                u = -K_true @ x_err
                u = np.clip(u, -10.0, 10.0)
                total_track += abs(env.target_theta - env.theta)
                dt_sub = env.dt / env.sub_steps
                for _ in range(env.sub_steps):
                    A_m, B_m = ab_matrix(env.M, env.C1, env.K0, env.K2,
                                         max(env.v0, 0.5), env.g)
                    B_s = B_m[:, 1:2]
                    x = np.array([[env.theta], [env.delta],
                                  [env.theta_dot], [env.delta_dot]])
                    xd = A_m @ x + B_s * u
                    env.theta += dt_sub * xd[0, 0]
                    env.delta += dt_sub * xd[1, 0]
                    env.theta_dot += dt_sub * xd[2, 0]
                    env.delta_dot += dt_sub * xd[3, 0]
                if abs(env.theta) > math.pi / 3:
                    fell = True
                    break
                if (step + 1) % 100 == 0:
                    env.target_theta = np.clip(env.rng.normal(0, sigma),
                                               -math.pi / 12, math.pi / 12)
            lqr_fell += fell
            lqr_steps.append(step + 1)
            lqr_track.append(total_track / (step + 1))

            # LQR + SAC
            state = env.reset(ep, seed=seed)
            total_track = 0.0
            total_eps = 0.0
            fell = False
            for step in range(2000):
                action = sac.select_action(state, evaluate=True)
                obs, _, done, info = env.step(action)
                total_track += info['tracking_err']
                total_eps += abs(action[0]) * ALPHA
                state = obs
                if done:
                    fell = abs(env.theta) > math.pi / 3
                    break
            sac_fell += fell
            sac_steps.append(step + 1)
            sac_track.append(total_track / (step + 1))
            sac_eps.append(total_eps / (step + 1))

        n = num_seeds
        track_improve = (1 - np.mean(sac_track) / (np.mean(lqr_track) + 1e-10)) * 100

        print(f"{ep:4d} {sigma:6.3f} {math.degrees(sigma):4.1f}° | "
              f"{lqr_fell/n:8.0%} {np.mean(lqr_steps):6.0f} {np.mean(lqr_track):7.4f} | "
              f"{sac_fell/n:9.0%} {np.mean(sac_steps):6.0f} {np.mean(sac_track):7.4f} {np.mean(sac_eps):6.3f} | "
              f"{track_improve:+7.1f}%")


if __name__ == '__main__':
    train_mbpo_sac()
