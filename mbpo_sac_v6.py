"""MBPO-SAC v6: Graduated residual controller.

Key insight: SAC can't learn if episodes are too short (falls immediately).
Solution: Start with LQR dominant, gradually increase SAC influence.

Approach:
1. tau = (1-w)*LQR + w*SAC*alpha, where w increases from 0 to 1 over training
2. Small alpha=0.1 so SAC corrections are gentle
3. Progressive disturbance
4. SAC learns to make small improvements to LQR
"""

import sys
import os
import csv
import json
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import random
from datetime import datetime
from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix
from scipy.linalg import solve_continuous_are


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6
ALPHA = 0.5  # SAC residual scale


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


def normalize_state(dis_angle, theta, theta_dot, v):
    return np.array([
        np.clip(dis_angle, -1.57, 1.57) / 1.57,
        np.clip(theta, -1.57, 1.57) / 1.57,
        np.clip(theta_dot, -10., 10.) / 10.,
        np.clip(v, -5., 5.) / 5.,
    ], dtype=np.float32)


def compute_reward(prev_dis, new_dis, w0, terminated, theta):
    if terminated:
        return -10.0 - abs(w0)
    # Track error reduction + angle penalty + angular velocity penalty
    dis_reward = abs(prev_dis) - abs(new_dis)
    angle_penalty = -2.0 * abs(theta)
    vel_penalty = -0.05 * abs(w0)
    # Bonus for staying near upright
    upright_bonus = 0.0 if abs(theta) > 0.05 else 0.1
    return dis_reward + angle_penalty + vel_penalty + upright_bonus


class GraduatedEnv:
    """LQR base with graduated SAC residual.

    sac_weight increases from 0 to 1 over training.
    tau = (1-w)*u_lqr + w*(u_lqr + alpha*sac) = u_lqr + w*alpha*sac
    """

    def __init__(self, K_lqr, v0=3.5, dt=1/30, max_steps=1000):
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
        self.sac_weight = 0.0

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
        return self._get_obs()

    def _get_obs(self):
        dis_angle = self.target_theta - self.theta
        return normalize_state(dis_angle, self.theta, self.theta_dot, self.v0)

    def step(self, action):
        action_val = float(np.clip(action[0], -1.0, 1.0))
        self.step_count += 1
        if self.step_count <= 100:
            self.target_theta = 0.0
        elif self.step_count % 100 == 1:
            self.target_theta = np.clip(self.rng.normal(0, self.sigma),
                                        -math.pi / 12, math.pi / 12)
        epsilon = action_val * ALPHA * self.sac_weight
        x_lqr = np.array([self.theta - self.target_theta, self.theta_dot,
                          self.delta, self.delta_dot])
        u_lqr = float(-self.K_lqr @ x_lqr)
        tau = u_lqr + epsilon

        prev_dis = self.target_theta - self.theta
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
        new_dis = self.target_theta - self.theta
        terminated = abs(self.theta) > math.pi / 3
        reward = compute_reward(prev_dis, new_dis, self.theta_dot, terminated, self.theta)
        done = terminated or (self.step_count >= self.max_steps)
        return self._get_obs(), reward, done, {
            'tracking_err': abs(new_dis),
            'theta': self.theta,
            'theta_dot': self.theta_dot,
            'target_theta': self.target_theta,
            'u_lqr': u_lqr,
            'epsilon': epsilon,
        }


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
            r = compute_reward(dis, new_dis, wn, term, tn)
            ns = normalize_state(new_dis, tn, wn, vn)
            data.append((state, action, r, ns, float(term)))
            state = ns
            theta = tn
            w = wn
            dis = new_dis
            if term:
                break
    return data


def evaluate_lqr_baseline(env, n_episodes=50, max_steps=1000, sigma_schedule=None):
    """Evaluate pure LQR baseline (no SAC residual).
    sigma_schedule: list of (ep, sigma) for consistent comparison. If None, use fixed ep=80.
    """
    theta_list = []
    w_list = []
    track_list = []
    reward_list = []
    fell_count = 0
    steps_list = []

    for seed in range(n_episodes):
        ep_val = sigma_schedule[seed] if sigma_schedule else 80
        state = env.reset(ep_val, seed=seed)
        env.sac_weight = 0.0  # Pure LQR
        ep_theta = []
        ep_w = []
        ep_track = []
        ep_reward = 0.0
        fell = False

        for step in range(max_steps):
            action = np.array([0.0])
            obs, reward, done, info = env.step(action)
            ep_theta.append(abs(info['theta']))
            ep_w.append(abs(info['theta_dot']))
            ep_track.append(info['tracking_err'])
            ep_reward += reward
            if done:
                fell = abs(env.theta) > math.pi / 3
                break

        theta_list.extend(ep_theta)
        w_list.extend(ep_w)
        track_list.append(np.mean(ep_track))
        reward_list.append(ep_reward)
        steps_list.append(step + 1)
        if fell:
            fell_count += 1

    return {
        'theta_rms': float(np.sqrt(np.mean(np.array(theta_list)**2))),
        'theta_mean': float(np.mean(np.abs(np.array(theta_list)))),
        'theta_std': float(np.std(np.array(theta_list))),
        'w_mean': float(np.mean(w_list)),
        'w_std': float(np.std(w_list)),
        'tracking_err_mean': float(np.mean(track_list)),
        'tracking_err_std': float(np.std(track_list)),
        'reward_mean': float(np.mean(reward_list)),
        'reward_std': float(np.std(reward_list)),
        'steps_mean': float(np.mean(steps_list)),
        'et_count': fell_count,
        'et_rate': fell_count / n_episodes,
        'n_episodes': n_episodes,
    }


def evaluate_policy(env, sac, n_episodes=50, max_steps=1000, sigma_schedule=None):
    """Evaluate SAC policy on top of LQR."""
    theta_list = []
    w_list = []
    track_list = []
    reward_list = []
    fell_count = 0
    steps_list = []

    for seed in range(n_episodes):
        ep_val = sigma_schedule[seed] if sigma_schedule else 80
        state = env.reset(ep_val, seed=seed)
        env.sac_weight = 1.0  # Full SAC
        ep_theta = []
        ep_w = []
        ep_track = []
        ep_reward = 0.0
        fell = False

        for step in range(max_steps):
            action = sac.select_action(state, evaluate=True)
            obs, reward, done, info = env.step(action)
            ep_theta.append(abs(info['theta']))
            ep_w.append(abs(info['theta_dot']))
            ep_track.append(info['tracking_err'])
            ep_reward += reward
            if done:
                fell = abs(env.theta) > math.pi / 3
                break

        theta_list.extend(ep_theta)
        w_list.extend(ep_w)
        track_list.append(np.mean(ep_track))
        reward_list.append(ep_reward)
        steps_list.append(step + 1)
        if fell:
            fell_count += 1

    return {
        'theta_rms': float(np.sqrt(np.mean(np.array(theta_list)**2))),
        'theta_mean': float(np.mean(np.abs(np.array(theta_list)))),
        'theta_std': float(np.std(np.array(theta_list))),
        'w_mean': float(np.mean(w_list)),
        'w_std': float(np.std(w_list)),
        'tracking_err_mean': float(np.mean(track_list)),
        'tracking_err_std': float(np.std(track_list)),
        'reward_mean': float(np.mean(reward_list)),
        'reward_std': float(np.std(reward_list)),
        'steps_mean': float(np.mean(steps_list)),
        'et_count': fell_count,
        'et_rate': fell_count / n_episodes,
        'n_episodes': n_episodes,
    }


def save_training_report(output_dir, config, baseline_stats, training_log, final_eval):
    """Save training report in HRRL format."""
    report_path = os.path.join(output_dir, 'rl_training_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("强化学习训练完整报告 - 第一阶段：MBPO-SAC渐进式残差控制\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        last_20 = training_log[-20:] if len(training_log) >= 20 else training_log
        theta_rms_all = np.mean([e['theta_rms'] for e in training_log])
        theta_rms_last20 = np.mean([e['theta_rms'] for e in last_20])
        w_all = np.mean([e['w_mean'] for e in training_log])
        w_last20 = np.mean([e['w_mean'] for e in last_20])
        reward_all = np.mean([e['reward'] for e in training_log])
        reward_last20 = np.mean([e['reward'] for e in last_20])

        f.write("📊 整体统计:\n")
        f.write(f"  总Episodes: {len(training_log)}\n")
        f.write(f"  平均倾斜角误差: {theta_rms_all:.6f} ± {np.std([e['theta_rms'] for e in training_log]):.6f} rad ({np.degrees(theta_rms_all):.2f}°)\n")
        f.write(f"  平均角速度: {w_all:.6f} ± {np.std([e['w_mean'] for e in training_log]):.6f} rad/s\n")
        f.write(f"  平均奖励: {reward_all:.2f} ± {np.std([e['reward'] for e in training_log]):.2f}\n\n")

        f.write("📈 最近20集统计:\n")
        f.write(f"  平均倾斜角误差: {theta_rms_last20:.6f} rad ({np.degrees(theta_rms_last20):.2f}°)\n")
        f.write(f"  平均角速度: {w_last20:.6f} rad/s\n")
        f.write(f"  平均奖励: {reward_last20:.2f}\n\n")

        # Baseline comparison
        f.write("🔬 与LQR基线对比:\n")
        f.write(f"  基线倾斜角误差: {baseline_stats['theta_rms']:.6f} rad ({np.degrees(baseline_stats['theta_rms']):.2f}°)\n")
        f.write(f"  基线角速度: {baseline_stats['w_mean']:.6f} rad/s\n")
        f.write(f"  基线跟踪误差: {baseline_stats['tracking_err_mean']:.6f}\n")
        f.write(f"  基线参数: Q=diag(1000,100,10,1), R=0.2\n\n")

        theta_improve = (1 - theta_rms_last20 / baseline_stats['theta_rms']) * 100
        w_change = (1 - w_last20 / baseline_stats['w_mean']) * 100
        track_improve = (1 - final_eval['sac']['tracking_err_mean'] / baseline_stats['tracking_err_mean']) * 100

        f.write("  性能提升:\n")
        f.write(f"    倾斜角误差: {theta_improve:+.2f}%\n")
        f.write(f"    角速度: {w_change:+.2f}%\n")
        f.write(f"    跟踪误差: {track_improve:+.2f}%\n\n")

        # Best episode
        best_ep = min(training_log, key=lambda e: e['theta_rms'])
        f.write("⭐ 最佳Episode (最小倾斜角误差):\n")
        f.write(f"  Episode: {best_ep['episode']}\n")
        f.write(f"  倾斜角误差: {best_ep['theta_rms']:.6f} rad ({np.degrees(best_ep['theta_rms']):.2f}°)\n")
        f.write(f"  角速度: {best_ep['w_mean']:.6f} rad/s\n")
        f.write(f"  奖励: {best_ep['reward']:.2f}\n\n")

        best_improve = (1 - best_ep['theta_rms'] / baseline_stats['theta_rms']) * 100
        f.write("🏆 最佳Episode相较基线的提升:\n")
        f.write(f"  倾斜角误差提升: {best_improve:+.2f}%\n")
        if best_improve > 0:
            f.write(f"  评价: 最佳性能超越LQR基线！\n")
        else:
            f.write(f"  评价: 未能超越LQR基线\n")

    print(f"  训练报告已保存: {report_path}")


def save_baseline_stats(output_dir, baseline_stats):
    """Save LQR baseline statistics."""
    stats_path = os.path.join(output_dir, 'lqr_baseline_stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("LQR基线控制器统计信息\n")
        f.write("=" * 50 + "\n")
        f.write(f"平均倾斜角误差: {baseline_stats['theta_rms']:.6f} ± {baseline_stats['theta_std']:.6f} rad ({np.degrees(baseline_stats['theta_rms']):.2f}°)\n")
        f.write(f"平均角速度: {baseline_stats['w_mean']:.6f} ± {baseline_stats['w_std']:.6f} rad/s\n")
        f.write(f"平均跟踪误差: {baseline_stats['tracking_err_mean']:.6f}\n")
        f.write(f"平均奖励: {baseline_stats['reward_mean']:.2f}\n")
        f.write(f"平均存活步数: {baseline_stats['steps_mean']:.1f}\n")
        f.write(f"倾倒率: {baseline_stats['et_rate']:.2%}\n")
        f.write(f"评估Episodes: {baseline_stats['n_episodes']}\n\n")
        f.write("控制策略:\n")
        f.write("  LQR: K = [-103.9, -34.2, 38.0, 3.70]\n")
        f.write("  Q = diag(1000, 100, 10, 1), R = 0.2\n")
    print(f"  基线统计已保存: {stats_path}")


def save_csv_files(output_dir, baseline_errors, rl_training_errors, angle_errors):
    """Save CSV files in HRRL format."""
    # Baseline errors
    path = os.path.join(output_dir, 'lqr_baseline_errors.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tilt_error_rad', 'angular_velocity_rad_s'])
        for row in baseline_errors:
            writer.writerow([row['theta_rms'], row['w_mean']])
    print(f"  基线误差CSV已保存: {path}")

    # RL training errors
    path = os.path.join(output_dir, 'rl_training_errors.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'steps', 'avg_tilt_error', 'std_tilt_error',
                         'max_tilt_error', 'avg_angular_vel', 'std_angular_vel',
                         'reward'])
        for row in rl_training_errors:
            writer.writerow([row['episode'], row['steps'], row['theta_rms'],
                           row['theta_std'], row['theta_max'],
                           row['w_mean'], row['w_std'],
                           row['reward']])
    print(f"  RL训练误差CSV已保存: {path}")

    # Angle errors per episode
    path = os.path.join(output_dir, 'stage1_angle_error.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'theta_rms', 'angular_velocity', 'tracking_err_mean'])
        for row in angle_errors:
            writer.writerow([row['episode'], row['theta_rms'], row['w_mean'],
                           row['tracking_err_mean']])
    print(f"  角度误差CSV已保存: {path}")


def save_trajectory_data(output_dir, episode_data):
    """Save trajectory data CSV."""
    path = os.path.join(output_dir, 'stage1_data.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'theta', 'theta_dot', 'target_theta', 'tracking_err',
                         'u_lqr', 'epsilon', 'action', 'reward'])
        for row in episode_data:
            writer.writerow([row['step'], row['theta'], row['theta_dot'],
                           row['target_theta'], row['tracking_err'],
                           row['u_lqr'], row['epsilon'], row['action'], row['reward']])
    print(f"  轨迹数据已保存: {path}")


def train():
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('D:/系统辨识作业/sindy_bicycle', f'model_stage1_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("MBPO-SAC v6: Graduated residual controller")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"ALPHA: {ALPHA}")

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
    K_lqr = (np.linalg.inv(R) @ B_lqr.T @ P).flatten()
    print(f"LQR K: {K_lqr}")

    sindy_data = np.load('D:/系统辨识作业/sindy_bicycle/meijaard_sindy_v35.npz')
    Xi = sindy_data['coefficients']
    A_d = np.eye(4) + Xi[1:5, :].T
    B_d = Xi[5:6, :].T
    sindy_model = SINDyWorldModel(A_d, B_d, K_lqr)

    env = GraduatedEnv(K_lqr, v0=v0, max_steps=1000)
    sac = SACAgent(4, 1, device, lr=3e-4, gamma=0.99, tau=0.005)

    real_buffer = ReplayBuffer(300000)
    virtual_buffer = ReplayBuffer(300000)

    num_episodes = 600
    batch_size = 256
    warmup_episodes = 200
    ramp_episodes = 200

    # Training config
    config = {
        'algorithm': 'MBPO-SAC v6',
        'num_episodes': num_episodes,
        'max_steps': 1000,
        'warmup_episodes': warmup_episodes,
        'ramp_episodes': ramp_episodes,
        'alpha': ALPHA,
        'gamma': 0.99,
        'tau': 0.005,
        'lr': 3e-4,
        'batch_size': batch_size,
        'buffer_size': 300000,
        'v0': v0,
        'lqr_Q': [1000, 100, 10, 1],
        'lqr_R': 0.2,
        'lqr_K': K_lqr.tolist(),
        'device': str(device),
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ============================================================
    # LQR Baseline Evaluation
    # ============================================================
    # Use sigma schedule matching training distribution for fair comparison
    eval_sigma_schedule = [int(200 + i * 8) for i in range(50)]  # ep 200-592, ramping phase

    print(f"\n{'='*70}")
    print("LQR Baseline Evaluation (50 episodes, matched sigma schedule)")
    print(f"{'='*70}")
    baseline_stats = evaluate_lqr_baseline(env, n_episodes=50, max_steps=1000,
                                           sigma_schedule=eval_sigma_schedule)
    print(f"  theta_rms: {baseline_stats['theta_rms']:.6f} rad ({np.degrees(baseline_stats['theta_rms']):.2f}°)")
    print(f"  angular_vel: {baseline_stats['w_mean']:.6f} rad/s")
    print(f"  tracking_err: {baseline_stats['tracking_err_mean']:.6f}")
    print(f"  reward: {baseline_stats['reward_mean']:.2f}")
    print(f"  et_rate: {baseline_stats['et_rate']:.2%}")
    save_baseline_stats(output_dir, baseline_stats)

    # Collect baseline error per episode for CSV
    baseline_errors = []
    for seed in range(50):
        ep_val = eval_sigma_schedule[seed]
        state = env.reset(ep_val, seed=seed)
        env.sac_weight = 0.0
        ep_theta = []
        ep_w = []
        ep_track = []
        ep_reward = 0.0
        for step in range(1000):
            action = np.array([0.0])
            obs, reward, done, info = env.step(action)
            ep_theta.append(abs(info['theta']))
            ep_w.append(abs(info['theta_dot']))
            ep_track.append(info['tracking_err'])
            ep_reward += reward
            if done:
                break
        baseline_errors.append({
            'episode': seed,
            'theta_rms': float(np.sqrt(np.mean(np.array(ep_theta)**2))),
            'w_mean': float(np.mean(ep_w)),
            'tracking_err': float(np.mean(ep_track)),
            'reward': float(ep_reward),
            'steps': step + 1,
        })

    # ============================================================
    # Training
    # ============================================================
    print(f"\n{'='*70}")
    print(f"Training {num_episodes} episodes, max {env.max_steps} steps")
    print(f"Warmup: {warmup_episodes} eps pure LQR, then ramp to full SAC over {ramp_episodes} eps")
    print(f"{'='*70}")
    print(f"{'Ep':>4} {'sigma':>6} {'sig°':>5} | {'Reward':>8} {'Steps':>6} "
          f"{'Track':>7} {'theta_rms':>10} {'w_mean':>8} | {'Alpha':>6} {'W':>4} {'Buf':>6}")

    training_log = []
    rl_training_errors = []
    best_theta_rms = float('inf')
    best_episode = 0

    for ep in range(num_episodes):
        # Graduated schedule
        if ep < warmup_episodes:
            env.sac_weight = 0.0
        else:
            env.sac_weight = min(1.0, (ep - warmup_episodes) / ramp_episodes)

        sigma = compute_sigma(ep)
        state = env.reset(ep, seed=ep)
        ep_reward = 0.0
        ep_track = 0.0
        ep_eps = 0.0
        ep_theta = []
        ep_w = []
        steps = 0

        while True:
            action = sac.select_action(state)
            next_state, reward, done, info = env.step(action)
            real_buffer.push(state, action, reward, next_state, float(done))
            ep_reward += reward
            ep_track += info['tracking_err']
            ep_eps += abs(action[0]) * ALPHA * env.sac_weight
            ep_theta.append(abs(info['theta']))
            ep_w.append(abs(info['theta_dot']))
            steps += 1
            state = next_state
            if steps % 100 == 0 and len(real_buffer) >= batch_size and env.sac_weight > 0:
                vdata = sindy_virtual_rollout(sindy_model, real_buffer, sac, 10, 8)
                for (s, a, r, ns, d) in vdata:
                    virtual_buffer.push(s, a, r, ns, d)
            if len(real_buffer) >= batch_size and env.sac_weight > 0:
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
        ep_theta_arr = np.array(ep_theta)
        ep_w_arr = np.array(ep_w)
        ep_theta_rms = float(np.sqrt(np.mean(ep_theta_arr**2)))
        ep_theta_std = float(np.std(ep_theta_arr))
        ep_theta_max = float(np.max(ep_theta_arr))
        ep_w_mean = float(np.mean(ep_w_arr))
        ep_w_std = float(np.std(ep_w_arr))

        # Log training data
        training_log.append({
            'episode': ep,
            'theta_rms': ep_theta_rms,
            'theta_std': ep_theta_std,
            'theta_max': ep_theta_max,
            'w_mean': ep_w_mean,
            'w_std': ep_w_std,
            'tracking_err': avg_track,
            'reward': ep_reward,
            'steps': steps,
            'sigma': sigma,
            'sac_weight': env.sac_weight,
        })

        rl_training_errors.append({
            'episode': ep,
            'theta_rms': ep_theta_rms,
            'theta_std': ep_theta_std,
            'theta_max': ep_theta_max,
            'w_mean': ep_w_mean,
            'w_std': ep_w_std,
            'tracking_err': avg_track,
            'reward': ep_reward,
            'steps': steps,
            'alpha': sac.alpha_val,
            'sac_weight': env.sac_weight,
            'sigma': sigma,
        })

        # Track best model
        if ep_theta_rms < best_theta_rms and ep >= warmup_episodes:
            best_theta_rms = ep_theta_rms
            best_episode = ep
            torch.save({
                'policy': sac.policy.state_dict(),
                'critic1': sac.critic1.state_dict(),
                'critic2': sac.critic2.state_dict(),
                'config': config,
                'episode': ep,
                'theta_rms': ep_theta_rms,
            }, os.path.join(output_dir, 'stage1_agent_best.pt'))

        if (ep + 1) % 5 == 0:
            alpha = sac.alpha_val
            buf = len(real_buffer) + len(virtual_buffer)
            print(f"{ep:4d} {sigma:6.3f} {math.degrees(sigma):4.1f}° | "
                  f"{ep_reward:8.1f} {steps:6d} {avg_track:7.4f} {ep_theta_rms:10.6f} {ep_w_mean:8.4f} | "
                  f"{alpha:6.3f} {env.sac_weight:4.2f} {buf:6d}")
            sys.stdout.flush()

    # Save final model
    torch.save({
        'policy': sac.policy.state_dict(),
        'critic1': sac.critic1.state_dict(),
        'critic2': sac.critic2.state_dict(),
        'config': config,
        'episode': num_episodes,
    }, os.path.join(output_dir, 'stage1_agent_final.pt'))
    print(f"\n模型已保存: {output_dir}/stage1_agent_best.pt, stage1_agent_final.pt")

    # ============================================================
    # Final Evaluation
    # ============================================================
    print(f"\n{'='*70}")
    print("Final Evaluation: LQR vs LQR+SAC (50 episodes)")
    print(f"{'='*70}")

    # Load best model
    best_ckpt = torch.load(os.path.join(output_dir, 'stage1_agent_best.pt'))
    sac.policy.load_state_dict(best_ckpt['policy'])
    sac.critic1.load_state_dict(best_ckpt['critic1'])
    sac.critic2.load_state_dict(best_ckpt['critic2'])

    final_lqr = evaluate_lqr_baseline(env, n_episodes=50, max_steps=1000,
                                      sigma_schedule=eval_sigma_schedule)
    final_sac = evaluate_policy(env, sac, n_episodes=50, max_steps=1000,
                                sigma_schedule=eval_sigma_schedule)

    final_eval = {'lqr': final_lqr, 'sac': final_sac}
    with open(os.path.join(output_dir, 'final_eval.json'), 'w') as f:
        json.dump(final_eval, f, indent=2)

    print(f"\n{'指标':<20} {'LQR':>12} {'SAC':>12} {'提升':>10}")
    print("-" * 56)
    print(f"{'theta_rms (rad)':<20} {final_lqr['theta_rms']:>12.6f} {final_sac['theta_rms']:>12.6f} "
          f"{(1-final_sac['theta_rms']/final_lqr['theta_rms'])*100:>+9.2f}%")
    print(f"{'theta_rms (°)':<20} {np.degrees(final_lqr['theta_rms']):>12.2f} {np.degrees(final_sac['theta_rms']):>12.2f} "
          f"{(1-final_sac['theta_rms']/final_lqr['theta_rms'])*100:>+9.2f}%")
    print(f"{'angular_vel':<20} {final_lqr['w_mean']:>12.6f} {final_sac['w_mean']:>12.6f} "
          f"{(1-final_sac['w_mean']/final_lqr['w_mean'])*100:>+9.2f}%")
    print(f"{'tracking_err':<20} {final_lqr['tracking_err_mean']:>12.6f} {final_sac['tracking_err_mean']:>12.6f} "
          f"{(1-final_sac['tracking_err_mean']/final_lqr['tracking_err_mean'])*100:>+9.2f}%")
    print(f"{'reward':<20} {final_lqr['reward_mean']:>12.2f} {final_sac['reward_mean']:>12.2f} "
          f"{(final_sac['reward_mean']-final_lqr['reward_mean']):>+9.2f}")
    print(f"{'et_rate':<20} {final_lqr['et_rate']:>12.2%} {final_sac['et_rate']:>12.2%}")

    # Save trajectory data from best episode
    print(f"\n保存最佳episode轨迹数据...")
    state = env.reset(best_episode, seed=best_episode)
    env.sac_weight = 1.0
    trajectory = []
    for step in range(1000):
        action = sac.select_action(state, evaluate=True)
        obs, reward, done, info = env.step(action)
        trajectory.append({
            'step': step,
            'theta': info['theta'],
            'theta_dot': info['theta_dot'],
            'target_theta': info['target_theta'],
            'tracking_err': info['tracking_err'],
            'u_lqr': info['u_lqr'],
            'epsilon': info['epsilon'],
            'action': float(action[0]),
            'reward': reward,
        })
        state = obs
        if done:
            break
    save_trajectory_data(output_dir, trajectory)

    # Save angle errors per episode
    angle_errors = []
    for log in training_log:
        angle_errors.append({
            'episode': log['episode'],
            'theta_rms': log['theta_rms'],
            'w_mean': log['w_mean'],
            'tracking_err_mean': log['tracking_err'],
        })

    # Save CSV files
    save_csv_files(output_dir, baseline_errors, rl_training_errors, angle_errors)

    # Save training report
    save_training_report(output_dir, config, baseline_stats, training_log, final_eval)

    print(f"\n{'='*70}")
    print(f"所有输出已保存到: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    train()
