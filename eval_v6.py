"""Evaluate v6 MBPO-SAC model against LQR baseline."""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix
from scipy.linalg import solve_continuous_are

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6
ALPHA = 0.1


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


def compute_sigma(ep):
    return min(0.3, (1.08 ** ep) * 0.01)


def normalize_state(dis_angle, theta, theta_dot, v):
    return np.array([
        np.clip(dis_angle, -1.57, 1.57) / 1.57,
        np.clip(theta, -1.57, 1.57) / 1.57,
        np.clip(theta_dot, -10., 10.) / 10.,
        np.clip(v, -5., 5.) / 5.,
    ], dtype=np.float32)


class GraduatedEnv:
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
        self.sac_weight = 1.0

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
        done = terminated or (self.step_count >= self.max_steps)
        return self._get_obs(), 0, done, {'tracking_err': abs(new_dis)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Load SAC model
    policy = GaussianPolicy(4, 1).to(device)
    checkpoint = torch.load('D:/系统辨识作业/sindy_bicycle/mbpo_sac_v6.pt',
                            map_location=device)
    policy.load_state_dict(checkpoint['policy'])
    policy.eval()

    def select_action(state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        _, _, action = policy.sample(state_t)
        return action.detach().cpu().numpy()[0]

    env = GraduatedEnv(K_true, v0=v0, max_steps=1000)
    env.sac_weight = 1.0

    print(f"\n{'='*70}")
    print("Evaluation: LQR vs LQR+SAC v6 (graduated, full weight)")
    print(f"{'='*70}")
    print(f"{'Ep':>4} {'sigma':>6} {'sig°':>5} | "
          f"{'LQR fell%':>9} {'steps':>6} {'track':>7} | "
          f"{'SAC fell%':>9} {'steps':>6} {'track':>7} | "
          f"{'Improve':>8}")

    results = []
    for ep in [1, 10, 20, 30, 40, 53, 80]:
        sigma = compute_sigma(ep)
        num_seeds = 20

        lqr_fell = 0; lqr_steps = []; lqr_track = []
        sac_fell = 0; sac_steps = []; sac_track = []

        for seed in range(num_seeds):
            state = env.reset(ep, seed=seed)
            total_track = 0.0; fell = False
            for step in range(1000):
                action = np.array([0.0])
                obs, _, done, info = env.step(action)
                total_track += info['tracking_err']
                state = obs
                if done:
                    fell = abs(env.theta) > math.pi / 3
                    break
            lqr_fell += fell; lqr_steps.append(step + 1)
            lqr_track.append(total_track / (step + 1))

            state = env.reset(ep, seed=seed)
            total_track = 0.0; fell = False
            for step in range(1000):
                action = select_action(state)
                obs, _, done, info = env.step(action)
                total_track += info['tracking_err']
                state = obs
                if done:
                    fell = abs(env.theta) > math.pi / 3
                    break
            sac_fell += fell; sac_steps.append(step + 1)
            sac_track.append(total_track / (step + 1))

        n = num_seeds
        track_improve = (1 - np.mean(sac_track) / (np.mean(lqr_track) + 1e-10)) * 100
        results.append({
            'ep': ep, 'sigma': sigma,
            'lqr_track': np.mean(lqr_track), 'sac_track': np.mean(sac_track),
            'improve': track_improve,
            'lqr_fell': lqr_fell/n, 'sac_fell': sac_fell/n
        })

        print(f"{ep:4d} {sigma:6.3f} {math.degrees(sigma):4.1f}° | "
              f"{lqr_fell/n:8.0%} {np.mean(lqr_steps):6.0f} {np.mean(lqr_track):7.4f} | "
              f"{sac_fell/n:8.0%} {np.mean(sac_steps):6.0f} {np.mean(sac_track):7.4f} | "
              f"{track_improve:+7.1f}%")

    return results


if __name__ == '__main__':
    main()
