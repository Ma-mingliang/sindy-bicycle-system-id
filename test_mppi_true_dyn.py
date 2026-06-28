"""MPPI residual test using true Meijaard dynamics for planning (not SINDy).
Tests if MPPI can improve over LQR when using accurate model."""

import numpy as np
import math
from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix
from scipy.linalg import solve_continuous_are


class MPPIResidualTrue:
    """MPPI using true Meijaard dynamics for forward simulation."""

    def __init__(self, M, C1, K0, K2, v0, g, K_lqr, target_theta,
                 n_samples=32, horizon=3, noise_std=0.1,
                 temperature=0.3, epsilon_max=0.3):
        self.M = M
        self.C1 = C1
        self.K0 = K0
        self.K2 = K2
        self.v0 = v0
        self.g = g
        self.K_lqr = K_lqr
        self.target_theta = target_theta
        self.n_samples = n_samples
        self.horizon = horizon
        self.noise_std = noise_std
        self.temperature = temperature
        self.epsilon_max = epsilon_max

    def _predict_next(self, theta, delta, theta_dot, delta_dot, u, dt=1/30):
        """Predict next state using true Meijaard linearized dynamics."""
        A_m, B_m = ab_matrix(self.M, self.C1, self.K0, self.K2,
                             max(self.v0, 0.5), self.g)
        B_s = B_m[:, 1:2]
        x = np.array([[theta], [delta], [theta_dot], [delta_dot]])
        xd = A_m @ x + B_s * u
        return (theta + dt * xd[0, 0],
                delta + dt * xd[1, 0],
                theta_dot + dt * xd[2, 0],
                delta_dot + dt * xd[3, 0])

    def compute_residual(self, theta, delta, theta_dot, delta_dot):
        n_samples = self.n_samples
        horizon = self.horizon
        noise = np.random.randn(n_samples, horizon) * self.noise_std
        costs = np.zeros(n_samples)

        for k in range(n_samples):
            t, d, td, dd = theta, delta, theta_dot, delta_dot
            total_cost = 0.0
            prev_eps = 0.0

            for h in range(horizon):
                eps = noise[k, h]
                eps = np.clip(eps, -self.epsilon_max, self.epsilon_max)

                tgt = self.target_theta
                x_err = np.array([t - tgt, td, d, dd])
                u_lqr = -self.K_lqr @ x_err
                u_total = np.clip(u_lqr + eps, -10.0, 10.0)

                t, d, td, dd = self._predict_next(t, d, td, dd, u_total)

                tracking_cost = (t - tgt) ** 2
                res_cost = 0.5 * eps ** 2
                smooth_cost = 1.0 * (eps - prev_eps) ** 2
                total_cost += tracking_cost + res_cost + smooth_cost
                prev_eps = eps

                if abs(t) > math.pi / 3:
                    total_cost += 1000.0
                    break

            costs[k] = total_cost

        costs_min = np.min(costs)
        weights = np.exp(-(costs - costs_min) / self.temperature)
        weights /= (np.sum(weights) + 1e-10)
        residual = np.sum(weights * noise[:, 0])
        residual = np.clip(residual, -self.epsilon_max, self.epsilon_max)
        return residual


def compute_sigma(ep):
    return min(0.3, (1.08 ** ep) * 0.01)


def run_episode(K_lqr, ep, seed, v0=3.5, use_mppi=False,
                M=None, C1=None, K0=None, K2=None, max_steps=1000, dt=1/30):
    g = 9.81
    rng = np.random.RandomState(seed)
    sub_steps = 5
    dt_sub = dt / sub_steps

    theta = rng.uniform(-0.05, 0.05)
    theta_dot = 0.0
    delta = 0.0
    delta_dot = 0.0

    sigma = compute_sigma(ep)
    target_theta = np.clip(rng.normal(0, sigma), -math.pi / 12, math.pi / 12)

    mppi = None
    if use_mppi and M is not None:
        mppi = MPPIResidualTrue(M, C1, K0, K2, v0, g, K_lqr, target_theta,
                                n_samples=32, horizon=3, noise_std=0.1,
                                temperature=0.3, epsilon_max=0.3)

    total_track_err = 0.0
    total_res = 0.0
    fell = False

    for step in range(max_steps):
        if step % 100 == 0:
            target_theta = np.clip(rng.normal(0, sigma), -math.pi / 12, math.pi / 12)
            if mppi is not None:
                mppi.target_theta = target_theta

        x_err = np.array([theta - target_theta, theta_dot, delta, delta_dot])
        u_lqr = -K_lqr @ x_err

        if mppi is not None:
            epsilon = mppi.compute_residual(theta, delta, theta_dot, delta_dot)
        else:
            epsilon = 0.0

        u_total = np.clip(u_lqr + epsilon, -10.0, 10.0)
        total_track_err += abs(theta - target_theta)
        total_res += abs(epsilon)

        for _ in range(sub_steps):
            A_m, B_m = ab_matrix(M, C1, K0, K2, max(v0, 0.5), g)
            B_s = B_m[:, 1:2]
            x = np.array([[theta], [delta], [theta_dot], [delta_dot]])
            xd = A_m @ x + B_s * u_total
            theta += dt_sub * xd[0, 0]
            delta += dt_sub * xd[1, 0]
            theta_dot += dt_sub * xd[2, 0]
            delta_dot += dt_sub * xd[3, 0]

        if abs(theta) > math.pi / 3:
            fell = True
            break

    return {
        'steps': step + 1,
        'fell': fell,
        'avg_track_err': total_track_err / (step + 1),
        'avg_residual': total_res / (step + 1),
    }


def main():
    print("=" * 70)
    print("MPPI Residual Test with TRUE Dynamics at v=3.5 m/s")
    print("=" * 70)

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
    print(f"True LQR at v={v0}: {K_true}")

    num_seeds = 5

    print(f"\n{'='*70}")
    print(f"Progressive Disturbance (v={v0}, {num_seeds} seeds, true dynamics MPPI)")
    print(f"{'='*70}")
    print(f"{'Ep':>4} {'sigma':>6} {'sig°':>5} | "
          f"{'LQR fell%':>9} {'steps':>6} {'track':>7} | "
          f"{'LQR+MPPI fell%':>14} {'steps':>6} {'track':>7} {'eps':>6} | "
          f"{'Improve':>8}")

    for ep in [1, 10, 20, 30, 40, 53, 80]:
        sigma = compute_sigma(ep)

        lqr_fell = 0; lqr_steps = []; lqr_track = []
        mppi_fell = 0; mppi_steps = []; mppi_track = []; mppi_eps = []

        for seed in range(num_seeds):
            r = run_episode(K_true, ep, seed, v0, use_mppi=False,
                           M=M, C1=C1, K0=K0, K2=K2)
            lqr_fell += r['fell']; lqr_steps.append(r['steps']); lqr_track.append(r['avg_track_err'])

            r = run_episode(K_true, ep, seed, v0, use_mppi=True,
                           M=M, C1=C1, K0=K0, K2=K2)
            mppi_fell += r['fell']; mppi_steps.append(r['steps'])
            mppi_track.append(r['avg_track_err']); mppi_eps.append(r['avg_residual'])

        n = num_seeds
        track_improve = (1 - np.mean(mppi_track) / (np.mean(lqr_track) + 1e-10)) * 100

        print(f"{ep:4d} {sigma:6.3f} {math.degrees(sigma):4.1f}° | "
              f"{lqr_fell/n:8.0%} {np.mean(lqr_steps):6.0f} {np.mean(lqr_track):7.4f} | "
              f"{mppi_fell/n:13.0%} {np.mean(mppi_steps):6.0f} {np.mean(mppi_track):7.4f} {np.mean(mppi_eps):6.3f} | "
              f"{track_improve:+7.1f}%")


if __name__ == '__main__':
    main()
