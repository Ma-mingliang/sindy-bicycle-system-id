"""Verify LQR with HRRL Stage 1 progressive disturbance on Meijaard bicycle.

The actual HRRL disturbance is NOT wind/impulse forces. It's a progressively
increasing target lean angle:

  sigma = min(0.3, 1.08^episode_count * 0.01)
  target_theta ~ N(0, sigma)   # sampled every 100 steps

LQR must track this changing target_theta while keeping the bicycle stable.
RL residual helps when target_theta becomes large enough to challenge LQR.
"""

import numpy as np
import math
from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix


def compute_hrrl_sigma(episode_count):
    """HRRL progressive disturbance sigma formula."""
    return min(0.3, (1.08 ** episode_count) * 0.01)


def run_hrrl_stage1_episode(K_lqr, episode_count, v0=5.5, max_steps=1000,
                             dt=1/30, seed=0, alpha=0.15, use_residual=False):
    """Run one HRRL Stage 1 episode with Meijaard dynamics.

    Args:
        K_lqr: LQR gains [phi, phi_dot, delta, delta_dot]
        episode_count: episode number (determines sigma)
        v0: forward speed
        max_steps: max steps per episode
        dt: time step
        seed: random seed
        alpha: residual scaling factor
        use_residual: whether to add RL residual (for comparison)

    Returns: dict with episode results
    """
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
    g = 9.81
    M, C1, K0, K2 = benchmark_par_to_canonical(p)

    rng = np.random.RandomState(seed)
    sub_steps = 5
    dt_sub = dt / sub_steps

    # Initial state
    theta = rng.uniform(-0.05, 0.05)
    theta_dot = 0.0
    delta = 0.0
    delta_dot = 0.0

    # Progressive disturbance sigma
    sigma = compute_hrrl_sigma(episode_count)

    # Initial target_theta
    target_theta = rng.normal(0.0, sigma)
    target_theta = np.clip(target_theta, -math.pi / 12, math.pi / 12)

    max_phi = 0.0
    sum_tracking_error = 0.0
    fell = False

    for step in range(max_steps):
        # Progressive disturbance: resample target_theta every 100 steps
        if step % 100 == 0:
            target_theta = rng.normal(0.0, sigma)
            target_theta = np.clip(target_theta, -math.pi / 12, math.pi / 12)

        # LQR control: track target_theta
        x_lqr = np.array([
            theta - target_theta,  # error: current - target
            theta_dot,
            delta,
            delta_dot,
        ])
        u_lqr = -K_lqr @ x_lqr

        # RL residual (simulated as small random correction)
        if use_residual:
            epsilon = rng.uniform(-1, 1) * alpha
        else:
            epsilon = 0.0

        u_total = u_lqr + epsilon
        u_total = np.clip(u_total, -10.0, 10.0)

        max_phi = max(max_phi, abs(theta))
        sum_tracking_error += abs(theta - target_theta)

        # RK4 with Meijaard dynamics
        for _ in range(sub_steps):
            A_m, B_m = ab_matrix(M, C1, K0, K2, max(v0, 0.5), g)
            B_s = B_m[:, 1:2]
            x = np.array([[theta], [delta], [theta_dot], [delta_dot]])
            xd = A_m @ x + B_s * u_total
            theta += dt_sub * xd[0, 0]
            delta += dt_sub * xd[1, 0]
            theta_dot += dt_sub * xd[2, 0]
            delta_dot += dt_sub * xd[3, 0]

        # Termination: |theta| > 60 deg (matching HRRL)
        if abs(theta) > math.pi / 3:
            fell = True
            break

    return {
        'episode': episode_count,
        'sigma': sigma,
        'steps': step + 1,
        'time': (step + 1) * dt,
        'fell': fell,
        'max_phi': max_phi,
        'avg_tracking_error': sum_tracking_error / (step + 1),
    }


def main():
    print("=" * 70)
    print("HRRL Stage 1 Progressive Disturbance Verification (Meijaard)")
    print("=" * 70)

    # Load SINDy LQR
    data = np.load('D:/系统辨识作业/sindy_bicycle/sindy_lqr.npz')
    K_sindy = data['K_lqr']
    print(f"SINDy LQR: {K_sindy}")

    # True Meijaard LQR for comparison
    from scipy.linalg import solve_continuous_are
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
    A5, B5 = ab_matrix(M, C1, K0, K2, 5.5, 9.81)
    reorder = [0, 2, 1, 3]
    A_lqr = A5[np.ix_(reorder, reorder)]
    B_lqr = B5[reorder, 1:2]
    Q = np.diag([1000.0, 100.0, 10.0, 1.0])
    R = np.array([[0.2]])
    P = solve_continuous_are(A_lqr, B_lqr, Q, R)
    K_true = (np.linalg.inv(R) @ B_lqr.T @ P).flatten()
    print(f"True  LQR: {K_true}")

    v0 = 5.5
    num_seeds = 10

    # Test across episodes (sigma grows)
    episodes = [1, 5, 10, 15, 20, 30, 40, 53, 60, 80, 100]

    print(f"\n{'='*70}")
    print(f"Progressive Disturbance: sigma = min(0.3, 1.08^ep * 0.01)")
    print(f"Speed: {v0} m/s, {num_seeds} seeds per episode, max 1000 steps")
    print(f"{'='*70}")
    print(f"{'Ep':>4} {'sigma':>7} {'sigma°':>7} | "
          f"{'SINDy fell%':>11} {'steps':>6} {'max_phi':>8} {'track_err':>10} | "
          f"{'True fell%':>10} {'steps':>6} {'max_phi':>8} {'track_err':>10}")

    for ep in episodes:
        sigma = compute_hrrl_sigma(ep)

        s_fell, s_steps, s_phi, s_err = 0, [], [], []
        t_fell, t_steps, t_phi, t_err = 0, [], [], []

        for seed in range(num_seeds):
            r = run_hrrl_stage1_episode(K_sindy, ep, v0, 1000, 1/30, seed)
            if r['fell']: s_fell += 1
            s_steps.append(r['steps'])
            s_phi.append(r['max_phi'])
            s_err.append(r['avg_tracking_error'])

            r = run_hrrl_stage1_episode(K_true, ep, v0, 1000, 1/30, seed)
            if r['fell']: t_fell += 1
            t_steps.append(r['steps'])
            t_phi.append(r['max_phi'])
            t_err.append(r['avg_tracking_error'])

        print(f"{ep:4d} {sigma:7.4f} {math.degrees(sigma):6.1f}° | "
              f"{s_fell/num_seeds:10.0%} {np.mean(s_steps):6.0f} {np.mean(s_phi):8.4f} {np.mean(s_err):10.4f} | "
              f"{t_fell/num_seeds:9.0%} {np.mean(t_steps):6.0f} {np.mean(t_phi):8.4f} {np.mean(t_err):10.4f}")

    # === With RL residual (simulated) ===
    print(f"\n{'='*70}")
    print(f"With Simulated RL Residual (alpha=0.15)")
    print(f"{'='*70}")
    print(f"{'Ep':>4} {'sigma':>7} {'sigma°':>7} | "
          f"{'SINDy fell%':>11} {'steps':>6} {'max_phi':>8} {'track_err':>10} | "
          f"{'True fell%':>10} {'steps':>6} {'max_phi':>8} {'track_err':>10}")

    for ep in episodes:
        sigma = compute_hrrl_sigma(ep)

        s_fell, s_steps, s_phi, s_err = 0, [], [], []
        t_fell, t_steps, t_phi, t_err = 0, [], [], []

        for seed in range(num_seeds):
            r = run_hrrl_stage1_episode(K_sindy, ep, v0, 1000, 1/30, seed,
                                         use_residual=True)
            if r['fell']: s_fell += 1
            s_steps.append(r['steps'])
            s_phi.append(r['max_phi'])
            s_err.append(r['avg_tracking_error'])

            r = run_hrrl_stage1_episode(K_true, ep, v0, 1000, 1/30, seed,
                                         use_residual=True)
            if r['fell']: t_fell += 1
            t_steps.append(r['steps'])
            t_phi.append(r['max_phi'])
            t_err.append(r['avg_tracking_error'])

        print(f"{ep:4d} {sigma:7.4f} {math.degrees(sigma):6.1f}° | "
              f"{s_fell/num_seeds:10.0%} {np.mean(s_steps):6.0f} {np.mean(s_phi):8.4f} {np.mean(s_err):10.4f} | "
              f"{t_fell/num_seeds:9.0%} {np.mean(t_steps):6.0f} {np.mean(t_phi):8.4f} {np.mean(t_err):10.4f}")

    print(f"\n{'='*70}")
    print("Conclusion")
    print(f"{'='*70}")
    print("""
HRRL Stage 1 progressive disturbance:
  - sigma grows as 1.08^ep * 0.01, capped at 0.3 rad (17.2°)
  - target_theta ~ N(0, sigma), resampled every 100 steps
  - LQR tracks target_theta; RL residual adds small corrections

A suitable LQR should:
  - Survive early episodes (small sigma) with low tracking error
  - Start struggling around ep 30-50 (sigma ~ 0.1-0.3 rad)
  - Leave room for RL residual to improve tracking in later episodes
""")


if __name__ == '__main__':
    main()
