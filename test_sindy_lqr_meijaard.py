"""Test SINDy-derived LQR on true Meijaard dynamics.

Verifies that LQR designed from SINDy model works on the real bicycle.
"""

import numpy as np
import math
from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix


def run_meijaard_lqr(K_lqr, v0=5.5, theta0=0.02, num_seeds=50,
                     max_steps=3000, dt=1/30):
    """Run Meijaard bicycle with SINDy-derived LQR.

    Args:
        K_lqr: LQR gains [phi, phi_dot, delta, delta_dot]
        v0: forward speed
        theta0: initial lean angle (or range)
        num_seeds: number of random seeds
        max_steps: max steps
        dt: time step

    Returns: list of result dicts
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

    sub_steps = 5
    dt_sub = dt / sub_steps
    results = []

    for seed in range(num_seeds):
        rng = np.random.RandomState(seed)

        # Initial state
        theta = theta0 + rng.uniform(-0.01, 0.01) if isinstance(theta0, float) else rng.uniform(*theta0)
        theta_dot = rng.uniform(-0.02, 0.02)
        delta = rng.uniform(-0.01, 0.01)
        delta_dot = rng.uniform(-0.02, 0.02)
        x_pos = 0.0

        fell = False
        for step in range(max_steps):
            # LQR control: state = [phi, phi_dot, delta, delta_dot]
            state_lqr = np.array([theta, theta_dot, delta, delta_dot])
            tau = -K_lqr @ state_lqr
            tau = np.clip(tau, -10.0, 10.0)

            # RK4 with Meijaard dynamics
            # Meijaard state: [phi, delta, phi_dot, delta_dot]
            # LQR state: [phi, phi_dot, delta, delta_dot]
            for _ in range(sub_steps):
                def derivs(s):
                    th, thd, dl, dld = s  # LQR order
                    A_m, B_m = ab_matrix(M, C1, K0, K2, max(v0, 0.5), g)
                    B_steer = B_m[:, 1:2]
                    # Convert to Meijaard order: [phi, delta, phi_dot, delta_dot]
                    x = np.array([[th], [dl], [thd], [dld]])
                    x_dot = A_m @ x + B_steer * tau
                    # Convert back to LQR order: [phi, phi_dot, delta, delta_dot]
                    return np.array([x_dot[0, 0], x_dot[2, 0], x_dot[1, 0], x_dot[3, 0]])

                s = np.array([theta, theta_dot, delta, delta_dot])
                k1 = derivs(s)
                k2 = derivs(s + 0.5 * dt_sub * k1)
                k3 = derivs(s + 0.5 * dt_sub * k2)
                k4 = derivs(s + dt_sub * k3)
                s = s + (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                theta, theta_dot, delta, delta_dot = s

            x_pos += v0 * dt

            if abs(theta) > math.pi / 4:
                fell = True
                break

        results.append({
            'seed': seed,
            'distance': x_pos,
            'steps': step + 1,
            'time': (step + 1) * dt,
            'fell': fell,
            'final_theta': theta,
        })

    return results


def main():
    print("=" * 60)
    print("SINDy LQR on True Meijaard Dynamics")
    print("=" * 60)

    # Load SINDy LQR
    data = np.load('D:/系统辨识作业/sindy_bicycle/sindy_lqr.npz')
    K_sindy = data['K_lqr']
    print(f"SINDy LQR gains: {K_sindy}")
    print(f"State order: [phi, phi_dot, delta, delta_dot]")

    # Also compute true Meijaard LQR for comparison
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
    print(f"True LQR gains:   {K_true}")

    # Test at different speeds
    for v0 in [4.5, 5.0, 5.5, 6.0, 7.0]:
        print(f"\n--- v = {v0} m/s ---")

        # SINDy LQR
        results_sindy = run_meijaard_lqr(K_sindy, v0=v0, theta0=0.02,
                                          num_seeds=50, max_steps=3000)
        dist_s = [r['distance'] for r in results_sindy]
        fell_s = sum(1 for r in results_sindy if r['fell']) / len(results_sindy)

        # True LQR
        results_true = run_meijaard_lqr(K_true, v0=v0, theta0=0.02,
                                         num_seeds=50, max_steps=3000)
        dist_t = [r['distance'] for r in results_true]
        fell_t = sum(1 for r in results_true if r['fell']) / len(results_true)

        print(f"  SINDy LQR: dist={np.mean(dist_s):.1f}±{np.std(dist_s):.1f}m, fell={fell_s:.0%}")
        print(f"  True  LQR: dist={np.mean(dist_t):.1f}±{np.std(dist_t):.1f}m, fell={fell_t:.0%}")

    # Open-loop comparison
    print(f"\n--- Open-loop (no control) at v=5.5 ---")
    results_open = run_meijaard_lqr(np.zeros(4), v0=5.5, theta0=0.02,
                                     num_seeds=50, max_steps=3000)
    dist_o = [r['distance'] for r in results_open]
    fell_o = sum(1 for r in results_open if r['fell']) / len(results_open)
    print(f"  Open-loop: dist={np.mean(dist_o):.1f}±{np.std(dist_o):.1f}m, fell={fell_o:.0%}")


if __name__ == '__main__':
    main()
