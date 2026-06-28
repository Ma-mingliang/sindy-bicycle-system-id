"""HRRL Progressive Disturbance Verification.

Tests LQR suitability by progressively increasing disturbance magnitude.
A suitable LQR should:
  - Handle small-moderate disturbances well (stable baseline)
  - Show degraded but acceptable performance under large disturbances
  - Leave room for RL residual to improve performance

This verifies the LQR is neither too weak (fails immediately) nor too strong
(no room for RL to help).
"""

import numpy as np
import math
from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix


def run_lqr_with_disturbance(K_lqr, v0=5.5, disturbance_amp=0.0,
                             disturbance_type='wind', num_seeds=20,
                             max_steps=1500, dt=1/30):
    """Run Meijaard bicycle with LQR and progressive disturbance.

    Args:
        K_lqr: LQR gains [phi, phi_dot, delta, delta_dot]
        v0: forward speed
        disturbance_amp: disturbance amplitude (Nm)
        disturbance_type: 'wind' (sinusoidal) or 'impulse'
        num_seeds: number of random seeds
        max_steps: max steps per episode
        dt: time step

    Returns: dict with results
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
        theta = 0.02 + rng.uniform(-0.01, 0.01)
        theta_dot = rng.uniform(-0.02, 0.02)
        delta = rng.uniform(-0.01, 0.01)
        delta_dot = rng.uniform(-0.02, 0.02)

        # Disturbance state
        wind_state = 0.0
        impulse_force = 0.0
        impulse_until = 0

        max_phi = 0.0
        max_tau = 0.0
        fell = False

        for step in range(max_steps):
            # Generate disturbance
            if disturbance_type == 'wind' and disturbance_amp > 0:
                freq = 0.5 + 0.2 * seed
                dist = disturbance_amp * math.sin(2 * math.pi * freq * step * dt)
                wind_state += rng.randn() * 0.3 * dt
                wind_state *= 0.99
                dist += wind_state
            elif disturbance_type == 'impulse' and disturbance_amp > 0:
                if step < impulse_until:
                    dist = impulse_force
                elif rng.random() < 0.02:
                    impulse_force = rng.uniform(-disturbance_amp, disturbance_amp)
                    impulse_until = step + rng.randint(3, 10)
                    dist = impulse_force
                else:
                    dist = 0.0
            else:
                dist = 0.0

            # LQR control
            state_lqr = np.array([theta, theta_dot, delta, delta_dot])
            tau = -K_lqr @ state_lqr
            tau = np.clip(tau, -10.0, 10.0)

            max_phi = max(max_phi, abs(theta))
            max_tau = max(max_tau, abs(tau))

            # RK4 with Meijaard dynamics + disturbance
            for _ in range(sub_steps):
                A_m, B_m = ab_matrix(M, C1, K0, K2, max(v0, 0.5), g)
                B_s = B_m[:, 1:2]
                x = np.array([[theta], [delta], [theta_dot], [delta_dot]])
                xd = A_m @ x + B_s * tau
                # Add disturbance to roll acceleration
                xd[2, 0] += dist
                theta += dt_sub * xd[0, 0]
                delta += dt_sub * xd[1, 0]
                theta_dot += dt_sub * xd[2, 0]
                delta_dot += dt_sub * xd[3, 0]

            if abs(theta) > math.pi / 4:
                fell = True
                break

        results.append({
            'seed': seed,
            'steps': step + 1,
            'time': (step + 1) * dt,
            'fell': fell,
            'max_phi': max_phi,
            'max_tau': max_tau,
        })

    return results


def main():
    print("=" * 70)
    print("HRRL Progressive Disturbance Verification")
    print("=" * 70)

    # Load SINDy LQR
    data = np.load('D:/系统辨识作业/sindy_bicycle/sindy_lqr.npz')
    K_sindy = data['K_lqr']
    print(f"SINDy LQR gains: {K_sindy}")

    # Also compute true LQR for comparison
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

    v0 = 5.5
    num_seeds = 20
    max_steps = 1500

    # === Wind disturbance ===
    print(f"\n{'='*70}")
    print(f"Wind Disturbance Test (v={v0} m/s, {num_seeds} seeds, {max_steps} steps)")
    print(f"{'='*70}")
    print(f"{'Amp(Nm)':>8} | {'SINDy fell%':>12} {'avg_steps':>10} {'max_phi':>10} | "
          f"{'True fell%':>11} {'avg_steps':>10} {'max_phi':>10}")

    wind_amps = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    for amp in wind_amps:
        r_sindy = run_lqr_with_disturbance(K_sindy, v0, amp, 'wind', num_seeds, max_steps)
        r_true = run_lqr_with_disturbance(K_true, v0, amp, 'wind', num_seeds, max_steps)

        fell_s = sum(1 for r in r_sindy if r['fell']) / len(r_sindy)
        steps_s = np.mean([r['steps'] for r in r_sindy])
        phi_s = np.mean([r['max_phi'] for r in r_sindy])

        fell_t = sum(1 for r in r_true if r['fell']) / len(r_true)
        steps_t = np.mean([r['steps'] for r in r_true])
        phi_t = np.mean([r['max_phi'] for r in r_true])

        print(f"{amp:8.1f} | {fell_s:11.0%} {steps_s:10.0f} {phi_s:10.4f} | "
              f"{fell_t:10.0%} {steps_t:10.0f} {phi_t:10.4f}")

    # === Impulse disturbance ===
    print(f"\n{'='*70}")
    print(f"Impulse Disturbance Test (v={v0} m/s, {num_seeds} seeds, {max_steps} steps)")
    print(f"{'='*70}")
    print(f"{'Amp(Nm)':>8} | {'SINDy fell%':>12} {'avg_steps':>10} {'max_phi':>10} | "
          f"{'True fell%':>11} {'avg_steps':>10} {'max_phi':>10}")

    impulse_amps = [0.0, 2.0, 5.0, 8.0, 12.0, 20.0, 30.0]
    for amp in impulse_amps:
        r_sindy = run_lqr_with_disturbance(K_sindy, v0, amp, 'impulse', num_seeds, max_steps)
        r_true = run_lqr_with_disturbance(K_true, v0, amp, 'impulse', num_seeds, max_steps)

        fell_s = sum(1 for r in r_sindy if r['fell']) / len(r_sindy)
        steps_s = np.mean([r['steps'] for r in r_sindy])
        phi_s = np.mean([r['max_phi'] for r in r_sindy])

        fell_t = sum(1 for r in r_true if r['fell']) / len(r_true)
        steps_t = np.mean([r['steps'] for r in r_true])
        phi_t = np.mean([r['max_phi'] for r in r_true])

        print(f"{amp:8.1f} | {fell_s:11.0%} {steps_s:10.0f} {phi_s:10.4f} | "
              f"{fell_t:10.0%} {steps_t:10.0f} {phi_t:10.4f}")

    # === Analysis ===
    print(f"\n{'='*70}")
    print("Analysis: LQR Suitability for HRRL")
    print(f"{'='*70}")
    print("""
A suitable LQR for HRRL should:
1. Handle small disturbances (0-3 Nm) with 0% fall rate → baseline stability ✓
2. Show graceful degradation under moderate disturbances (3-10 Nm)
3. Leave room for RL residual to improve under large disturbances (>10 Nm)
4. NOT be so aggressive that RL has nothing to contribute

If LQR handles everything → RL residual has no room to help (bad)
If LQR fails immediately → baseline is too weak (bad)
Ideal: LQR handles ~70-80% of scenarios, RL improves the rest
""")


if __name__ == '__main__':
    main()
