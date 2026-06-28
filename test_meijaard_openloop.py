"""Open-loop straight-line test for Meijaard 2007 benchmark bicycle.

No LQR, no Stanley, no RL — just pure bicycle dynamics with zero steer torque.
Tests how far the bike can go before falling on flat ground.
"""

import numpy as np
import math
from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix


def run_openloop_straight(v0=5.0, theta0_range=(-0.05, 0.05),
                          num_seeds=50, max_steps=3000, dt=1/30):
    """Run open-loop straight-line simulation.

    Args:
        v0: initial forward speed (m/s)
        theta0_range: initial lean angle range (rad)
        num_seeds: number of random seeds
        max_steps: max simulation steps
        dt: time step (s)

    Returns: dict with results
    """
    # Meijaard 2007 benchmark parameters (browser + jason rider)
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
    wheelbase = p['w']

    results = []
    sub_steps = 5
    dt_sub = dt / sub_steps

    for seed in range(num_seeds):
        rng = np.random.RandomState(seed)

        # Initial state
        x_pos = 0.0
        y_pos = 0.0
        heading = 0.0
        v = v0 + rng.uniform(-0.3, 0.3)
        theta = rng.uniform(*theta0_range)
        theta_dot = 0.0
        delta = 0.0
        delta_dot = 0.0

        fell = False
        for step in range(max_steps):
            # Zero steer torque (open-loop)
            u = 0.0

            # RK4 integration with sub-stepping
            for _ in range(sub_steps):
                def derivs(s):
                    x_p, y_p, h_p, v_p, th_p, thd_p, dl_p, dld_p = s

                    # Meijaard dynamics: x_dot = A(v)*x + B*u
                    A_m, B_m = ab_matrix(M, C1, K0, K2, max(v_p, 0.5), g)
                    B_steer = B_m[:, 1:2]
                    x_state = np.array([[th_p], [dl_p], [thd_p], [dld_p]])
                    x_dot = A_m @ x_state + B_steer * u

                    return np.array([
                        v_p * math.cos(h_p),      # x_dot
                        v_p * math.sin(h_p),      # y_dot
                        -v_p * dl_p / wheelbase,   # heading_dot
                        0.0,                       # v_dot (constant speed)
                        x_dot[0, 0],              # theta_dot
                        x_dot[2, 0],              # theta_ddot
                        x_dot[1, 0],              # delta_dot
                        x_dot[3, 0],              # delta_ddot
                    ])

                s = np.array([x_pos, y_pos, heading, v,
                              theta, theta_dot, delta, delta_dot])
                k1 = derivs(s)
                k2 = derivs(s + 0.5 * dt_sub * k1)
                k3 = derivs(s + 0.5 * dt_sub * k2)
                k4 = derivs(s + dt_sub * k3)
                s = s + (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

                x_pos, y_pos, heading, v = s[0], s[1], s[2], s[3]
                theta, theta_dot = s[4], s[5]
                delta, delta_dot = s[6], s[7]

            # Check fall
            if abs(theta) > math.pi / 4:
                fell = True
                break

        distance = x_pos
        time_alive = (step + 1) * dt
        results.append({
            'seed': seed,
            'distance': distance,
            'steps': step + 1,
            'time': time_alive,
            'fell': fell,
            'final_theta': theta,
        })

    return results


def main():
    print("=" * 60)
    print("Open-Loop Straight-Line Test: Meijaard 2007 Bicycle")
    print("Zero steer torque, flat ground, no control")
    print("=" * 60)

    # Test at different speeds
    for v0 in [3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]:
        results = run_openloop_straight(v0=v0, num_seeds=50, max_steps=3000)

        distances = [r['distance'] for r in results]
        steps_list = [r['steps'] for r in results]
        fell_rate = sum(1 for r in results if r['fell']) / len(results)

        print(f"\nv={v0:.1f} m/s:")
        print(f"  Distance: {np.mean(distances):.1f} ± {np.std(distances):.1f} m "
              f"(min={min(distances):.1f}, max={max(distances):.1f})")
        print(f"  Steps:    {np.mean(steps_list):.0f} ± {np.std(steps_list):.0f} "
              f"(min={min(steps_list)}, max={max(steps_list)})")
        print(f"  Time:     {np.mean(steps_list)*1/30:.1f} ± {np.std(steps_list)*1/30:.1f} s")
        print(f"  Fell:     {fell_rate:.0%}")

    # Detailed trajectory for v=5.0, seed=0
    print("\n" + "=" * 60)
    print("Detailed trajectory: v=5.0 m/s, seed=0")
    print("=" * 60)

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

    x_pos, y_pos, heading = 0.0, 0.0, 0.0
    v = 5.0
    theta = 0.02  # small initial lean
    theta_dot, delta, delta_dot = 0.0, 0.0, 0.0
    dt = 1/30
    sub_steps = 5
    dt_sub = dt / sub_steps

    print(f"{'Step':>5} {'t(s)':>6} {'x(m)':>8} {'theta':>10} {'th_dot':>10} {'delta':>10}")
    for step in range(1500):
        u = 0.0
        for _ in range(sub_steps):
            def derivs(s):
                xp, yp, hp, vp, thp, thdp, dlp, dldp = s
                Am, Bm = ab_matrix(M, C1, K0, K2, max(vp, 0.5), g)
                Bs = Bm[:, 1:2]
                xs = np.array([[thp], [dlp], [thdp], [dldp]])
                xd = Am @ xs + Bs * u
                return np.array([
                    vp * math.cos(hp), vp * math.sin(hp),
                    -vp * dlp / 1.121, 0.0,
                    xd[0, 0], xd[2, 0], xd[1, 0], xd[3, 0],
                ])
            s = np.array([x_pos, y_pos, heading, v, theta, theta_dot, delta, delta_dot])
            k1 = derivs(s)
            k2 = derivs(s + 0.5*dt_sub*k1)
            k3 = derivs(s + 0.5*dt_sub*k2)
            k4 = derivs(s + dt_sub*k3)
            s = s + (dt_sub/6)*(k1 + 2*k2 + 2*k3 + k4)
            x_pos, y_pos, heading, v = s[0], s[1], s[2], s[3]
            theta, theta_dot = s[4], s[5]
            delta, delta_dot = s[6], s[7]

        if step < 5 or step % 100 == 0 or abs(theta) > math.pi/4:
            print(f"{step:5d} {step*dt:6.2f} {x_pos:8.2f} {theta:+10.6f} {theta_dot:+10.6f} {delta:+10.6f}")

        if abs(theta) > math.pi/4:
            print(f"  >>> FELL at step {step+1}, distance={x_pos:.2f} m")
            break


if __name__ == '__main__':
    main()
