"""Collect open-loop data from Meijaard 2007 benchmark bicycle.

Generates training data for SINDy by applying random steering torque
perturbations and recording state transitions.

Data format: [phi, delta, phi_dot, delta_dot, tau] -> [phi_next, delta_next, phi_dot_next, delta_dot_next]
"""

import numpy as np
import math
from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix


def get_meijaard_params():
    """Return Meijaard 2007 benchmark parameters."""
    return {
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


def simulate_episode(M, C1, K0, K2, v, g, dt, max_steps, torque_fn, rng):
    """Simulate one episode with given torque function.

    Returns: list of (state, tau, next_state) tuples
    state = [phi, delta, phi_dot, delta_dot]
    """
    sub_steps = 5
    dt_sub = dt / sub_steps

    # Initial state (small perturbation from upright)
    phi = rng.uniform(-0.03, 0.03)
    delta = rng.uniform(-0.02, 0.02)
    phi_dot = rng.uniform(-0.05, 0.05)
    delta_dot = rng.uniform(-0.05, 0.05)

    transitions = []
    fell = False

    for step in range(max_steps):
        state = np.array([phi, delta, phi_dot, delta_dot])
        tau = torque_fn(step, dt, rng)

        # RK4 integration
        for _ in range(sub_steps):
            def derivs(s):
                ph, dl, phd, dld = s
                A_m, B_m = ab_matrix(M, C1, K0, K2, max(v, 0.5), g)
                B_steer = B_m[:, 1:2]
                x = np.array([[ph], [dl], [phd], [dld]])
                x_dot = A_m @ x + B_steer * tau
                return np.array([x_dot[0, 0], x_dot[1, 0], x_dot[2, 0], x_dot[3, 0]])

            s = np.array([phi, delta, phi_dot, delta_dot])
            k1 = derivs(s)
            k2 = derivs(s + 0.5 * dt_sub * k1)
            k3 = derivs(s + 0.5 * dt_sub * k2)
            k4 = derivs(s + dt_sub * k3)
            s = s + (dt_sub / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            phi, delta, phi_dot, delta_dot = s

        next_state = np.array([phi, delta, phi_dot, delta_dot])

        # Check fall
        if abs(phi) > math.pi / 4:
            fell = True
            break

        transitions.append((state, tau, next_state))

    return transitions, fell


def torque_sinusoidal(step, dt, rng, amp=2.0, freq=None):
    """Sinusoidal torque with random frequency."""
    if freq is None:
        freq = rng.uniform(0.1, 2.0)
    return amp * math.sin(2 * math.pi * freq * step * dt)


def torque_impulse(step, dt, rng, prob=0.02, mag=3.0):
    """Random impulse torque."""
    if rng.random() < prob:
        return rng.uniform(-mag, mag)
    return 0.0


def torque_mixed(step, dt, rng):
    """Mixed: sinusoidal + occasional impulse."""
    t = torque_sinusoidal(step, dt, rng, amp=1.5)
    if rng.random() < 0.01:
        t += rng.uniform(-2.0, 2.0)
    return t


def torque_chirp(step, dt, rng, amp=2.0, f0=0.1, f1=3.0, T=100):
    """Chirp signal: frequency increases over time."""
    t = step * dt
    freq = f0 + (f1 - f0) * (t / T)
    return amp * math.sin(2 * math.pi * freq * t)


def collect_data(v=5.5, num_episodes=500, max_steps=300, dt=1/30):
    """Collect open-loop data from Meijaard bicycle.

    Args:
        v: forward speed (m/s) - use self-stable speed
        num_episodes: number of episodes to collect
        max_steps: max steps per episode
        dt: time step (s)

    Returns: dict with arrays
    """
    p = get_meijaard_params()
    g = 9.81
    M, C1, K0, K2 = benchmark_par_to_canonical(p)

    all_states = []
    all_taus = []
    all_next_states = []
    total_falls = 0

    torque_fns = [
        ('sinusoidal', torque_sinusoidal),
        ('impulse', torque_impulse),
        ('mixed', torque_mixed),
        ('chirp', torque_chirp),
        ('zero', lambda s, dt, rng: 0.0),
    ]

    for ep in range(num_episodes):
        rng = np.random.RandomState(ep)
        # Rotate through torque types
        fn_name, fn = torque_fns[ep % len(torque_fns)]

        transitions, fell = simulate_episode(
            M, C1, K0, K2, v, g, dt, max_steps, fn, rng)

        if fell:
            total_falls += 1

        for state, tau, next_state in transitions:
            all_states.append(state)
            all_taus.append(tau)
            all_next_states.append(next_state)

        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep+1}/{num_episodes}, "
                  f"transitions={len(all_states)}, falls={total_falls}")

    states = np.array(all_states, dtype=np.float64)
    taus = np.array(all_taus, dtype=np.float64).reshape(-1, 1)
    next_states = np.array(all_next_states, dtype=np.float64)

    print(f"\nTotal: {len(states)} transitions from {num_episodes} episodes")
    print(f"Falls: {total_falls} ({total_falls/num_episodes:.0%})")
    print(f"State ranges:")
    for i, name in enumerate(['phi', 'delta', 'phi_dot', 'delta_dot']):
        print(f"  {name}: [{states[:, i].min():.4f}, {states[:, i].max():.4f}]")
    print(f"  tau: [{taus.min():.4f}, {taus.max():.4f}]")

    return {
        'states': states,
        'taus': taus,
        'next_states': next_states,
        'v': v,
        'dt': dt,
        'total_episodes': num_episodes,
        'total_falls': total_falls,
    }


def main():
    print("=" * 60)
    print("Meijaard 2007 Open-Loop Data Collection")
    print("=" * 60)

    # Collect at self-stable speed v=5.5 m/s
    print("\n--- Collecting at v=5.5 m/s (self-stable) ---")
    data = collect_data(v=5.5, num_episodes=500, max_steps=300, dt=1/30)

    out_path = 'D:/系统辨识作业/sindy_bicycle/meijaard_openloop_data.npz'
    np.savez(out_path,
             states=data['states'],
             taus=data['taus'],
             next_states=data['next_states'],
             v=data['v'],
             dt=data['dt'],
             total_episodes=data['total_episodes'],
             total_falls=data['total_falls'])
    print(f"\nSaved to {out_path}")

    # Also collect at v=5.0 (boundary of self-stability)
    print("\n--- Collecting at v=5.0 m/s (self-stability boundary) ---")
    data5 = collect_data(v=5.0, num_episodes=500, max_steps=300, dt=1/30)

    out_path5 = 'D:/系统辨识作业/sindy_bicycle/meijaard_openloop_data_v5.npz'
    np.savez(out_path5,
             states=data5['states'],
             taus=data5['taus'],
             next_states=data5['next_states'],
             v=data5['v'],
             dt=data5['dt'],
             total_episodes=data5['total_episodes'],
             total_falls=data5['total_falls'])
    print(f"\nSaved to {out_path5}")


if __name__ == '__main__':
    main()
