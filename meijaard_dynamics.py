"""Compute Meijaard 2007 benchmark bicycle A matrix using exact parameters.

Uses the benchmark_par_to_canonical and ab_matrix functions from
bicycleparameters library, reimplemented here with exact parameter values.
"""

import numpy as np


def benchmark_par_to_canonical(p):
    """Compute M, C1, K0, K2 from Meijaard 2007 benchmark parameters.

    Exact copy from bicycleparameters library.
    State: [roll angle (phi), steer angle (delta)]
    """
    mT = p['mR'] + p['mB'] + p['mH'] + p['mF']
    xT = (p['xB']*p['mB'] + p['xH']*p['mH'] + p['w']*p['mF']) / mT
    zT = (-p['rR']*p['mR'] + p['zB']*p['mB'] +
          p['zH']*p['mH'] - p['rF']*p['mF']) / mT

    ITxx = (p['IRxx'] + p['IBxx'] + p['IHxx'] + p['IFxx'] + p['mR'] *
            p['rR']**2 + p['mB']*p['zB']**2 + p['mH']*p['zH']**2 +
            p['mF']*p['rF']**2)
    ITxz = (p['IBxz'] + p['IHxz'] - p['mB']*p['xB']*p['zB'] -
            p['mH']*p['xH']*p['zH'] + p['mF']*p['w']*p['rF'])
    IRzz = p['IRxx']
    IFzz = p['IFxx']
    ITzz = (IRzz + p['IBzz'] + p['IHzz'] + IFzz +
            p['mB']*p['xB']**2 + p['mH']*p['xH']**2 + p['mF']*p['w']**2)

    mA = p['mH'] + p['mF']
    xA = (p['xH']*p['mH'] + p['w']*p['mF']) / mA
    zA = (p['zH']*p['mH'] - p['rF']*p['mF']) / mA

    IAxx = (p['IHxx'] + p['IFxx'] + p['mH']*(p['zH'] - zA)**2 +
            p['mF']*(p['rF'] + zA)**2)
    IAxz = (p['IHxz'] - p['mH']*(p['xH'] - xA)*(p['zH'] - zA) + p['mF'] *
            (p['w'] - xA)*(p['rF'] + zA))
    IAzz = (p['IHzz'] + IFzz + p['mH']*(p['xH'] - xA)**2 + p['mF'] *
            (p['w'] - xA)**2)
    uA = (xA - p['w'] - p['c'])*np.cos(p['lam']) - zA*np.sin(p['lam'])
    IAll = (mA*uA**2 + IAxx*np.sin(p['lam'])**2 +
            2*IAxz*np.sin(p['lam'])*np.cos(p['lam']) +
            IAzz*np.cos(p['lam'])**2)
    IAlx = (-mA*uA*zA + IAxx*np.sin(p['lam']) + IAxz*np.cos(p['lam']))
    IAlz = (mA*uA*xA + IAxz*np.sin(p['lam']) + IAzz*np.cos(p['lam']))

    mu = p['c'] / p['w'] * np.cos(p['lam'])

    SR = p['IRyy'] / p['rR']
    SF = p['IFyy'] / p['rF']
    ST = SR + SF
    SA = mA*uA + mu*mT*xT

    Mpp = ITxx
    Mpd = IAlx + mu*ITxz
    Mdp = Mpd
    Mdd = IAll + 2*mu*IAlz + mu**2*ITzz
    M = np.array([[Mpp, Mpd], [Mdp, Mdd]])

    K0pp = mT*zT
    K0pd = -SA
    K0dp = K0pd
    K0dd = -SA*np.sin(p['lam'])
    K0 = np.array([[K0pp, K0pd], [K0dp, K0dd]])

    K2pp = 0.
    K2pd = (ST - mT*zT) / p['w'] * np.cos(p['lam'])
    K2dp = 0.
    K2dd = (SA + SF*np.sin(p['lam'])) / p['w'] * np.cos(p['lam'])
    K2 = np.array([[K2pp, K2pd], [K2dp, K2dd]])

    C1pp = 0.
    C1pd = (mu*ST + SF*np.cos(p['lam']) + ITxz / p['w'] *
            np.cos(p['lam']) - mu*mT*zT)
    C1dp = -(mu*ST + SF*np.cos(p['lam']))
    C1dd = (IAlz / p['w']*np.cos(p['lam']) + mu*(SA +
            ITzz / p['w']*np.cos(p['lam'])))
    C1 = np.array([[C1pp, C1pd], [C1dp, C1dd]])

    return M, C1, K0, K2


def ab_matrix(M, C1, K0, K2, v, g):
    """Compute A and B matrices for linearized Whipple model.

    A = [[0, I], [-M^-1*(g*K0 + v^2*K2), -M^-1*v*C1]]
    B = [[0], [M^-1]]

    States: [roll angle, steer angle, roll rate, steer rate]
    Inputs: [roll torque, steer torque]
    """
    invM = (1. / (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) *
            np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]))

    a11 = np.zeros((2, 2))
    a12 = np.eye(2)
    a21 = -invM @ (g * K0 + v**2 * K2)
    a22 = -invM @ (v * C1)

    A = np.vstack((np.hstack((a11, a12)), np.hstack((a21, a22))))
    B = np.vstack((np.zeros((2, 2)), invM))

    return A, B


def main():
    # Meijaard 2007 benchmark parameters (browser + jason rider)
    p = {
        'IBxx': 11.3557360401,
        'IBxz': -1.96756380745,
        'IByy': 12.2177848012,
        'IBzz': 3.12354397008,
        'IFxx': 0.0904106601579,
        'IFyy': 0.149389340425,
        'IHxx': 0.253379594731,
        'IHxz': -0.0720452391817,
        'IHyy': 0.246138810935,
        'IHzz': 0.0955770796289,
        'IRxx': 0.0883819364527,
        'IRyy': 0.152467620286,
        'c': 0.0685808540382,
        'g': 9.81,
        'lam': 0.399680398707,
        'mB': 81.86,
        'mF': 2.02,
        'mH': 3.22,
        'mR': 3.11,
        'rF': 0.34352982332,
        'rR': 0.340958858855,
        'w': 1.121,
        'xB': 0.289099434117,
        'xH': 0.866949640247,
        'zB': -1.04029228321,
        'zH': -0.748236400835,
    }

    g = 9.81

    print("=" * 70)
    print("Meijaard 2007 Benchmark Bicycle - Exact Dynamics")
    print("=" * 70)

    total_mass = p['mR'] + p['mB'] + p['mH'] + p['mF']
    print(f"\nTotal mass: {total_mass:.2f} kg")
    print(f"  Rear frame (mB): {p['mB']:.2f} kg")
    print(f"  Rider + bike total: {total_mass:.2f} kg")
    print(f"Wheelbase (w): {p['w']:.3f} m")
    print(f"Trail (c): {p['c']:.4f} m")
    print(f"Head angle (lam): {np.degrees(p['lam']):.1f} deg")
    print(f"Front wheel radius: {p['rF']:.4f} m")
    print(f"Rear wheel radius: {p['rR']:.4f} m")

    # Compute canonical matrices
    M, C1, K0, K2 = benchmark_par_to_canonical(p)

    print(f"\n--- Canonical Matrices ---")
    print(f"\nM (mass matrix):")
    for row in M:
        print(f"  [{row[0]:12.6f}, {row[1]:12.6f}]")

    print(f"\nC1 (speed-proportional damping):")
    for row in C1:
        print(f"  [{row[0]:12.6f}, {row[1]:12.6f}]")

    print(f"\nK0 (gravity stiffness):")
    for row in K0:
        print(f"  [{row[0]:12.6f}, {row[1]:12.6f}]")

    print(f"\nK2 (centrifugal stiffness):")
    for row in K2:
        print(f"  [{row[0]:12.6f}, {row[1]:12.6f}]")

    # Verify A at v=5.0
    print(f"\n--- A matrix at v=5.0 m/s (verification) ---")
    A5, B5 = ab_matrix(M, C1, K0, K2, 5.0, g)
    for row in A5:
        print(f"  [{', '.join(f'{x:+12.8f}' for x in row)}]")

    print(f"\n--- Eigenvalue analysis ---")
    # Scan speeds
    print(f"\n{'v (m/s)':>8} | {'Max Re':>10} | {'Stable':>6} | Eigenvalues")
    print("-" * 80)

    stable_range = []
    for v in np.arange(1.0, 10.1, 0.5):
        A, B = ab_matrix(M, C1, K0, K2, v, g)
        eigvals = np.linalg.eigvals(A)
        max_real = np.max(eigvals.real)
        is_stable = max_real < 0
        if is_stable:
            stable_range.append(v)
        eig_str = ', '.join(f'{e:.4f}' for e in eigvals)
        print(f"  {v:6.1f}  | {max_real:+10.4f} | {'YES' if is_stable else 'NO':>6} | {eig_str}")

    if stable_range:
        print(f"\n*** Self-stable speed range: {min(stable_range):.1f} - {max(stable_range):.1f} m/s ***")

    # Fine-grained scan around stable region
    print(f"\n--- Fine scan (0.1 m/s steps) ---")
    stable_fine = []
    for v in np.arange(1.0, 10.01, 0.1):
        A, B = ab_matrix(M, C1, K0, K2, v, g)
        eigvals = np.linalg.eigvals(A)
        max_real = np.max(eigvals.real)
        if max_real < 0:
            stable_fine.append(v)

    if stable_fine:
        print(f"Self-stable: {min(stable_fine):.1f} - {max(stable_fine):.1f} m/s")

    # Show B matrix (steer torque input) at v=5
    print(f"\n--- B matrix at v=5.0 m/s ---")
    for row in B5:
        print(f"  [{row[0]:12.8f}, {row[1]:12.8f}]")

    # B for steer torque only (column 1)
    B_steer = B5[:, 1:2]
    print(f"\nB_steer (steer torque input):")
    for row in B_steer:
        print(f"  [{row[0]:12.8f}]")

    # LQR gains at v=5
    from scipy.linalg import solve_continuous_are
    A5, _ = ab_matrix(M, C1, K0, K2, 5.0, g)
    Q = np.diag([1000.0, 100.0, 10.0, 1.0])
    R = np.array([[0.2]])
    P = solve_continuous_are(A5, B_steer, Q, R)
    K = (np.linalg.inv(R) @ B_steer.T @ P).flatten()
    print(f"\nLQR gains at v=5.0 (Q=[1000,100,10,1], R=0.2):")
    print(f"  K = [{', '.join(f'{k:.4f}' for k in K)}]")

    # Now show what this means for path_tracking_env.py
    print(f"\n" + "=" * 70)
    print("Recommendations for path_tracking_env.py")
    print("=" * 70)
    print(f"""
    Current parameters (unrealistic):
      h = 0.8, wheelbase = 1.0, trail = 0.15
      g/h = 12.26, NO self-stability at any speed

    Meijaard benchmark (realistic):
      h (CoM height) ≈ {abs((-p['rR']*p['mR'] + p['zB']*p['mB'] + p['zH']*p['mH'] - p['rF']*p['mF']) / total_mass):.2f} m
      wheelbase = {p['w']:.3f} m
      trail = {p['c']:.4f} m
      head angle = {np.degrees(p['lam']):.1f} deg
      Self-stable at ~4-6 m/s (verified experimentally)

    The key difference: the Meijaard model includes ALL coupling terms:
      - M[0,1] ≠ 0 (mass coupling between roll and steer)
      - K0[0,1] ≠ 0 (gravity coupling: steer angle -> roll torque)
      - C1[0,1] ≠ 0 (gyroscopic coupling: steer rate -> roll torque)
      - C1[1,0] ≠ 0 (gyroscopic coupling: roll rate -> steer torque)
    """)


if __name__ == '__main__':
    main()
