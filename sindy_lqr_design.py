"""Design LQR for SINDy-identified bicycle model.

Extracts A_d, B_d matrices from SINDy coefficients and designs
discrete-time LQR controller.

SINDy state: [phi, delta, phi_dot, delta_dot]
LQR state:   [phi, phi_dot, delta, delta_dot] (reordered)
"""

import numpy as np
from scipy.linalg import solve_discrete_are


STATE_NAMES_SINDY = ['phi', 'delta', 'phi_dot', 'delta_dot']
STATE_NAMES_LQR = ['phi', 'phi_dot', 'delta', 'delta_dot']
REORDER_TO_LQR = [0, 2, 1, 3]  # sindy -> lqr
REORDER_TO_SINDY = [0, 2, 1, 3]  # lqr -> sindy (same permutation, it's symmetric)


def extract_sindy_matrices(data_path, threshold=0.01):
    """Extract discrete-time A_d, B_d from SINDy coefficients.

    SINDy model: s_{t+1} = s_t + Xi^T @ theta(s_t, tau)
    For linear terms: s_{t+1} = (I + A_lin) @ s_t + B_lin * tau
    So: A_d = I + A_lin, B_d = B_lin
    """
    data = np.load(data_path, allow_pickle=True)
    Xi = data['coefficients']  # (n_library, n_states)
    labels = list(data['feature_names'])
    dt = float(data['dt'])
    v = float(data['v'])

    print(f"Loaded SINDy model from {data_path}")
    print(f"  Speed: {v} m/s, dt: {dt} s")
    print(f"  Labels: {labels}")

    # Feature order: [1, phi, delta, phi_dot, delta_dot, tau, phi^2, ...]
    # Linear indices: 0=const, 1=phi, 2=delta, 3=phi_dot, 4=delta_dot, 5=tau
    n_states = 4

    # Extract linear A coefficients (rows 1-4 of Xi, columns for states)
    A_lin = Xi[1:5, :].T  # (4, 4) - coefficient of [phi, delta, phi_dot, delta_dot]
    B_lin = Xi[5:6, :].T  # (4, 1) - coefficient of tau

    # Discrete-time matrices
    A_d = np.eye(n_states) + A_lin
    B_d = B_lin

    print(f"\n--- SINDy Discrete-Time Matrices (state: [phi, delta, phidot, deltadot]) ---")
    print(f"A_d:")
    for i, row in enumerate(A_d):
        print(f"  [{', '.join(f'{x:+10.6f}' for x in row)}]  # {STATE_NAMES_SINDY[i]}")
    print(f"B_d:")
    for i, row in enumerate(B_d):
        print(f"  [{row[0]:+10.6f}]  # {STATE_NAMES_SINDY[i]}")

    # Check eigenvalues
    eigvals = np.linalg.eigvals(A_d)
    print(f"\nA_d eigenvalues: {eigvals}")
    print(f"Stable (all |lambda| < 1)? {all(abs(e) < 1 for e in eigvals)}")

    # Compare with true Meijaard linear model
    from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix
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
    A_true, B_true = ab_matrix(M, C1, K0, K2, v, 9.81)
    B_steer_true = B_true[:, 1:2]

    # True discrete-time (forward Euler): A_d_true = I + dt*A, B_d_true = dt*B
    A_d_true = np.eye(4) + dt * A_true
    B_d_true = dt * B_steer_true

    print(f"\n--- True Meijaard Discrete-Time (Euler) ---")
    print(f"A_d_true:")
    for i, row in enumerate(A_d_true):
        print(f"  [{', '.join(f'{x:+10.6f}' for x in row)}]  # {STATE_NAMES_SINDY[i]}")
    print(f"B_d_true:")
    for i, row in enumerate(B_d_true):
        print(f"  [{row[0]:+10.6f}]  # {STATE_NAMES_SINDY[i]}")

    # Compare
    print(f"\n--- SINDy vs True Comparison ---")
    print(f"A_d error (max abs): {np.max(np.abs(A_d - A_d_true)):.6f}")
    print(f"B_d error (max abs): {np.max(np.abs(B_d - B_d_true)):.6f}")

    return A_d, B_d, dt, v


def design_lqr_discrete(A_d, B_d, Q_diag=None, R_val=0.2):
    """Design discrete-time LQR.

    Args:
        A_d: discrete-time system matrix (4x4)
        B_d: discrete-time input matrix (4x1)
        Q_diag: state cost diagonal (in SINDy state order)
        R_val: input cost

    Returns:
        K: LQR gain matrix (in LQR state order: [phi, phidot, delta, deltadot])
    """
    if Q_diag is None:
        Q_diag = [1000.0, 100.0, 10.0, 1.0]

    # Reorder to LQR state convention: [phi, phi_dot, delta, delta_dot]
    reorder = REORDER_TO_LQR
    A_lqr = A_d[np.ix_(reorder, reorder)]
    B_lqr = B_d[reorder]

    Q = np.diag(Q_diag)
    R = np.array([[R_val]])

    print(f"\n--- Discrete-Time LQR Design ---")
    print(f"State order: {STATE_NAMES_LQR}")
    print(f"Q = diag({Q_diag})")
    print(f"R = {R_val}")

    # Solve discrete-time ARE
    P = solve_discrete_are(A_lqr, B_lqr, Q, R)
    K = np.linalg.inv(B_lqr.T @ P @ B_lqr + R) @ B_lqr.T @ P @ A_lqr

    print(f"\nP (solution to DARE):")
    for row in P:
        print(f"  [{', '.join(f'{x:+10.4f}' for x in row)}]")
    print(f"\nK = {K.flatten()}")
    print(f"  phi: {K[0,0]:+.4f}")
    print(f"  phi_dot: {K[0,1]:+.4f}")
    print(f"  delta: {K[0,2]:+.4f}")
    print(f"  delta_dot: {K[0,3]:+.4f}")

    # Closed-loop eigenvalues
    A_cl = A_lqr - B_lqr @ K
    eigvals_cl = np.linalg.eigvals(A_cl)
    print(f"\nClosed-loop eigenvalues: {eigvals_cl}")
    print(f"All |lambda| < 1? {all(abs(e) < 1 for e in eigvals_cl)}")

    # Compare with continuous LQR from true model
    from meijaard_dynamics import benchmark_par_to_canonical, ab_matrix
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
    v = 5.5
    A_true, B_true = ab_matrix(M, C1, K0, K2, v, 9.81)
    B_steer_true = B_true[:, 1:2]

    # Reorder true model to LQR convention
    reorder = REORDER_TO_LQR
    A_true_lqr = A_true[np.ix_(reorder, reorder)]
    B_true_lqr = B_steer_true[reorder]

    P_true = solve_continuous_are(A_true_lqr, B_true_lqr, Q, R)
    K_true = (np.linalg.inv(R) @ B_true_lqr.T @ P_true).flatten()

    print(f"\n--- Comparison with Continuous LQR (true Meijaard) ---")
    print(f"K_true = {K_true}")
    print(f"K_sindy = {K.flatten()}")
    print(f"K error (max abs): {np.max(np.abs(K.flatten() - K_true)):.4f}")

    return K.flatten()


def simulate_sindy_lqr(A_d, B_d, K, dt, v, num_steps=3000):
    """Simulate SINDy model with LQR control.

    State: [phi, delta, phi_dot, delta_dot] (SINDy order)
    K is in LQR order: [phi, phi_dot, delta, delta_dot]
    """
    reorder_from_lqr = [0, 2, 1, 3]  # LQR [phi,phidot,delta,deltadot] -> SINDy [phi,delta,phidot,deltadot]

    # Initial state (small perturbation)
    s = np.array([0.02, 0.0, 0.0, 0.0])  # [phi, delta, phi_dot, delta_dot]

    phi_log = []
    delta_log = []
    tau_log = []

    for step in range(num_steps):
        # Reorder state for LQR: [phi, delta, phidot, deltadot] -> [phi, phidot, delta, deltadot]
        s_lqr = s[reorder_from_lqr]

        # LQR control
        tau = -K @ s_lqr

        # Clamp torque
        tau = np.clip(tau, -10.0, 10.0)

        # Propagate SINDy model
        s = A_d @ s + B_d.flatten() * tau

        phi_log.append(s[0])
        delta_log.append(s[1])
        tau_log.append(tau)

        # Check fall
        if abs(s[0]) > np.pi / 4:
            print(f"  Fell at step {step+1}")
            break

    phi_log = np.array(phi_log)
    delta_log = np.array(delta_log)
    tau_log = np.array(tau_log)

    print(f"\n--- SINDy + LQR Simulation ({len(phi_log)} steps, v={v} m/s) ---")
    print(f"  phi range: [{phi_log.min():.6f}, {phi_log.max():.6f}] rad")
    print(f"  delta range: [{delta_log.min():.6f}, {delta_log.max():.6f}] rad")
    print(f"  tau range: [{tau_log.min():.6f}, {tau_log.max():.6f}] Nm")
    print(f"  Final |phi|: {abs(phi_log[-1]):.6f} rad")

    return phi_log, delta_log, tau_log


def main():
    print("=" * 60)
    print("LQR Design for SINDy-Identified Bicycle Model")
    print("=" * 60)

    # Extract SINDy matrices at v=5.5
    A_d, B_d, dt, v = extract_sindy_matrices(
        'D:/系统辨识作业/sindy_bicycle/meijaard_sindy.npz',
        threshold=0.01)

    # Design LQR with different Q values
    Q_configs = [
        ([1000.0, 100.0, 10.0, 1.0], 0.2, "default"),
        ([10000.0, 100.0, 100.0, 1.0], 0.2, "high-phi"),
        ([1000.0, 10.0, 100.0, 1.0], 0.1, "high-delta-low-R"),
        ([5000.0, 50.0, 50.0, 5.0], 0.5, "balanced"),
    ]

    best_K = None
    best_name = None
    best_score = -1

    for Q_diag, R_val, name in Q_configs:
        print(f"\n{'='*60}")
        print(f"Config: {name} (Q={Q_diag}, R={R_val})")
        print(f"{'='*60}")

        K = design_lqr_discrete(A_d, B_d, Q_diag=Q_diag, R_val=R_val)

        # Simulate
        phi_log, delta_log, tau_log = simulate_sindy_lqr(A_d, B_d, K, dt, v)

        # Score: how long it survives + how small phi stays
        n_steps = len(phi_log)
        max_phi = np.max(np.abs(phi_log))
        score = n_steps / (1 + max_phi * 100)

        if score > best_score:
            best_score = score
            best_K = K
            best_name = name

    print(f"\n{'='*60}")
    print(f"Best config: {best_name}")
    print(f"K_lqr = [{', '.join(f'{k:.6f}' for k in best_K)}]")
    print(f"{'='*60}")

    # Save
    out_path = 'D:/系统辨识作业/sindy_bicycle/sindy_lqr.npz'
    np.savez(out_path, K_lqr=best_K, A_d=A_d, B_d=B_d,
             state_names_lqr=STATE_NAMES_LQR, v=v, dt=dt)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
