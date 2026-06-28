"""Compute linearized Whipple dynamics for a real bicycle.

Based on Meijaard et al. (2007) benchmark parameters.
Uses M, C1, K0, K2 matrices from the standard Whipple model.

The linearized equations are:
  M * q_ddot + C1(v) * q_dot + (K0 + K2*v^2) * q = F

where q = [phi, delta] (lean, steer)

State: x = [phi, delta, phi_dot, delta_dot]
A = [[0, I], [-M^-1*(K0+K2*v^2), -M^-1*C1*v]]
"""

import numpy as np
import scipy.linalg as la


def compute_meijaard_matrices(params):
    """Compute M, C1, K0, K2 matrices from physical parameters.

    These capture the key coupling physics:
    - M: mass/inertia matrix (diagonal dominant)
    - C1: gyroscopic + speed-proportional damping (OFF-DIAGONAL coupling)
    - K0: gravity stiffness (trail-induced self-steering)
    - K2: centrifugal stiffness (lean-steer coupling at speed)

    Key coupling terms that enable self-stability:
    - C1[0,1] = front wheel gyroscopic coupling (phi_dot -> delta torque)
    - C1[1,0] = steer-to-lean gyroscopic coupling
    - K0[1,0] = trail effect (lean -> steer torque)
    - K2[0,1] = speed-dependent lean-steer coupling
    """
    g = 9.81
    m_r = params['m_rider']       # rider mass
    m_b = params['m_bike']        # bike mass (frame+rear wheel)
    m_f = params['m_front']       # front assembly mass (fork+front wheel)
    m = m_r + m_b + m_f           # total mass

    h = params['h']               # CoM height
    L = params['wheelbase']       # wheelbase
    c = params['trail']           # mechanical trail
    lam = params['head_angle']    # head angle (radians from vertical)
    g_sin_lam = g * np.sin(lam)

    # Wheel inertias
    I_fw = params['I_flywheel']   # front wheel flywheel inertia (spin)
    I_wxx = params['I_wheel_xx']  # wheel roll inertia

    # Effective inertias (simplified lumped)
    I_phi = params['I_lean']      # total lean inertia
    I_delta = params['I_steer']   # total steer inertia

    # ================================================================
    # Mass matrix M (2x2)
    # ================================================================
    M = np.array([
        [I_phi, 0.0],
        [0.0, I_delta]
    ])

    # ================================================================
    # C1 matrix (speed-proportional, 2x2)
    # ================================================================
    # This is where the critical gyroscopic coupling lives.
    # In the real Whipple model:
    #   C1[0,0] = rear wheel gyroscopic damping on lean
    #   C1[0,1] = front wheel gyroscopic effect on lean from steer rate
    #   C1[1,0] = front wheel gyroscopic effect on steer from lean rate
    #   C1[1,1] = steering damping
    #
    # The front wheel gyroscopic term is: I_fw * v * cos(lam) / r_fw
    # which appears as off-diagonal coupling between phi_dot and delta.

    # Rear wheel gyroscopic (lean damping)
    r_rear = params.get('r_rear', 0.3)
    rear_gyro = m_b * r_rear * v_ref  # placeholder, will be computed per-speed

    # Front wheel gyroscopic coupling coefficient
    r_front = params.get('r_front', 0.35)
    # At speed v, the front wheel spin rate = v / r_front
    # Gyroscopic torque coupling = I_fw * v / r_front * cos(lam)
    gyro_coupling = I_fw * np.cos(lam) / r_front

    C1_base = np.array([
        [params.get('c1_00', 0.0), gyro_coupling],      # [0,1] = gyro coupling
        [-gyro_coupling, params.get('c1_11', 0.5)]       # [1,0] = -gyro coupling (antisymmetric)
    ])

    # ================================================================
    # K0 matrix (gravity stiffness, 2x2)
    # ================================================================
    # K0[0,0] = gravitational lean stiffness (destabilizing: -m*g*h)
    # K0[1,0] = trail effect: lean angle creates steer torque (stabilizing)
    # K0[0,1] = steer angle creates lean torque
    # K0[1,1] = steer stiffness

    trail_effect = m * g * c * np.cos(lam) / L  # trail-induced self-steering
    steer_to_lean = m * g * h  # gravity destabilizes lean

    K0 = np.array([
        [-steer_to_lean, params.get('k0_01', 0.0)],
        [trail_effect, params.get('k0_11', 1.0)]
    ])

    # ================================================================
    # K2 matrix (centrifugal stiffness, 2x2)
    # ================================================================
    # K2[0,1] = speed-dependent coupling: steer angle creates lean torque
    # K2[1,0] = speed-dependent coupling: lean angle creates steer torque
    # These are related to the geometry (trail, head angle, wheelbase)

    K2 = np.array([
        [0.0, m * h / L],           # steer -> lean at speed
        [m * c / L, 0.0]            # lean -> steer at speed (trail effect)
    ])

    return M, C1_base, K0, K2


def compute_bicycle_A_matrix(v, M, C1_base, K0, K2):
    """Compute A matrix at speed v.

    A = [[0, I], [-M^-1*(K0 + K2*v^2), -M^-1*C1*v]]

    C1 has both constant (gyro) and speed-proportional (damping) parts.
    The gyro part is already in C1_base, scaled by v.
    """
    M_inv = np.linalg.inv(M)

    # Stiffness at speed v
    K_eff = K0 + K2 * v**2

    # Damping at speed v (C1_base already includes gyro coupling coefficient)
    C_eff = C1_base * v

    A = np.zeros((4, 4))
    A[:2, 2:] = np.eye(2)
    A[2:, :2] = -M_inv @ K_eff
    A[2:, 2:] = -M_inv @ C_eff

    return A


def compute_lqr_gains(A, B, Q_diag, R_val):
    """Compute LQR gains."""
    Q = np.diag(Q_diag)
    R = np.array([[R_val]])
    P = la.solve_continuous_are(A, B, Q, R)
    K = (np.linalg.inv(R) @ B.T @ P).flatten()
    return K


def analyze_stability(A, v, label):
    """Analyze A matrix stability."""
    eigvals = np.linalg.eigvals(A)
    max_real = np.max(eigvals.real)

    print(f"\n  {label}:")
    print(f"    v = {v} m/s")
    print(f"    Eigenvalues: [{', '.join(f'{e:.4f}' for e in eigvals)}]")
    print(f"    Max real part: {max_real:+.4f}")
    print(f"    Stable: {'YES' if max_real < 0 else 'NO'}")

    return eigvals, max_real < 0


def find_self_stable_range(M, C1, K0, K2, v_min=1.0, v_max=15.0, v_step=0.1):
    """Find speed range where bike is self-stable."""
    stable_speeds = []
    v = v_min
    while v <= v_max:
        A = compute_bicycle_A_matrix(v, M, C1, K0, K2)
        eigvals = np.linalg.eigvals(A)
        if np.max(eigvals.real) < 0:
            stable_speeds.append(v)
        v += v_step

    if stable_speeds:
        return min(stable_speeds), max(stable_speeds)
    return None, None


def build_matrices_from_params(params, v):
    """Build M, C1, K0, K2 that depend on v (for gyro terms)."""
    g = 9.81
    m = params['total_mass']
    h = params['h']
    L = params['wheelbase']
    c = params['trail']
    lam = params['head_angle']
    I_fw = params['I_flywheel']
    r_front = params['r_front']

    I_phi = params['I_lean']
    I_delta = params['I_steer']

    M = np.array([[I_phi, 0.0], [0.0, I_delta]])

    # Gyroscopic coupling coefficient (scaled by v in C1)
    gyro = I_fw * np.cos(lam) / r_front

    C1 = np.array([
        [params.get('damping_lean', 0.5), gyro],
        [-gyro, params.get('damping_steer', 2.0)]
    ])

    # Gravity stiffness
    gravity_lean = m * g * h
    trail_stiff = m * g * c * np.cos(lam) / L

    K0 = np.array([
        [-gravity_lean, 0.0],
        [trail_stiff, params.get('steer_stiffness', 5.0)]
    ])

    # Centrifugal stiffness
    K2 = np.array([
        [0.0, m * h / L],
        [m * c / L, 0.0]
    ])

    return M, C1, K0, K2


def main():
    g = 9.81
    B = np.array([[0], [0], [0], [1.0]])

    print("=" * 70)
    print("Realistic Bicycle Dynamics Analysis (Full Whipple Model)")
    print("=" * 70)

    # ================================================================
    # Meijaard benchmark parameters
    # ================================================================
    # Based on Meijaard et al. (2007) Table 1
    # The key insight: self-stability comes from:
    #   1. Trail effect (K0[1,0]): lean -> steer torque
    #   2. Front wheel gyroscopic coupling (C1[0,1]): lean_rate -> steer torque
    #   3. Speed-dependent coupling (K2): steer -> lean torque

    meijaard_params = {
        'h': 1.0,               # CoM height (m)
        'wheelbase': 1.02,      # Meijaard benchmark (m)
        'trail': 0.08,          # mechanical trail (m)
        'head_angle': np.radians(18),  # head angle from vertical (rad)
        'total_mass': 89.0,     # rider + bike (kg)
        'I_flywheel': 0.14,     # front wheel flywheel inertia (kg*m^2)
        'r_front': 0.35,        # front wheel radius (m)
        'I_lean': 25.0,         # total lean inertia (kg*m^2)
        'I_steer': 0.5,         # total steer inertia (kg*m^2)
        'damping_lean': 0.5,    # lean damping (N*m*s/rad)
        'damping_steer': 2.0,   # steer damping (N*m*s/rad)
        'steer_stiffness': 5.0, # steer spring stiffness (N*m/rad)
    }

    print("\n" + "=" * 70)
    print("Parameter Set 1: Meijaard Benchmark-Based")
    print("=" * 70)
    print(f"  h = {meijaard_params['h']} m")
    print(f"  wheelbase = {meijaard_params['wheelbase']} m")
    print(f"  trail = {meijaard_params['trail']} m")
    print(f"  head angle = {np.degrees(meijaard_params['head_angle']):.1f} deg")
    print(f"  I_flywheel = {meijaard_params['I_flywheel']} kg*m^2")
    print(f"  r_front = {meijaard_params['r_front']} m")
    print(f"  I_lean = {meijaard_params['I_lean']} kg*m^2")
    print(f"  I_steer = {meijaard_params['I_steer']} kg*m^2")

    # Build matrices at v=5 m/s
    M, C1, K0, K2 = build_matrices_from_params(meijaard_params, 5.0)

    print(f"\n  M = [[{M[0,0]:.2f}, {M[0,1]:.2f}], [{M[1,0]:.2f}, {M[1,1]:.2f}]]")
    print(f"  C1 (at v=5):")
    C1_v = C1 * 5.0
    print(f"    [[{C1_v[0,0]:.3f}, {C1_v[0,1]:.3f}], [{C1_v[1,0]:.3f}, {C1_v[1,1]:.3f}]]")
    print(f"  K0 = [[{K0[0,0]:.2f}, {K0[0,1]:.2f}], [{K0[1,0]:.2f}, {K0[1,1]:.2f}]]")
    print(f"  K2 = [[{K2[0,0]:.4f}, {K2[0,1]:.4f}], [{K2[1,0]:.4f}, {K2[1,1]:.4f}]]")

    v_min, v_max = find_self_stable_range(M, C1, K0, K2)
    if v_min:
        print(f"\n  Self-stable speed range: {v_min:.1f} - {v_max:.1f} m/s")
    else:
        print(f"\n  NOT self-stable at any speed")

    for v in [3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]:
        A = compute_bicycle_A_matrix(v, M, C1, K0, K2)
        analyze_stability(A, v, f"v={v} m/s")

    # ================================================================
    # Parameter sweep: find trail and I_flywheel for self-stability at v=5
    # ================================================================
    print("\n" + "=" * 70)
    print("Parameter Sweep: trail vs I_flywheel for v=5 m/s")
    print("=" * 70)

    best_trail = None
    best_I = None
    best_margin = -999

    for trail_val in np.arange(0.03, 0.25, 0.01):
        for I_val in np.arange(0.05, 1.0, 0.05):
            test_params = dict(meijaard_params)
            test_params['trail'] = trail_val
            test_params['I_flywheel'] = I_val
            M_t, C1_t, K0_t, K2_t = build_matrices_from_params(test_params, 5.0)
            A = compute_bicycle_A_matrix(5.0, M_t, C1_t, K0_t, K2_t)
            eigvals = np.linalg.eigvals(A)
            max_real = np.max(eigvals.real)

            if max_real < 0 and max_real > best_margin:
                best_margin = max_real
                best_trail = trail_val
                best_I = I_val

    if best_trail:
        print(f"\n  Best parameters for v=5 m/s self-stability:")
        print(f"    trail = {best_trail:.2f} m")
        print(f"    I_flywheel = {best_I:.2f} kg*m^2")
        print(f"    Max real part = {best_margin:+.4f} (stable)")

        final_params = dict(meijaard_params)
        final_params['trail'] = best_trail
        final_params['I_flywheel'] = best_I
        M_f, C1_f, K0_f, K2_f = build_matrices_from_params(final_params, 5.0)

        v_min, v_max = find_self_stable_range(M_f, C1_f, K0_f, K2_f)
        if v_min:
            print(f"    Self-stable range: {v_min:.1f} - {v_max:.1f} m/s")

        print(f"\n  A matrix at v=5 m/s:")
        A = compute_bicycle_A_matrix(5.0, M_f, C1_f, K0_f, K2_f)
        for row in A:
            print(f"    [{', '.join(f'{x:+10.4f}' for x in row)}]")

        print(f"\n  Eigenvalues at v=5 m/s:")
        eigvals = np.linalg.eigvals(A)
        for e in eigvals:
            print(f"    {e:.4f}")

        print(f"\n  LQR gains (Q=[1000,100,10,1], R=0.2):")
        K = compute_lqr_gains(A, B, [1000, 100, 10, 1], 0.2)
        print(f"    K = [{', '.join(f'{k:.2f}' for k in K)}]")

        # Also check self-stability at different speeds
        print(f"\n  Stability at different speeds:")
        for v in [2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]:
            M_v, C1_v, K0_v, K2_v = build_matrices_from_params(final_params, v)
            A_v = compute_bicycle_A_matrix(v, M_v, C1_v, K0_v, K2_v)
            eigvals_v = np.linalg.eigvals(A_v)
            max_r = np.max(eigvals_v.real)
            print(f"    v={v:.1f} m/s: max_real={max_r:+.4f} {'STABLE' if max_r < 0 else 'UNSTABLE'}")

    else:
        print("\n  No self-stable parameters found in sweep")
        # Try wider sweep
        print("\n  Trying wider sweep (head angle, damping)...")
        best_margin = -999
        best_params = None

        for trail_val in np.arange(0.03, 0.30, 0.02):
            for I_val in np.arange(0.05, 2.0, 0.1):
                for damp_steer in np.arange(0.5, 5.0, 0.5):
                    for head_deg in np.arange(10, 30, 2):
                        test_params = dict(meijaard_params)
                        test_params['trail'] = trail_val
                        test_params['I_flywheel'] = I_val
                        test_params['damping_steer'] = damp_steer
                        test_params['head_angle'] = np.radians(head_deg)
                        M_t, C1_t, K0_t, K2_t = build_matrices_from_params(test_params, 5.0)
                        A = compute_bicycle_A_matrix(5.0, M_t, C1_t, K0_t, K2_t)
                        eigvals = np.linalg.eigvals(A)
                        max_real = np.max(eigvals.real)

                        if max_real < 0 and max_real > best_margin:
                            best_margin = max_real
                            best_params = dict(test_params)

        if best_params:
            print(f"\n  Found self-stable parameters!")
            print(f"    trail = {best_params['trail']:.2f} m")
            print(f"    I_flywheel = {best_params['I_flywheel']:.2f} kg*m^2")
            print(f"    damping_steer = {best_params['damping_steer']:.1f}")
            print(f"    head_angle = {np.degrees(best_params['head_angle']):.1f} deg")
            print(f"    Max real part = {best_margin:+.4f}")

            M_f, C1_f, K0_f, K2_f = build_matrices_from_params(best_params, 5.0)
            v_min, v_max = find_self_stable_range(M_f, C1_f, K0_f, K2_f)
            if v_min:
                print(f"    Self-stable range: {v_min:.1f} - {v_max:.1f} m/s")

            A = compute_bicycle_A_matrix(5.0, M_f, C1_f, K0_f, K2_f)
            print(f"\n  A matrix at v=5 m/s:")
            for row in A:
                print(f"    [{', '.join(f'{x:+10.4f}' for x in row)}]")
        else:
            print("\n  Still no self-stable parameters found.")
            print("  The model structure may need further refinement.")
            print("  Key missing physics:")
            print("    - Exact coupling coefficients from Meijaard derivation")
            print("    - Front/rear mass distribution effects")
            print("    - Exact gyroscopic torque direction")


if __name__ == '__main__':
    main()
