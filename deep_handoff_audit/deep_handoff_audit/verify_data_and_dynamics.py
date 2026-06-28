"""Phase 1: Data inventory and dynamics verification.

Reads all .npz data files, computes statistics, and verifies 4D dynamics chain.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import csv

# ============================================================
# 1. DATA INVENTORY
# ============================================================
def inventory_all_data():
    """Read all .npz files and compute statistics."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npz_files = []
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.endswith('.npz'):
                npz_files.append(os.path.join(root, f))

    results = []
    for fpath in sorted(npz_files):
        rel = os.path.relpath(fpath, base)
        try:
            data = np.load(fpath, allow_pickle=True)
            for key in data.files:
                arr = data[key]
                if isinstance(arr, np.ndarray) and arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                    entry = {
                        'file': rel,
                        'key': key,
                        'shape': str(arr.shape),
                        'dtype': str(arr.dtype),
                        'n_rows': arr.shape[0],
                        'n_cols': arr.shape[1] if arr.ndim > 1 else 1,
                        'min': float(np.nanmin(arr)) if arr.size < 1e6 else 'too_large',
                        'max': float(np.nanmax(arr)) if arr.size < 1e6 else 'too_large',
                        'mean': float(np.nanmean(arr)) if arr.size < 1e6 else 'too_large',
                        'std': float(np.nanstd(arr)) if arr.size < 1e6 else 'too_large',
                        'nan_count': int(np.sum(np.isnan(arr))),
                        'inf_count': int(np.sum(np.isinf(arr))),
                    }
                    results.append(entry)
                    print(f"  {rel} [{key}]: shape={arr.shape}, dtype={arr.dtype}")
        except Exception as e:
            print(f"  ERROR reading {rel}: {e}")
    return results


# ============================================================
# 2. 4D DYNAMICS VERIFICATION
# ============================================================
def verify_4d_dynamics():
    """Verify the 4D Meijaard dynamics chain step by step."""
    from methods_common import (
        p, v0, dt, M, C1, K0, K2, invM,
        nonlinear_dynamics, nonlinear_step_rk4, real_step,
        generate_training_data, K_lqr
    )

    print("\n=== 4D DYNAMICS VERIFICATION ===\n")

    # 1. Verify parameters
    print(f"1. Meijaard parameters: {len(p)} params")
    print(f"   v0 = {v0} m/s")
    print(f"   dt = {dt} s (= {1/dt:.0f} Hz)")
    print(f"   M = \n{M}")
    print(f"   C1 = \n{C1}")
    print(f"   K0 = \n{K0}")
    print(f"   K2 = \n{K2}")

    # 2. Verify linearized dynamics at equilibrium
    from meijaard_dynamics import ab_matrix
    A, B = ab_matrix(M, C1, K0, K2, v0, 9.81)
    print(f"\n2. Linearized A matrix at v={v0}:")
    print(f"   A = \n{A}")
    print(f"   B = \n{B}")
    eigvals = np.linalg.eigvals(A)
    print(f"   Eigenvalues: {eigvals}")
    print(f"   Max real part: {np.max(eigvals.real):.6f}")
    print(f"   Stable: {np.max(eigvals.real) < 0}")

    # 3. Test nonlinear dynamics at equilibrium
    s_eq = np.array([0.0, 0.0, 0.0, 0.0])
    dsdt = nonlinear_dynamics(s_eq, 0.0)
    print(f"\n3. Nonlinear dynamics at equilibrium (s=0, tau=0):")
    print(f"   ds/dt = {dsdt}")
    print(f"   Should be [0, 0, 0, 0]: {np.allclose(dsdt, 0)}")

    # 4. Test single RK4 step
    s_test = np.array([0.1, 0.05, 0.1, 0.05])
    tau_test = 1.0
    dt_sub = dt / 5
    s_next_sub = nonlinear_step_rk4(s_test, tau_test, dt_sub)
    print(f"\n4. Single RK4 sub-step:")
    print(f"   s = {s_test}")
    print(f"   tau = {tau_test}")
    print(f"   dt_sub = {dt_sub:.6f} s")
    print(f"   s_next = {s_next_sub}")
    print(f"   delta = {s_next_sub - s_test}")

    # 5. Test real_step (5 sub-steps)
    s_next_full = real_step(s_test, tau_test)
    print(f"\n5. Full real_step (5 sub-steps):")
    print(f"   s_next = {s_next_full}")
    print(f"   delta = {s_next_full - s_test}")

    # 6. Verify linearity near equilibrium
    print(f"\n6. Linearity check near equilibrium:")
    for eps in [0.001, 0.01, 0.1, 0.5]:
        s1 = np.array([eps, 0, 0, 0])
        s2 = np.array([2*eps, 0, 0, 0])
        d1 = real_step(s1, 0.0) - s1
        d2 = real_step(s2, 0.0) - s2
        ratio = np.linalg.norm(d2) / np.linalg.norm(d1) if np.linalg.norm(d1) > 1e-15 else float('inf')
        print(f"   eps={eps:.3f}: ||delta(2eps)||/||delta(eps)|| = {ratio:.4f} (linear=2.0)")

    # 7. Test training data generation
    print(f"\n7. Training data generation:")
    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(100, seed=42)
    print(f"   states shape: {states.shape}, range: [{states.min():.3f}, {states.max():.3f}]")
    print(f"   actions shape: {actions.shape}, range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"   deltas shape: {deltas.shape}, range: [{deltas.min():.6f}, {deltas.max():.6f}]")
    print(f"   state_std: {state_std}")
    print(f"   action_std: {action_std:.4f}")
    print(f"   delta_std: {delta_std}")

    # 8. LQR controller
    print(f"\n8. LQR controller:")
    print(f"   K_lqr = {K_lqr}")
    print(f"   Q = diag([1000, 100, 10, 1])")
    print(f"   R = 0.2")

    return {
        'M': M.tolist(), 'C1': C1.tolist(), 'K0': K0.tolist(), 'K2': K2.tolist(),
        'v0': v0, 'dt': dt, 'A': A.tolist(), 'B': B.tolist(),
        'eigenvalues': eigvals.tolist(),
        'K_lqr': K_lqr.tolist(),
        'state_std': state_std.tolist(),
        'action_std': float(action_std),
        'delta_std': delta_std.tolist(),
    }


# ============================================================
# 3. NORMALIZATION CHAIN VERIFICATION
# ============================================================
def verify_normalization():
    """Verify complete normalization chain with numerical example."""
    from methods_common import generate_training_data, real_step

    print("\n=== NORMALIZATION CHAIN VERIFICATION ===\n")

    # Generate training data
    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000, seed=42)

    # Pick a sample point
    idx = 0
    s = states[idx]
    tau = actions[idx]
    delta_real = deltas[idx]

    print(f"Sample point idx={idx}:")
    print(f"  Raw state s = {s} [phi, delta, phi_dot, delta_dot]")
    print(f"  Raw action tau = {tau:.6f}")
    print(f"  Raw delta = {delta_real}")

    # Normalization (std-based, no mean subtraction)
    s_norm = s / state_std
    a_norm = tau / action_std
    delta_norm = delta_real / delta_std
    print(f"\n  s_norm = s / state_std = {s_norm}")
    print(f"  a_norm = tau / action_std = {a_norm:.6f}")
    print(f"  delta_norm = delta / delta_std = {delta_norm}")

    # Denormalization check
    s_next_from_norm = s + delta_norm * delta_std
    print(f"\n  s + delta_norm * delta_std = {s_next_from_norm}")
    print(f"  s + delta_real = {s + delta_real}")
    print(f"  Match: {np.allclose(s_next_from_norm, s + delta_real)}")

    # Real dynamics
    s_next_real = real_step(s, tau)
    print(f"\n  s_next_real (from real_step) = {s_next_real}")
    print(f"  delta_real = s_next_real - s = {s_next_real - s}")

    # Verify state_std computation
    print(f"\n  state_std (from code) = {state_std}")
    print(f"  state_std (manual) = {np.std(states, axis=0)}")
    print(f"  Match: {np.allclose(state_std, np.std(states, axis=0))}")

    # Verify delta_std computation
    print(f"\n  delta_std (from code) = {delta_std}")
    print(f"  delta_std (manual) = {np.std(deltas, axis=0)}")
    print(f"  Match: {np.allclose(delta_std, np.std(deltas, axis=0))}")

    # Check: is normalization just division by std (no mean subtraction)?
    print(f"\n  Normalization type: DIVISION BY STD ONLY (no mean subtraction)")
    print(f"  state_mean = {np.mean(states, axis=0)}")
    print(f"  If mean subtraction: s_norm = (s - mean) / std")
    print(f"  Actual: s_norm = s / std (mean NOT subtracted)")

    return {
        'state_std': state_std.tolist(),
        'action_std': float(action_std),
        'delta_std': delta_std.tolist(),
        'sample_state': s.tolist(),
        'sample_action': float(tau),
        'sample_delta_real': delta_real.tolist(),
        'delta_norm': delta_norm.tolist(),
        'normalization_type': 'std_only_no_mean_subtraction',
        'state_mean': np.mean(states, axis=0).tolist(),
    }


# ============================================================
# 4. 8D ROUTE CHECK
# ============================================================
def check_8d_route():
    """Check 8D data files and models."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("\n=== 8D ROUTE CHECK ===\n")

    # Check bicycle_data.npz
    bdata_path = os.path.join(base, 'bicycle_data.npz')
    if os.path.exists(bdata_path):
        data = np.load(bdata_path, allow_pickle=True)
        print(f"bicycle_data.npz keys: {data.files}")
        for key in data.files:
            arr = data[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            if arr.ndim == 2:
                print(f"    min={arr.min(axis=0)}")
                print(f"    max={arr.max(axis=0)}")
                print(f"    mean={arr.mean(axis=0)}")
    else:
        print("bicycle_data.npz NOT FOUND")

    # Check for 8D models
    sindy_path = os.path.join(base, 'sindy_model.npz')
    if os.path.exists(sindy_path):
        data = np.load(sindy_path, allow_pickle=True)
        print(f"\nsindy_model.npz keys: {data.files}")
        for key in data.files:
            arr = data[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
    else:
        print("\nsindy_model.npz NOT FOUND")

    # Check data_collector.py
    dc_path = os.path.join(base, 'data_collector.py')
    if os.path.exists(dc_path):
        print(f"\ndata_collector.py EXISTS")
    else:
        print(f"\ndata_collector.py NOT FOUND")

    # Check world_model.py
    wm_path = os.path.join(base, 'world_model.py')
    if os.path.exists(wm_path):
        print(f"world_model.py EXISTS")
    else:
        print(f"world_model.py NOT FOUND")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("DEEP HANDOFF AUDIT - Phase 1: Data & Dynamics Verification")
    print("=" * 70)

    # 1. Data inventory
    print("\n--- DATA INVENTORY ---")
    inv = inventory_all_data()

    # 2. 4D dynamics
    dyn = verify_4d_dynamics()

    # 3. Normalization
    norm = verify_normalization()

    # 4. 8D route
    check_8d_route()

    # Save results
    output = {
        'data_inventory': inv,
        'dynamics_verification': dyn,
        'normalization_verification': norm,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phase1_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
