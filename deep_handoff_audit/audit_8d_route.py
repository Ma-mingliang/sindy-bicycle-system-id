"""Phase 11: 8D route deep audit.

Deep audit of the 8D Simplified Whipple route.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json

def audit_8d_dynamics():
    """Audit 8D dynamics model."""
    print("=" * 70)
    print("8D ROUTE DEEP AUDIT")
    print("=" * 70)

    # Check bicycle_dynamics.py
    print("\n1. 8D DYNAMICS MODEL:")
    try:
        from bicycle_dynamics import BicycleParams, BicycleDynamics
        params = BicycleParams()
        print(f"   Wheelbase: {params.wheelbase} m")
        print(f"   Head angle: {params.head_angle} rad")
        print(f"   Trail: {params.trail} m")
        print(f"   State: [ey, epsi, v, theta, theta_dot, k, delta, delta_dot]")
    except Exception as e:
        print(f"   Error importing: {e}")

    # Check data_collector.py
    print("\n2. DATA COLLECTION:")
    try:
        from data_collector import NORMALIZATION
        print(f"   Normalization factors:")
        for key, val in NORMALIZATION.items():
            print(f"     {key}: {val}")
    except Exception as e:
        print(f"   Error importing: {e}")

    # Check SINDy model
    print("\n3. SINDy MODEL:")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sindy_path = os.path.join(base, 'sindy_model.npz')
    if os.path.exists(sindy_path):
        data = np.load(sindy_path, allow_pickle=True)
        print(f"   Coefficients shape: {data['coefficients'].shape}")
        print(f"   Threshold: {data['threshold']}")
        print(f"   Feature names: {data['feature_names']}")
        print(f"   State names: {data['state_names']}")

        # Analyze sparsity
        Xi = data['coefficients']
        n_nonzero = np.sum(np.abs(Xi) > 1e-10)
        n_total = Xi.size
        print(f"   Sparsity: {n_nonzero}/{n_total} non-zero ({n_nonzero/n_total*100:.1f}%)")
    else:
        print("   sindy_model.npz NOT FOUND")

    # Check world_model.py
    print("\n4. WORLD MODEL:")
    try:
        from world_model import SINDyWorldModel
        model = SINDyWorldModel()
        print(f"   Loaded successfully")
        print(f"   Xi shape: {model.Xi.shape}")
    except Exception as e:
        print(f"   Error: {e}")

    return {'status': 'audited'}


def audit_8d_data():
    """Audit 8D data files."""
    print("\n5. 8D DATA FILES:")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_files = [
        'bicycle_data.npz',
        'bicycle_data_improved.npz',
    ]

    for fname in data_files:
        fpath = os.path.join(base, fname)
        if os.path.exists(fpath):
            data = np.load(fpath, allow_pickle=True)
            print(f"\n   {fname}:")
            for key in data.files:
                arr = data[key]
                if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                    print(f"     {key}: shape={arr.shape}")
                    if arr.ndim == 2 and arr.shape[1] <= 10:
                        print(f"       min={arr.min(axis=0)}")
                        print(f"       max={arr.max(axis=0)}")
        else:
            print(f"\n   {fname}: NOT FOUND")

    return {'status': 'audited'}


def audit_8d_sindy_model():
    """Audit SINDy model for 8D route."""
    print("\n6. SINDy MODEL ANALYSIS:")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check all SINDy models
    sindy_files = [
        'sindy_model.npz',
        'sindy_model_improved.npz',
        'sindy_full_model.npz',
        'meijaard_sindy.npz',
        'meijaard_sindy_v35.npz',
        'meijaard_sindy_v5.npz',
    ]

    for fname in sindy_files:
        fpath = os.path.join(base, fname)
        if os.path.exists(fpath):
            data = np.load(fpath, allow_pickle=True)
            print(f"\n   {fname}:")
            for key in data.files:
                arr = data[key]
                if isinstance(arr, np.ndarray):
                    print(f"     {key}: shape={arr.shape}, dtype={arr.dtype}")

    return {'status': 'audited'}


if __name__ == '__main__':
    results = {
        'dynamics': audit_8d_dynamics(),
        'data': audit_8d_data(),
        'sindy': audit_8d_sindy_model(),
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_8d_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
