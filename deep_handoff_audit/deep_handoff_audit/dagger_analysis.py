"""Phase 7: DAgger distribution analysis.

Analyzes the DAgger data collection process and distribution shift.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import csv

def analyze_dagger_process():
    """Analyze DAgger data collection and distribution shift."""
    print("=" * 70)
    print("DAGGER DISTRIBUTION ANALYSIS")
    print("=" * 70)

    from methods_common import generate_training_data, real_step
    from methods_evaluate import make_tau_func

    # Generate base training data
    states, actions, deltas, s_std, a_std, d_std = generate_training_data(30000, seed=42)

    print("\n1. BASE TRAINING DATA DISTRIBUTION:")
    print(f"   States shape: {states.shape}")
    print(f"   State ranges (normalized):")
    for i, name in enumerate(['phi', 'delta', 'phi_dot', 'delta_dot']):
        s_norm = states[:, i] / s_std[i]
        print(f"     {name}: [{s_norm.min():.3f}, {s_norm.max():.3f}], mean={s_norm.mean():.3f}, std={s_norm.std():.3f}")

    # Simulate DAgger collection
    print("\n2. DAGGER DATA COLLECTION SIMULATION:")
    dagger_data = []
    for seg_i in range(3):
        tau_func = make_tau_func(100 + seg_i, 500)
        phi_init = np.random.uniform(-0.25, 0.25)
        s = np.array([phi_init, 0.0, 0.0, 0.0])

        segment_states = []
        segment_actions = []
        for step in range(500):
            tau = tau_func(step, s)
            s_norm = s / s_std
            a_norm = tau / a_std
            segment_states.append(s_norm.copy())
            segment_actions.append(a_norm)
            s = real_step(s, tau)
            if abs(s[0]) > np.pi / 3:
                break

        dagger_data.append({
            'segment': seg_i,
            'n_steps': len(segment_states),
            'states': np.array(segment_states),
            'actions': np.array(segment_actions),
        })
        print(f"   Segment {seg_i}: {len(segment_states)} steps, phi_init={phi_init:.3f}")

    # Compare distributions
    print("\n3. DISTRIBUTION SHIFT ANALYSIS:")
    base_norm = states / s_std
    base_mean = base_norm.mean(axis=0)
    base_std = base_norm.std(axis=0)

    for seg_data in dagger_data:
        if len(seg_data['states']) > 0:
            dagger_mean = seg_data['states'].mean(axis=0)
            dagger_std = seg_data['states'].std(axis=0)
            shift = np.abs(dagger_mean - base_mean) / base_std
            print(f"\n   Segment {seg_data['segment']} vs base training:")
            for i, name in enumerate(['phi', 'delta', 'phi_dot', 'delta_dot']):
                print(f"     {name}: shift={shift[i]:.2f}σ, "
                      f"dagger_range=[{seg_data['states'][:, i].min():.3f}, {seg_data['states'][:, i].max():.3f}]")

    return dagger_data


def analyze_evaluation_distribution():
    """Analyze evaluation trajectory distribution."""
    print("\n4. EVALUATION TRAJECTORY DISTRIBUTION:")
    from methods_common import generate_training_data, real_step
    from methods_evaluate import make_tau_func

    # Get normalization constants
    _, _, _, s_std_eval, _, _ = generate_training_data(30000, seed=42)

    eval_data = []
    for seg_i in range(5):
        tau_func = make_tau_func(seg_i, 500)
        phi_init = np.random.uniform(-0.25, 0.25)
        s = np.array([phi_init, 0.0, 0.0, 0.0])

        states_norm = []
        for step in range(500):
            tau = tau_func(step, s)
            s_norm = s / s_std_eval
            states_norm.append(s_norm.copy())
            s = real_step(s, tau)
            if abs(s[0]) > np.pi / 3:
                break

        eval_data.append({
            'segment': seg_i,
            'n_steps': len(states_norm),
            'states': np.array(states_norm),
        })

    # Compare with training
    base_norm_states, _, _, s_std, _, _ = generate_training_data(30000, seed=42)
    base_norm = base_norm_states / s_std_eval
    base_mean = base_norm.mean(axis=0)
    base_std = base_norm.std(axis=0)

    print(f"   Base training: mean={base_mean}, std={base_std}")

    for seg_data in eval_data:
        if len(seg_data['states']) > 0:
            eval_mean = seg_data['states'].mean(axis=0)
            eval_std = seg_data['states'].std(axis=0)
            shift = np.abs(eval_mean - base_mean) / base_std
            max_shift = np.max(np.abs(seg_data['states'] - base_mean) / base_std, axis=0)
            print(f"\n   Segment {seg_data['segment']}: {seg_data['n_steps']} steps")
            for i, name in enumerate(['phi', 'delta', 'phi_dot', 'delta_dot']):
                print(f"     {name}: mean_shift={shift[i]:.2f}σ, max_shift={max_shift[i]:.2f}σ")

    return eval_data


if __name__ == '__main__':
    dagger_data = analyze_dagger_process()
    eval_data = analyze_evaluation_distribution()

    # Save results
    output = {
        'dagger_segments': [{
            'segment': d['segment'],
            'n_steps': d['n_steps'],
            'state_mean': d['states'].mean(axis=0).tolist() if len(d['states']) > 0 else [],
            'state_std': d['states'].std(axis=0).tolist() if len(d['states']) > 0 else [],
        } for d in dagger_data],
        'eval_segments': [{
            'segment': d['segment'],
            'n_steps': d['n_steps'],
            'state_mean': d['states'].mean(axis=0).tolist() if len(d['states']) > 0 else [],
            'state_std': d['states'].std(axis=0).tolist() if len(d['states']) > 0 else [],
        } for d in eval_data],
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dagger_analysis_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
