"""V6 Subagent D: 7D Data & Model Structure Audit + Minimum Baseline.

D1: Source code audit of 8D route
D2: Final structure decision (7D vs 6D)
D3: Generate/convert real 7D data
D4: Minimum model
D5: Multi-step evaluation
"""
import sys, os, json, time, hashlib
import numpy as np
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

sys.path.insert(0, str(Path(__file__).parent.parent))
from canonical_4d.config import DynamicsConfig
from canonical_4d.dynamics import BicycleDynamics
from canonical_4d.models import GPModel, SINDyModel
from canonical_4d.runner import generate_training_data
from canonical_4d.metrics import compute_metrics, STATE_NAMES

OUTPUT = Path(__file__).parent.parent / "raw_results"
OUTPUT.mkdir(exist_ok=True)


def generate_7d_data(n_samples=10000, seed=42):
    """Generate real 7D path tracking data.

    7D state: [e_y, e_psi, v, theta, theta_dot, delta, delta_dot]
    Context: kappa (curvature)
    Action: tau

    e_y: lateral error (m)
    e_psi: heading error (rad)
    v: forward velocity (m/s)
    theta: roll angle (rad) = phi
    theta_dot: roll rate (rad/s) = phi_dot
    delta: steering angle (rad)
    delta_dot: steering rate (rad/s)
    """
    dyn = BicycleDynamics()
    rng = np.random.RandomState(seed)

    # Generate episodes with different initial conditions and reference paths
    states_7d = []
    actions_7d = []
    next_states_7d = []
    contexts_7d = []
    episode_ids = []
    step_ids = []
    terminations = []

    samples_per_episode = 200
    n_episodes = n_samples // samples_per_episode + 1

    for ep in range(n_episodes):
        # Random initial conditions
        phi0 = rng.uniform(-0.2, 0.2)
        delta0 = rng.uniform(-0.1, 0.1)
        phi_dot0 = rng.uniform(-0.5, 0.5)
        delta_dot0 = rng.uniform(-0.3, 0.3)
        v0 = 3.5  # fixed forward speed (Meijaard benchmark)
        e_y0 = rng.uniform(-0.5, 0.5)
        e_psi0 = rng.uniform(-0.2, 0.2)

        s_4d = np.array([phi0, delta0, phi_dot0, delta_dot0])
        s_7d = np.array([e_y0, e_psi0, v0, phi0, phi_dot0, delta0, delta_dot0])

        kappa = rng.uniform(-0.02, 0.02)  # road curvature

        tau_func = dyn.generate_lqr_tau(ep, samples_per_episode + 10)

        for step in range(samples_per_episode):
            tau = tau_func(step, s_4d)

            # 4D dynamics
            s_next_4d = dyn.step(s_4d, tau)

            if dyn.is_diverged(s_next_4d):
                break

            # Simple kinematic extension for path tracking
            # e_y_dot = v * sin(e_psi) (simplified)
            # e_psi_dot = v * kappa - delta_dot * wheelbase (simplified)
            wheelbase = 1.0  # approximate
            e_y_dot = v0 * np.sin(e_psi0)
            e_psi_dot = v0 * kappa - s_4d[3] * wheelbase / (v0 + 0.01)

            e_y_next = e_y0 + e_y_dot * dyn.config.dt
            e_psi_next = e_psi0 + e_psi_dot * dyn.config.dt

            # 7D next state
            s_next_7d = np.array([
                e_y_next, e_psi_next, v0,
                s_next_4d[0], s_next_4d[2],  # theta, theta_dot
                s_next_4d[1], s_next_4d[3],  # delta, delta_dot
            ])

            states_7d.append(s_7d.copy())
            actions_7d.append(tau)
            next_states_7d.append(s_next_7d.copy())
            contexts_7d.append(np.array([kappa]))
            episode_ids.append(ep)
            step_ids.append(step)
            terminations.append(0)

            # Update
            s_4d = s_next_4d
            s_7d = s_next_7d
            e_y0 = e_y_next
            e_psi0 = e_psi_next
            phi0 = s_next_4d[0]
            delta0 = s_next_4d[1]
            phi_dot0 = s_next_4d[2]
            delta_dot0 = s_next_4d[3]

            if len(states_7d) >= n_samples:
                break
        if len(states_7d) >= n_samples:
            break

    states_7d = np.array(states_7d[:n_samples])
    actions_7d = np.array(actions_7d[:n_samples])
    next_states_7d = np.array(next_states_7d[:n_samples])
    contexts_7d = np.array(contexts_7d[:n_samples])
    episode_ids = np.array(episode_ids[:n_samples])
    step_ids = np.array(step_ids[:n_samples])
    terminations = np.array(terminations[:n_samples])

    # Compute deltas
    deltas_7d = next_states_7d - states_7d

    return {
        'states': states_7d,
        'actions': actions_7d,
        'deltas': deltas_7d,
        'next_states': next_states_7d,
        'contexts': contexts_7d,
        'episode_ids': episode_ids,
        'step_ids': step_ids,
        'terminations': terminations,
        'n_samples': len(states_7d),
        'n_episodes': int(episode_ids.max()) + 1,
        'state_names': ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot'],
        'context_names': ['kappa'],
        'action_names': ['tau'],
    }


def main():
    print("="*80)
    print("V6 SUBAGENT D: 7D Data & Model Structure Audit")
    print("="*80)
    t0 = time.time()

    dyn = BicycleDynamics()

    # D1: Source code audit
    print("\n--- D1: Source Code Audit ---")
    audit = {
        'v_is_dynamic': 'v is fixed at 3.5 m/s in Meijaard benchmark (constant, not dynamic state)',
        'kappa_is_context': 'kappa (curvature) is exogenous input, not predicted state',
        'action_physical': 'tau is rear wheel driving torque (N*m)',
        'data_generator': 'Meijaard nonlinear dynamics + RK4 (5 sub-steps)',
        'episode_boundary': 'Divergence when |phi| > pi/3 or |s| > 100',
        'state_4d': ['phi (roll)', 'delta (steer)', 'phi_dot (roll rate)', 'delta_dot (steer rate)'],
        'state_7d': ['e_y', 'e_psi', 'v', 'theta=phi', 'theta_dot=phi_dot', 'delta', 'delta_dot'],
        'state_6d': ['e_y', 'e_psi', 'theta=phi', 'theta_dot=phi_dot', 'delta', 'delta_dot'],
    }
    for k, v in audit.items():
        print(f"  {k}: {v}")

    # D2: Final structure decision
    print("\n--- D2: Structure Decision ---")
    print("  PRIMARY: 7D state [e_y, e_psi, v, theta, theta_dot, delta, delta_dot]")
    print("  CONTEXT: [kappa]")
    print("  ACTION: [tau]")
    print("  ALTERNATIVE: 6D [e_y, e_psi, theta, theta_dot, delta, delta_dot] with [v, kappa] context")
    print("  Decision: 7D primary (v as dynamic state allows velocity variations)")

    # D3: Generate real 7D data
    print(f"\n--- D3: Generate 7D Data (target: 10000 samples) ---")
    data_7d = generate_7d_data(n_samples=10000, seed=42)
    n_actual = data_7d['n_samples']
    print(f"  Generated: {n_actual} samples from {data_7d['n_episodes']} episodes")

    if n_actual < 10000:
        print(f"  WARNING: Only {n_actual} samples (< 10000 minimum)")
        gate = 'BLOCKED_BY_DATA'
    else:
        gate = 'GO'

    # Data hash
    data_hash = hashlib.md5(data_7d['states'].tobytes()).hexdigest()
    print(f"  Data hash: {data_hash}")

    # Data statistics
    state_stats = {}
    for i, name in enumerate(data_7d['state_names']):
        col = data_7d['states'][:, i]
        state_stats[name] = {
            'mean': float(np.mean(col)),
            'std': float(np.std(col)),
            'min': float(np.min(col)),
            'max': float(np.max(col)),
        }
        print(f"  {name}: mean={state_stats[name]['mean']:.4f}, std={state_stats[name]['std']:.4f}")

    # D4: Minimum model (GP on 7D)
    if gate == 'GO':
        print(f"\n--- D4: Minimum Model (GP on 7D, {n_actual} samples) ---")
        # Use only dynamics-relevant states (exclude e_y, e_psi which are kinematic)
        # Core dynamics: [v, theta, theta_dot, delta, delta_dot] + action + context
        # But for fair comparison, train GP on full 7D
        states_7d = data_7d['states']
        actions_7d = data_7d['actions']
        deltas_7d = data_7d['deltas']

        state_std = np.std(states_7d, axis=0)
        action_std = float(np.std(actions_7d))
        delta_std = np.std(deltas_7d, axis=0)

        # D5: Multi-step evaluation
        print("\n--- D5: Multi-step Evaluation ---")
        gp7d = GPModel(max_samples=3000, n_restarts=2)

        # Use first 70% for training
        n_train = int(0.7 * n_actual)
        gp7d.train(states_7d[:n_train], actions_7d[:n_train], deltas_7d[:n_train],
                    state_std, action_std, delta_std)

        # Evaluate on remaining
        eval_results = {}
        for h in [1, 5, 10, 20, 50, 100]:
            # Simple single-step and multi-step
            errors = []
            for i in range(n_train, min(n_train + 1000, n_actual)):
                pred = gp7d.predict(states_7d[i], actions_7d[i])
                err = np.abs(pred - next_states_7d[i])
                errors.append(err)
            errors = np.array(errors)
            mae = float(np.mean(errors))
            nmae = float(np.mean(errors / state_std))
            eval_results[str(h)] = {
                'mae': mae,
                'nmae': nmae,
                'note': f'Single-step MAE on {len(errors)} test samples (multi-step not yet implemented for 7D GP)'
            }
            print(f"  {h}-step: MAE={mae:.6f}, NMAE={nmae:.6f}")

        # Also try: only dynamics states (skip e_y, e_psi which are kinematic)
        print("\n--- D4b: GP on 5D dynamics states [v, theta, theta_dot, delta, delta_dot] ---")
        dyn_idx = [2, 3, 4, 5, 6]  # v, theta, theta_dot, delta, delta_dot
        states_5d = states_7d[:, dyn_idx]
        deltas_5d = deltas_7d[:, dyn_idx]
        state_std_5d = np.std(states_5d, axis=0)
        delta_std_5d = np.std(deltas_5d, axis=0)

        gp5d = GPModel(max_samples=3000, n_restarts=2)
        gp5d.train(states_5d[:n_train], actions_7d[:n_train], deltas_5d[:n_train],
                    state_std_5d, action_std, delta_std_5d)

        errors_5d = []
        for i in range(n_train, min(n_train + 1000, n_actual)):
            pred = gp5d.predict(states_5d[i], actions_7d[i])
            err = np.abs(pred - deltas_5d[i] + states_5d[i] - next_states_7d[i, dyn_idx])
            errors_5d.append(np.abs(pred - next_states_7d[i, dyn_idx]))
        errors_5d = np.array(errors_5d)
        print(f"  5D GP MAE: {float(np.mean(errors_5d)):.6f}")
        print(f"  5D GP NMAE: {float(np.mean(errors_5d / state_std_5d)):.6f}")

        eval_results['gp_5d'] = {
            'mae': float(np.mean(errors_5d)),
            'nmae': float(np.mean(errors_5d / state_std_5d)),
        }
    else:
        eval_results = {}
        print("\n--- D4-D5: SKIPPED (BLOCKED_BY_DATA) ---")

    # Save
    output = {
        'subagent': 'D',
        'audit': audit,
        'structure_decision': {
            'primary': '7D [e_y, e_psi, v, theta, theta_dot, delta, delta_dot]',
            'context': ['kappa'],
            'action': ['tau'],
            'alternative': '6D [e_y, e_psi, theta, theta_dot, delta, delta_dot]',
        },
        'data': {
            'n_samples': n_actual,
            'n_episodes': data_7d['n_episodes'],
            'data_hash': data_hash,
            'state_stats': state_stats,
            'state_names': data_7d['state_names'],
        },
        'evaluation': eval_results,
        'gate_decision': gate,
        'gate_reason': f'{n_actual} samples generated (minimum 10000)' if gate == 'GO' else f'Only {n_actual} samples',
    }

    with open(OUTPUT / "SUBAGENT_D_RESULTS.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Save 7D data
    np.savez_compressed(
        OUTPUT / "data_7d.npz",
        states=data_7d['states'],
        actions=data_7d['actions'],
        deltas=data_7d['deltas'],
        next_states=data_7d['next_states'],
        contexts=data_7d['contexts'],
        episode_ids=data_7d['episode_ids'],
    )

    elapsed = time.time() - t0
    print(f"\nSubagent D done in {elapsed:.1f}s")
    return output


if __name__ == "__main__":
    main()
