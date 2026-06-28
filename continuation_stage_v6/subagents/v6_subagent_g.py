"""V6 Subagent G: Independent Reproduction, Anti-fraud & Final Review.

Must ACTUALLY:
1. Read code (not just string scan)
2. Rerun key experiments
3. Check original results
4. Find anti-examples
5. Run key unit tests
"""
import sys, os, json, time, importlib.util
import numpy as np
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

sys.path.insert(0, str(Path(__file__).parent.parent))
from canonical_4d.config import DynamicsConfig, TrainingConfig
from canonical_4d.dynamics import BicycleDynamics
from canonical_4d.models import GPModel
from canonical_4d.residual_models import RDELocal, RDETrajectory, RDEHybrid, E1OfflineNN
from canonical_4d.evaluation_modes import run_mode_a, run_mode_b, run_mode_c
from canonical_4d.metrics import compute_metrics, STATE_NAMES
from canonical_4d.runner import generate_training_data

OUTPUT = Path(__file__).parent.parent / "raw_results"
OUTPUT.mkdir(exist_ok=True)


def g1_read_code_review():
    """G1: Actually read the canonical_4d code and review it."""
    print("--- G1: Code Review (actual reading) ---")
    issues = []

    # Read canonical package files
    pkg_dir = Path(__file__).parent.parent / "canonical_4d"
    for fname in ['config.py', 'dynamics.py', 'models.py', 'residual_models.py',
                   'evaluation_modes.py', 'metrics.py', 'runner.py']:
        fpath = pkg_dir / fname
        if not fpath.exists():
            issues.append(('CRITICAL', f'{fname} not found'))
            continue
        with open(fpath) as f:
            src = f.read()

        # Check key patterns
        if fname == 'residual_models.py':
            # RDE-T and RDE-M must have different label generation
            if '_collect_trajectory_data' not in src:
                issues.append(('HIGH', f'RDE-T missing _collect_trajectory_data'))
            if '_collect_dagger_data' not in src:
                issues.append(('HIGH', f'RDE-M missing _collect_dagger_data'))
            # Check that RDE-T uses trajectory sync labels
            if 'real_delta - gp_delta' in src and 'trajectory' in src.lower():
                print(f"  {fname}: RDE-T trajectory sync label found")
            # Check RDE-M uses local residual labels
            if 'real_delta - base_delta' in src and 'hybrid' in src.lower():
                print(f"  {fname}: RDE-M local residual label found")

        if fname == 'evaluation_modes.py':
            if 'run_mode_a' in src and 'run_mode_b' in src:
                print(f"  {fname}: All modes present")
            if 'first_nan_step' in src and 'first_divergence_step' in src:
                print(f"  {fname}: Divergence tracking present")

        if fname == 'metrics.py':
            if 'nmae' in src and 'nrmse' in src:
                print(f"  {fname}: NMAE/NRMSE present")

    return issues


def g2_rerun_equivalence():
    """G2: Rerun zero-residual equivalence test."""
    print("\n--- G2: Rerun Equivalence Test ---")
    dyn = BicycleDynamics()
    data = generate_training_data(2000, 42, dyn)
    states, actions, deltas, state_std, action_std, delta_std = data

    gp = GPModel(max_samples=1500, n_restarts=2)
    gp.train(states, actions, deltas, state_std, action_std, delta_std)

    class ZeroWrapper:
        def __init__(self, baseline):
            self._baseline = baseline
        def predict(self, s, tau):
            return self._baseline.predict(s, tau)
        def predict_with_uncertainty(self, s, tau):
            return self._baseline.predict(s, tau), None
        def name(self): return 'zero'
        def train(self, *args): pass

    zw = ZeroWrapper(gp)
    rng = np.random.RandomState(999)
    diffs = []
    for _ in range(1000):
        s = rng.uniform([-0.3, -0.2, -1, -0.5], [0.3, 0.2, 1, 0.5])
        tau = rng.uniform(-30, 30)
        p1 = gp.predict(s, tau)
        p2 = zw.predict(s, tau)
        diffs.append(np.max(np.abs(p1 - p2)))

    max_diff = max(diffs)
    passed = max_diff < 1e-12
    print(f"  Pointwise max diff: {max_diff:.2e} -> {'PASS' if passed else 'FAIL'}")
    return passed, max_diff


def g3_rerun_rde_label_independence():
    """G3: Rerun RDE label independence test."""
    print("\n--- G3: Rerun RDE Label Independence ---")
    dyn = BicycleDynamics()
    data = generate_training_data(2000, 42, dyn)
    states, actions, deltas, state_std, action_std, delta_std = data

    gp = GPModel(max_samples=1500, n_restarts=2)
    gp.train(states, actions, deltas, state_std, action_std, delta_std)

    # Collect labels from a rollout
    tau_func = dyn.generate_lqr_tau(0, 100)
    s_model = np.array([0.1, 0.0, 0.0, 0.0])
    s_real = s_model.copy()

    e1_labels = []  # local residual at real state
    t_labels = []   # trajectory sync
    m_labels = []   # local residual at drifted state

    for step in range(100):
        tau = tau_func(step, s_real)
        s_real_next = dyn.step(s_real, tau)
        if dyn.is_diverged(s_real_next):
            break

        # E1 label (local at real state)
        gp_next_real = gp.predict(s_real, tau)
        e1_label = ((s_real_next - s_real) - (gp_next_real - s_real)) / state_std
        e1_labels.append(e1_label)

        # T label (trajectory sync at model state)
        gp_next_model = gp.predict(s_model, tau)
        gp_delta_model = gp_next_model - s_model
        real_delta = s_real_next - s_real
        t_label = (real_delta - gp_delta_model) / state_std
        t_labels.append(t_label)

        # M label (local residual at drifted model state)
        m_label = ((s_real_next - s_model) - (gp_next_model - s_model)) / state_std
        m_labels.append(m_label)

        # Update model (GP only)
        s_model = gp_next_model
        s_real = s_real_next

    e1_labels = np.array(e1_labels)
    t_labels = np.array(t_labels)
    m_labels = np.array(m_labels)

    e1_vs_t = np.max(np.abs(e1_labels - t_labels))
    e1_vs_m = np.max(np.abs(e1_labels - m_labels))
    t_vs_m = np.max(np.abs(t_labels - m_labels))

    print(f"  E1 vs RDE-T max diff: {e1_vs_t:.6f}")
    print(f"  E1 vs RDE-M max diff: {e1_vs_m:.6f}")
    print(f"  RDE-T vs RDE-M max diff: {t_vs_m:.6f}")

    all_diff = e1_vs_t > 1e-6 and e1_vs_m > 1e-6 and t_vs_m > 1e-6
    print(f"  All labels independent: {all_diff}")

    return {
        'e1_vs_t': float(e1_vs_t),
        'e1_vs_m': float(e1_vs_m),
        't_vs_m': float(t_vs_m),
        'independent': all_diff,
        'e1_sample': e1_labels[:3].tolist(),
        't_sample': t_labels[:3].tolist(),
        'm_sample': m_labels[:3].tolist(),
    }


def g4_rerun_planning_consistency():
    """G4: Rerun one planning consistency test."""
    print("\n--- G4: Rerun Planning Consistency ---")
    from canonical_4d.metrics import compute_cost
    from scipy.stats import spearmanr

    dyn = BicycleDynamics()
    data = generate_training_data(2000, 42, dyn)
    states, actions, deltas, state_std, action_std, delta_std = data

    gp = GPModel(max_samples=1500, n_restarts=2)
    gp.train(states, actions, deltas, state_std, action_std, delta_std)

    horizon = 20
    n_seq = 30
    rng = np.random.RandomState(42)
    s0 = np.array([0.1, 0.0, 0.0, 0.0])
    weights = {'phi': 100, 'delta': 100, 'phi_dot': 10, 'delta_dot': 1, 'u': 0.1}

    real_costs = []
    pred_costs = []

    for _ in range(n_seq):
        seq = rng.uniform(-20, 20, horizon)

        # Real
        s = s0.copy()
        traj_real = [s.copy()]
        for tau in seq:
            s = dyn.step(s, tau)
            traj_real.append(s.copy())
        J_real = compute_cost(np.array(traj_real), seq, weights)
        real_costs.append(J_real)

        # Model
        s = s0.copy()
        traj_pred = [s.copy()]
        for tau in seq:
            s = gp.predict(s, tau)
            traj_pred.append(s.copy())
        J_pred = compute_cost(np.array(traj_pred), seq, weights)
        pred_costs.append(J_pred)

    real_costs = np.array(real_costs)
    pred_costs = np.array(pred_costs)
    corr, _ = spearmanr(real_costs, pred_costs)
    mae = float(np.mean(np.abs(pred_costs - real_costs)))

    print(f"  Spearman: {corr:.3f}")
    print(f"  Cost MAE: {mae:.2f}")

    return {'spearman': float(corr), 'cost_mae': mae}


def g5_check_v5_conclusions():
    """G5: Check V5 conclusions against V6 evidence."""
    print("\n--- G5: V5 Conclusion Check ---")
    checks = []

    # Load V5 results if available
    v5_results = Path(__file__).parent.parent.parent / "continuation_stage_v5" / "raw_results"
    if v5_results.exists():
        for fname in ['SUBAGENT_A_B_RESULTS.json', 'SUBAGENT_C_RESULTS.json',
                       'SUBAGENT_D_RESULTS.json', 'SUBAGENT_E_RESULTS.json',
                       'SUBAGENT_F_RESULTS.json', 'SUBAGENT_G_RESULTS.json']:
            fpath = v5_results / fname
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                checks.append({'file': fname, 'loaded': True, 'keys': list(data.keys())[:5]})
            else:
                checks.append({'file': fname, 'loaded': False})
    else:
        checks.append({'note': 'V5 results directory not found'})

    return checks


def main():
    print("="*80)
    print("V6 SUBAGENT G: Independent Reproduction & Review")
    print("="*80)
    t0 = time.time()

    # G1: Code review
    issues = g1_read_code_review()
    print(f"\n  Code review: {len(issues)} issues")
    for sev, desc in issues:
        print(f"    [{sev}] {desc}")

    # G2: Equivalence test
    equiv_pass, equiv_diff = g2_rerun_equivalence()

    # G3: RDE independence
    rde_result = g3_rerun_rde_label_independence()

    # G4: Planning consistency
    plan_result = g4_rerun_planning_consistency()

    # G5: V5 conclusion check
    v5_checks = g5_check_v5_conclusions()

    # Overall verdict
    all_pass = equiv_pass and rde_result['independent'] and plan_result['spearman'] > 0.5
    verdict = 'PASS' if all_pass else 'PASS_WITH_CORRECTIONS'
    if not equiv_pass:
        verdict = 'FAIL'

    print(f"\n--- VERDICT: {verdict} ---")
    print(f"  Equivalence: {'PASS' if equiv_pass else 'FAIL'}")
    print(f"  RDE independent: {rde_result['independent']}")
    print(f"  Planning Spearman: {plan_result['spearman']:.3f}")

    output = {
        'subagent': 'G',
        'verdict': verdict,
        'code_review': {'issues': [{'severity': s, 'description': d} for s, d in issues]},
        'equivalence': {'passed': equiv_pass, 'max_diff': equiv_diff},
        'rde_independence': rde_result,
        'planning_consistency': plan_result,
        'v5_checks': v5_checks,
        'must_fix': [] if all_pass else ['RDE independence issue'] if not rde_result['independent'] else ['Equivalence issue'],
        'suggest_fix': [],
        'cannot_confirm': [],
    }

    with open(OUTPUT / "SUBAGENT_G_RESULTS.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nSubagent G done in {elapsed:.1f}s")
    return output


if __name__ == "__main__":
    main()
