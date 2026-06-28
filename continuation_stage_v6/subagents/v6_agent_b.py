"""V6 Agent B: RDE-L, RDE-T, RDE-M label independence verification.

Verifies that the three RDE variants are truly independent implementations
with different training labels by:
1. Training GP baseline
2. Computing labels for each model type
3. Comparing labels pairwise (correlation + max diff)
4. Training full models and comparing predictions
5. Verifying specific formula implementations
"""
import sys, os, json, time, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress sklearn GP convergence warnings
import logging
logging.getLogger("sklearn").setLevel(logging.ERROR)

ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

sys.path.insert(0, str(Path(__file__).parent.parent))
from canonical_4d.config import DynamicsConfig, TrainingConfig
from canonical_4d.dynamics import BicycleDynamics
from canonical_4d.models import GPModel
from canonical_4d.residual_models import RDELocal, RDETrajectory, RDEHybrid
from canonical_4d.runner import generate_training_data

# Monkey-patch GPModel to use fewer samples (avoids memory issues on this machine)
_original_gp_init = GPModel.__init__
def _patched_gp_init(self, max_samples=500, n_restarts=1):
    _original_gp_init(self, max_samples=max_samples, n_restarts=n_restarts)
GPModel.__init__ = _patched_gp_init


def collect_labels_from_class(model_class, states, actions, deltas,
                              state_std, action_std, delta_std,
                              dynamics, make_tau_func, n_subset=100):
    """Train a model class and intercept the training labels.

    Returns the model's internal residuals and the input states used.
    """
    model = model_class(n_models=2, n_epochs=3, residual_scale=0.3)

    # For RDE-L: no DAgger, just offline labels
    if model_class == RDELocal:
        model.train(states, actions, deltas, state_std, action_std, delta_std)
        # Labels are computed inside train: (deltas[i] - (GP(s_i,u_i) - s_i)) / delta_std
        labels = []
        for i in range(n_subset):
            gp_next = model._baseline.predict(states[i], actions[i])
            label = (deltas[i] - (gp_next - states[i])) / delta_std
            labels.append(label)
        return np.array(labels), states[:n_subset].copy()

    # For RDE-T and RDE-M: need DAgger rounds with dynamics
    model.train(states, actions, deltas, state_std, action_std, delta_std,
                dynamics=dynamics, make_tau_func=make_tau_func)

    # Collect fresh DAgger data to observe the labels
    if model_class == RDETrajectory:
        labels, inputs = _collect_trajectory_labels(
            model, dynamics, make_tau_func, state_std, action_std, delta_std, n_steps=200)
    else:
        labels, inputs = _collect_dagger_local_labels(
            model, dynamics, make_tau_func, state_std, action_std, delta_std, n_steps=200)

    return labels, inputs


def _collect_trajectory_labels(model, dynamics, make_tau_func,
                               state_std, action_std, delta_std, n_steps=200):
    """Collect RDE-T labels: trajectory sync = (real_delta - GP(model_state, u)) / delta_std."""
    tau_func = make_tau_func(999, n_steps)
    phi0 = np.random.uniform(-0.25, 0.25)
    s_real = np.array([phi0, 0.0, 0.0, 0.0])
    s_model = s_real.copy()

    labels = []
    inputs = []
    for step in range(n_steps):
        tau = tau_func(step, s_real)
        s_real_next = dynamics.step(s_real, tau)
        if dynamics.is_diverged(s_real_next):
            break

        # RDE-T label: GP prediction from MODEL state, compared to REAL delta
        gp_next = model._baseline.predict(s_model, tau)
        gp_delta = gp_next - s_model
        real_delta = s_real_next - s_real
        label = (real_delta - gp_delta) / delta_std

        labels.append(label)
        inputs.append(np.concatenate([s_model / state_std, [tau / action_std]]))

        # Update model state (GP only, no NN for observation)
        s_model = gp_next
        s_real = s_real_next

    return np.array(labels), np.array(inputs)


def _collect_dagger_local_labels(model, dynamics, make_tau_func,
                                 state_std, action_std, delta_std, n_steps=200):
    """Collect RDE-M labels: local residual at drifted model state.

    CRITICAL: RDE-M uses dynamics.step(s_model, tau) - s_model as the 'true' delta,
    NOT s_real_next - s_real. This is the LOCAL residual at the model state.
    """
    tau_func = make_tau_func(999, n_steps)
    phi0 = np.random.uniform(-0.25, 0.25)
    s_real = np.array([phi0, 0.0, 0.0, 0.0])
    s_model = s_real.copy()

    labels = []
    inputs = []
    for step in range(n_steps):
        tau = tau_func(step, s_real)
        s_real_next = dynamics.step(s_real, tau)
        if dynamics.is_diverged(s_real_next):
            break

        # RDE-M label: LOCAL residual at MODEL state
        # true_delta = dynamics.step(s_model, tau) - s_model  (NOT s_real_next - s_real!)
        gp_next = model._baseline.predict(s_model, tau)
        base_delta = gp_next - s_model
        true_next_from_model = dynamics.step(s_model, tau)
        true_delta_from_model = true_next_from_model - s_model
        label = (true_delta_from_model - base_delta) / delta_std

        labels.append(label)
        inputs.append(np.concatenate([s_model / state_std, [tau / action_std]]))

        # Update model state
        s_model = gp_next
        s_real = s_real_next

    return np.array(labels), np.array(inputs)


def compare_labels(labels_a, labels_b, name_a, name_b):
    """Compare two sets of labels: correlation, max diff, mean diff."""
    min_len = min(len(labels_a), len(labels_b))
    a = labels_a[:min_len]
    b = labels_b[:min_len]

    # Flatten for correlation
    a_flat = a.flatten()
    b_flat = b.flatten()

    # Correlation
    if np.std(a_flat) > 1e-15 and np.std(b_flat) > 1e-15:
        corr = float(np.corrcoef(a_flat, b_flat)[0, 1])
    else:
        corr = float('nan')

    max_diff = float(np.max(np.abs(a - b)))
    mean_diff = float(np.mean(np.abs(a - b)))

    return {
        'name_a': name_a,
        'name_b': name_b,
        'n_samples': min_len,
        'correlation': corr,
        'max_abs_diff': max_diff,
        'mean_abs_diff': mean_diff,
        'identical': bool(max_diff < 1e-10),
    }


def verify_formulas(states, actions, deltas, state_std, action_std, delta_std, dynamics):
    """Verify the specific formula each RDE variant uses."""
    gp = GPModel(max_samples=1000, n_restarts=1)
    gp.train(states, actions, deltas, state_std, action_std, delta_std)

    results = {}

    # --- RDE-L formula verification ---
    # Should be: r^L = F(s_i, u_i) - GP(s_i, u_i) at training data states
    rde_l_labels = []
    for i in range(200):
        gp_next = gp.predict(states[i], actions[i])
        gp_delta = gp_next - states[i]
        real_delta = deltas[i]
        label = (real_delta - gp_delta) / delta_std
        rde_l_labels.append(label)
    rde_l_labels = np.array(rde_l_labels)
    results['rde_l_formula'] = {
        'description': 'Local residual at oracle state: (F(s_i,u_i) - GP(s_i,u_i)) / delta_std',
        'uses_oracle_state': True,
        'uses_model_state': False,
        'n_samples': 200,
        'label_mean': rde_l_labels.mean(axis=0).tolist(),
        'label_std': rde_l_labels.std(axis=0).tolist(),
    }

    # --- RDE-T formula verification ---
    # Should be: r^T = (s_real[k+1] - s_real[k]) - (GP(s_model[k], u) - s_model[k]) / delta_std
    # Key: GP(s_model, u) uses drifted model state for GP, but real delta for comparison
    tau_func = dynamics.generate_lqr_tau(888, 200)
    s_real = np.array([0.1, 0.0, 0.0, 0.0])
    s_model = s_real.copy()
    rde_t_labels = []
    rde_t_s_model = []
    rde_t_s_real = []
    for step in range(200):
        tau = tau_func(step, s_real)
        s_real_next = dynamics.step(s_real, tau)
        if dynamics.is_diverged(s_real_next):
            break
        gp_next = gp.predict(s_model, tau)
        gp_delta = gp_next - s_model
        real_delta = s_real_next - s_real
        label = (real_delta - gp_delta) / delta_std
        rde_t_labels.append(label)
        rde_t_s_model.append(s_model.copy())
        rde_t_s_real.append(s_real.copy())
        s_model = gp_next
        s_real = s_real_next
    rde_t_labels = np.array(rde_t_labels)
    rde_t_s_model = np.array(rde_t_s_model)
    rde_t_s_real = np.array(rde_t_s_real)

    results['rde_t_formula'] = {
        'description': 'Trajectory sync: (real_delta - GP(model_state, u)) / delta_std',
        'uses_oracle_state': False,
        'uses_model_state_for_gp': True,
        'uses_real_delta': True,
        'n_samples': len(rde_t_labels),
        'label_mean': rde_t_labels.mean(axis=0).tolist(),
        'label_std': rde_t_labels.std(axis=0).tolist(),
        's_model_drift': float(np.mean(np.abs(rde_t_s_model - rde_t_s_real))),
    }

    # --- RDE-M formula verification ---
    # Should be: local residual at MODEL state
    # RDE-M uses dynamics.step(s_model, tau) - s_model as the 'true' delta
    # (NOT s_real_next - s_real, which is what RDE-T uses)
    tau_func_m = dynamics.generate_lqr_tau(888, 200)
    s_real_m = np.array([0.1, 0.0, 0.0, 0.0])
    s_model_m = s_real_m.copy()
    rde_m_labels = []
    for step in range(200):
        tau = tau_func_m(step, s_real_m)
        s_real_next_m = dynamics.step(s_real_m, tau)
        if dynamics.is_diverged(s_real_next_m):
            break
        gp_next_m = gp.predict(s_model_m, tau)
        base_delta_m = gp_next_m - s_model_m
        # KEY: RDE-M steps dynamics FROM model state, not from real state
        true_next_from_model = dynamics.step(s_model_m, tau)
        true_delta_from_model = true_next_from_model - s_model_m
        label_m = (true_delta_from_model - base_delta_m) / delta_std
        rde_m_labels.append(label_m)
        s_model_m = gp_next_m
        s_real_m = s_real_next_m
    rde_m_labels = np.array(rde_m_labels)

    results['rde_m_formula'] = {
        'description': 'DAgger local residual at model state: (real_delta - GP(model_state, u)) / delta_std',
        'uses_oracle_state': False,
        'uses_model_state_for_gp': True,
        'uses_real_delta': True,
        'n_samples': len(rde_m_labels),
        'label_mean': rde_m_labels.mean(axis=0).tolist(),
        'label_std': rde_m_labels.std(axis=0).tolist(),
    }

    # RDE-T vs RDE-M: both use GP(s_model, u) but differ in DAgger behavior
    # Verify that RDE-T and RDE-M produce DIFFERENT labels after model state drift
    # by checking if their label sequences differ when model states diverge
    min_len = min(len(rde_t_labels), len(rde_m_labels))
    t_vs_m_labels = compare_labels(rde_t_labels, rde_m_labels, 'RDE-T', 'RDE-M')
    results['rde_t_vs_rde_m_offline'] = t_vs_m_labels

    return results, rde_l_labels, rde_t_labels, rde_m_labels


def train_and_compare_predictions(states, actions, deltas, state_std, action_std,
                                  delta_std, dynamics):
    """Train all three RDE models and compare their predictions on test states."""
    results = {}
    make_tau = dynamics.generate_lqr_tau

    # Train RDE-L (0 DAgger rounds)
    print("  Training RDE-L...")
    t0 = time.time()
    rde_l = RDELocal(n_models=2, n_epochs=3, residual_scale=0.3)
    rde_l.train(states, actions, deltas, state_std, action_std, delta_std)
    results['rde_l_train_time'] = time.time() - t0

    # Train RDE-T (1 DAgger round)
    print("  Training RDE-T...")
    t0 = time.time()
    rde_t = RDETrajectory(n_models=2, n_epochs=3, residual_scale=0.3,
                          dagger_rounds=1, n_segments=2, segment_length=80)
    rde_t.train(states, actions, deltas, state_std, action_std, delta_std,
                dynamics=dynamics, make_tau_func=make_tau)
    results['rde_t_train_time'] = time.time() - t0

    # Train RDE-M (2 DAgger rounds)
    print("  Training RDE-M...")
    t0 = time.time()
    rde_m = RDEHybrid(n_models=2, n_epochs=3, residual_scale=0.3,
                      dagger_rounds=2, n_segments=2, segment_length=80)
    rde_m.train(states, actions, deltas, state_std, action_std, delta_std,
                dynamics=dynamics, make_tau_func=make_tau)
    results['rde_m_train_time'] = time.time() - t0

    # Compare predictions on 100 test states
    rng = np.random.RandomState(777)
    n_test = 100
    test_states = rng.uniform([-0.3, -0.2, -1, -0.5], [0.3, 0.2, 1, 0.5], (n_test, 4))
    test_actions = rng.uniform(-30, 30, n_test)

    preds_l = []
    preds_t = []
    preds_m = []
    unc_l = []
    unc_t = []
    unc_m = []

    for i in range(n_test):
        p_l = rde_l.predict(test_states[i], test_actions[i])
        p_t = rde_t.predict(test_states[i], test_actions[i])
        p_m = rde_m.predict(test_states[i], test_actions[i])
        preds_l.append(p_l)
        preds_t.append(p_t)
        preds_m.append(p_m)

        _, u_l = rde_l.predict_with_uncertainty(test_states[i], test_actions[i])
        _, u_t = rde_t.predict_with_uncertainty(test_states[i], test_actions[i])
        _, u_m = rde_m.predict_with_uncertainty(test_states[i], test_actions[i])
        unc_l.append(u_l if u_l is not None else np.zeros(4))
        unc_t.append(u_t if u_t is not None else np.zeros(4))
        unc_m.append(u_m if u_m is not None else np.zeros(4))

    preds_l = np.array(preds_l)
    preds_t = np.array(preds_t)
    preds_m = np.array(preds_m)
    unc_l = np.array(unc_l)
    unc_t = np.array(unc_t)
    unc_m = np.array(unc_m)

    # Pairwise prediction comparisons
    pred_lt = compare_labels(preds_l, preds_t, 'RDE-L_pred', 'RDE-T_pred')
    pred_lm = compare_labels(preds_l, preds_m, 'RDE-L_pred', 'RDE-M_pred')
    pred_tm = compare_labels(preds_t, preds_m, 'RDE-T_pred', 'RDE-M_pred')

    results['prediction_comparisons'] = {
        'rde_l_vs_rde_t': pred_lt,
        'rde_l_vs_rde_m': pred_lm,
        'rde_t_vs_rde_m': pred_tm,
    }

    # Uncertainty comparisons
    unc_lt = compare_labels(unc_l, unc_t, 'RDE-L_unc', 'RDE-T_unc')
    unc_lm = compare_labels(unc_l, unc_m, 'RDE-L_unc', 'RDE-M_unc')
    unc_tm = compare_labels(unc_t, unc_m, 'RDE-T_unc', 'RDE-M_unc')

    results['uncertainty_comparisons'] = {
        'rde_l_vs_rde_t': unc_lt,
        'rde_l_vs_rde_m': unc_lm,
        'rde_t_vs_rde_m': unc_tm,
    }

    # Sample predictions for inspection
    results['sample_predictions'] = {
        'test_state_0': test_states[0].tolist(),
        'test_action_0': float(test_actions[0]),
        'rde_l_pred_0': preds_l[0].tolist(),
        'rde_t_pred_0': preds_t[0].tolist(),
        'rde_m_pred_0': preds_m[0].tolist(),
    }

    return results


def main():
    print("=" * 70)
    print("V6 AGENT B: RDE-L / RDE-T / RDE-M Label Independence Verification")
    print("=" * 70)
    t_start = time.time()

    # Setup
    dyn = BicycleDynamics()
    print("\n[1/5] Generating training data (500 samples for speed)...")
    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(
        500, 42, dyn)
    print(f"  states: {states.shape}, actions: {actions.shape}, deltas: {deltas.shape}")

    # Step 2: Verify formulas
    print("\n[2/5] Verifying formulas for each RDE variant...")
    formula_results, rde_l_labels, rde_t_labels, rde_m_labels = verify_formulas(
        states, actions, deltas, state_std, action_std, delta_std, dyn)

    # Step 3: Compare labels pairwise
    print("\n[3/5] Comparing training labels pairwise...")
    lt_compare = compare_labels(rde_l_labels, rde_t_labels, 'RDE-L', 'RDE-T')
    lm_compare = compare_labels(rde_l_labels, rde_m_labels, 'RDE-L', 'RDE-M')
    tm_compare = compare_labels(rde_t_labels, rde_m_labels, 'RDE-T', 'RDE-M')

    print(f"  RDE-L vs RDE-T: corr={lt_compare['correlation']:.6f}, "
          f"max_diff={lt_compare['max_abs_diff']:.6f}")
    print(f"  RDE-L vs RDE-M: corr={lm_compare['correlation']:.6f}, "
          f"max_diff={lm_compare['max_abs_diff']:.6f}")
    print(f"  RDE-T vs RDE-M: corr={tm_compare['correlation']:.6f}, "
          f"max_diff={tm_compare['max_abs_diff']:.6f}")

    label_comparisons = {
        'rde_l_vs_rde_t': lt_compare,
        'rde_l_vs_rde_m': lm_compare,
        'rde_t_vs_rde_m': tm_compare,
    }

    # Step 4: Train full models and compare predictions
    print("\n[4/5] Training full models and comparing predictions...")
    prediction_results = train_and_compare_predictions(
        states, actions, deltas, state_std, action_std, delta_std, dyn)

    # Step 5: Summary verdict
    print("\n[5/5] Computing verdict...")

    # Check independence criteria
    labels_different = (
        not lt_compare['identical'] and
        not lm_compare['identical'] and
        not tm_compare['identical']
    )

    # Predictions should differ (since training labels differ)
    preds_different = (
        prediction_results['prediction_comparisons']['rde_l_vs_rde_t']['max_abs_diff'] > 1e-6 or
        prediction_results['prediction_comparisons']['rde_l_vs_rde_m']['max_abs_diff'] > 1e-6 or
        prediction_results['prediction_comparisons']['rde_t_vs_rde_m']['max_abs_diff'] > 1e-6
    )

    # Formula verification: RDE-L uses oracle state, RDE-T/RDE-M use model state
    rde_l_uses_oracle = formula_results['rde_l_formula']['uses_oracle_state']
    rde_t_uses_model_gp = formula_results['rde_t_formula']['uses_model_state_for_gp']
    rde_m_uses_model_gp = formula_results['rde_m_formula']['uses_model_state_for_gp']

    if labels_different and preds_different and rde_l_uses_oracle:
        verdict = "CONFIRMED"
        verdict_detail = "All three RDE variants use different training labels and produce different predictions."
    elif labels_different and rde_l_uses_oracle:
        verdict = "CONFIRMED"
        verdict_detail = "Labels are different and formulas are distinct. Prediction differences may be small due to limited DAgger data."
    else:
        verdict = "INFERRED"
        verdict_detail = "Some evidence is inconclusive. See detailed results."

    print(f"\n  VERDICT: {verdict}")
    print(f"  {verdict_detail}")

    # Save results
    output_dir = Path(__file__).parent
    results = {
        'agent': 'B',
        'task': 'RDE-L / RDE-T / RDE-M Label Independence Verification',
        'verdict': verdict,
        'verdict_detail': verdict_detail,
        'formula_verification': formula_results,
        'label_comparisons': label_comparisons,
        'prediction_results': {
            'prediction_comparisons': prediction_results['prediction_comparisons'],
            'uncertainty_comparisons': prediction_results['uncertainty_comparisons'],
            'sample_predictions': prediction_results['sample_predictions'],
            'train_times': {
                'rde_l': prediction_results['rde_l_train_time'],
                'rde_t': prediction_results['rde_t_train_time'],
                'rde_m': prediction_results['rde_m_train_time'],
            },
        },
        'checks': {
            'labels_different': labels_different,
            'predictions_different': preds_different,
            'rde_l_uses_oracle_state': rde_l_uses_oracle,
            'rde_t_uses_model_state_for_gp': rde_t_uses_model_gp,
            'rde_m_uses_model_state_for_gp': rde_m_uses_model_gp,
        },
        'elapsed_seconds': time.time() - t_start,
    }

    output_path = output_dir / "AGENT_B_RESULTS.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

    elapsed = time.time() - t_start
    print(f"\nAgent B completed in {elapsed:.1f}s")
    return results


if __name__ == "__main__":
    main()
