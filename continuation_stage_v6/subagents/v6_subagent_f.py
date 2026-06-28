"""V6 Subagent F: Planning Consistency & Closed-Loop Validation.

F1: Candidate action sequences
F2: True vs model cumulative cost
F3: Ranking ability (Spearman, Kendall, Top-k)
F4: Multi-horizon evaluation
F5: Closed-loop Mode D
"""
import sys, os, json, time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

sys.path.insert(0, str(Path(__file__).parent.parent))
from canonical_4d.config import TrainingConfig, PlanConfig
from canonical_4d.dynamics import BicycleDynamics
from canonical_4d.models import GPModel
from canonical_4d.residual_models import E1OfflineNN, RDEHybrid
from canonical_4d.evaluation_modes import run_mode_b, run_mode_d
from canonical_4d.metrics import compute_cost, compute_metrics
from canonical_4d.runner import generate_training_data

OUTPUT = Path(__file__).parent.parent / "raw_results"
OUTPUT.mkdir(exist_ok=True)

PLAN_CFG = PlanConfig()


def generate_candidate_sequences(n_sequences, horizon, rng):
    """Generate diverse candidate action sequences."""
    sequences = []

    # Small perturbations
    for _ in range(n_sequences // 5):
        base = rng.uniform(-5, 5, horizon)
        seq = base + rng.normal(0, 1, horizon)
        sequences.append(np.clip(seq, -50, 50))

    # Medium perturbations
    for _ in range(n_sequences // 5):
        base = rng.uniform(-20, 20, horizon)
        seq = base + rng.normal(0, 5, horizon)
        sequences.append(np.clip(seq, -50, 50))

    # Large perturbations
    for _ in range(n_sequences // 5):
        seq = rng.uniform(-50, 50, horizon)
        sequences.append(seq)

    # Smooth (low frequency)
    for _ in range(n_sequences // 5):
        t = np.linspace(0, 2 * np.pi, horizon)
        freq = rng.uniform(0.5, 2.0)
        seq = 20 * np.sin(freq * t + rng.uniform(0, 2*np.pi))
        sequences.append(seq)

    # High frequency
    for _ in range(n_sequences // 5):
        t = np.linspace(0, 10 * np.pi, horizon)
        seq = 15 * np.sin(t) + rng.normal(0, 3, horizon)
        sequences.append(np.clip(seq, -50, 50))

    return sequences


def evaluate_sequence(model, s0, actions, dynamics):
    """Evaluate a fixed action sequence with a model."""
    s_model = s0.copy()
    states = [s0.copy()]
    for tau in actions:
        s_next = model.predict(s_model, tau)
        states.append(s_next.copy())
        s_model = s_next
        if dynamics.is_diverged(s_model):
            break
    return np.array(states)


def evaluate_sequence_real(dynamics, s0, actions):
    """Evaluate a fixed action sequence with real dynamics."""
    s_real = s0.copy()
    states = [s0.copy()]
    for tau in actions:
        s_next = dynamics.step(s_real, tau)
        states.append(s_next.copy())
        s_real = s_next
        if dynamics.is_diverged(s_real):
            break
    return np.array(states)


def main():
    print("="*80)
    print("V6 SUBAGENT F: Planning Consistency & Closed-Loop Validation")
    print("="*80)
    t0 = time.time()

    dyn = BicycleDynamics()
    plan_cfg = PLAN_CFG

    # Train models
    print("--- Training models ---")
    data = generate_training_data(5000, 42, dyn)
    states, actions, deltas, state_std, action_std, delta_std = data

    gp = GPModel(max_samples=2000, n_restarts=2)
    gp.train(states, actions, deltas, state_std, action_std, delta_std)

    e1 = E1OfflineNN(n_models=5, n_epochs=50, residual_scale=0.3)
    e1.train(states, actions, deltas, state_std, action_std, delta_std)

    rde_m = RDEHybrid(n_models=5, n_epochs=50, residual_scale=0.3,
                       dagger_rounds=2, n_segments=3, segment_length=500)
    rde_m.train(states, actions, deltas, state_std, action_std, delta_std,
                dynamics=dyn, make_tau_func=dyn.generate_lqr_tau)

    models = {'gp': gp, 'e1': e1, 'rde_m': rde_m}

    # F1-F4: Planning consistency
    print("\n--- F1-F4: Planning Consistency ---")
    planning_results = {}

    for horizon in plan_cfg.plan_horizons:
        print(f"\n  Horizon: {horizon} steps")
        rng = np.random.RandomState(42)
        s0 = np.array([0.1, 0.0, 0.0, 0.0])
        candidate_seqs = generate_candidate_sequences(plan_cfg.n_candidate_sequences, horizon, rng)

        # Evaluate with real dynamics
        real_costs = []
        for seq in candidate_seqs:
            real_traj = evaluate_sequence_real(dyn, s0, seq)
            J_real = compute_cost(real_traj, seq, plan_cfg.cost_weights)
            real_costs.append(J_real)
        real_costs = np.array(real_costs)

        model_results = {}
        for model_name, model in models.items():
            pred_costs = []
            for seq in candidate_seqs:
                model_traj = evaluate_sequence(model, s0, seq, dyn)
                J_pred = compute_cost(model_traj, seq, plan_cfg.cost_weights)
                pred_costs.append(J_pred)
            pred_costs = np.array(pred_costs)

            # Cost errors
            abs_cost_err = np.abs(pred_costs - real_costs)
            rel_cost_err = abs_cost_err / (np.abs(real_costs) + 1e-10)

            # Ranking
            real_rank = np.argsort(real_costs)
            pred_rank = np.argsort(pred_costs)

            spearman_corr, _ = spearmanr(real_costs, pred_costs)
            kendall_corr, _ = kendalltau(real_costs, pred_costs)

            # Top-k hits
            top1_real = real_rank[0]
            top1_hit = int(pred_rank[0] == top1_real)
            top3_real = set(real_rank[:3])
            top3_hit = sum(1 for r in pred_rank[:3] if r in top3_real)
            top10pct = max(1, len(candidate_seqs) // 10)
            top10_real = set(real_rank[:top10pct])
            top10_hit = sum(1 for r in pred_rank[:top10pct] if r in top10_real) / top10pct

            # Divergence detection
            real_div = [i for i, seq in enumerate(candidate_seqs)
                       if len(evaluate_sequence_real(dyn, s0, seq)) < horizon + 1]
            model_div = [i for i, seq in enumerate(candidate_seqs)
                        if len(evaluate_sequence(model, s0, seq, dyn)) < horizon + 1]
            div_accuracy = len(set(real_div) & set(model_div)) / max(len(real_div), 1) if real_div else 0.0

            mr = {
                'abs_cost_error_mean': float(np.mean(abs_cost_err)),
                'abs_cost_error_std': float(np.std(abs_cost_err)),
                'rel_cost_error_mean': float(np.mean(rel_cost_err)),
                'spearman_corr': float(spearman_corr),
                'kendall_corr': float(kendall_corr),
                'top1_hit': top1_hit,
                'top3_hit': top3_hit / 3,
                'top10pct_recall': top10_hit,
                'div_detection_accuracy': div_accuracy,
                'n_real_divergent': len(real_div),
                'n_model_divergent': len(model_div),
            }
            model_results[model_name] = mr
            print(f"    {model_name:8s}: Spearman={spearman_corr:.3f}, Top1={top1_hit}, "
                  f"CostErr={np.mean(abs_cost_err):.2f}")

        planning_results[str(horizon)] = model_results

    # F5: Closed-loop Mode D
    print("\n--- F5: Closed-loop Mode D ---")
    closed_loop = {}
    for model_name, model in models.items():
        rng_d = np.random.RandomState(42)
        result_d = run_mode_d(model, np.array([0.1, 0.0, 0.0, 0.0]),
                              200, dyn, dyn.K_lqr, rng_d)
        n = min(201, len(result_d.states_model))
        errors = np.abs(result_d.states_model[:n] - result_d.states_real[:n])
        mae = float(np.mean(errors))
        survival = result_d.survival_steps
        closed_loop[model_name] = {
            'mae_200step': mae,
            'survival_steps': survival,
            'failure_before_horizon': result_d.failure_before_horizon,
        }
        print(f"  {model_name}: MAE={mae:.6f}, survival={survival}")

    # Save
    output = {
        'subagent': 'F',
        'planning_results': planning_results,
        'closed_loop': closed_loop,
        'cost_weights': plan_cfg.cost_weights,
        'candidate_actions': f'{plan_cfg.n_candidate_sequences} sequences per horizon',
        'plan_horizons': plan_cfg.plan_horizons,
    }

    with open(OUTPUT / "SUBAGENT_F_RESULTS.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nSubagent F done in {elapsed:.1f}s")
    return output


if __name__ == "__main__":
    main()
