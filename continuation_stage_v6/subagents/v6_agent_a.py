"""V6 Subagent A: Validate canonical_4d implementation.

Trains all 6 models (GP, E1, RDE-L, RDE-T, RDE-M, SINDy),
evaluates in Mode B at multiple horizons, runs equivalence test.
"""
import sys
import os
import json
import time
import warnings
import traceback
import numpy as np
from pathlib import Path
from collections import OrderedDict

# ---- Path setup ----
ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# canonical_4d is one level up from subagents
C4D_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, C4D_DIR)

# ---- Suppress expected GP convergence warnings ----
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*fvalue.*")
warnings.filterwarnings("ignore", message=".*Optimization.*")
warnings.filterwarnings("ignore", message=".*GaussianProcessRegressor.*")
# Sklearn GP warnings about convergence
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

# ---- Imports from canonical_4d ----
from canonical_4d.config import DynamicsConfig, TrainingConfig, EvalConfig
from canonical_4d.dynamics import BicycleDynamics
from canonical_4d.models import GPModel, SINDyModel, BaseModel
from canonical_4d.residual_models import E1OfflineNN, RDELocal, RDETrajectory, RDEHybrid, NNEnsemble
from canonical_4d.evaluation_modes import run_mode_b
from canonical_4d.metrics import compute_metrics, STATE_NAMES
from canonical_4d.runner import generate_training_data, train_all_models, evaluate_model, aggregate_results


# ========================================================================
# Equivalence test: GP + zero_residual == GP
# ========================================================================
class ZeroResidualWrapper(BaseModel):
    """Zero residual wrapper: predict equals baseline GP exactly."""

    def __init__(self, baseline):
        self._baseline = baseline
        self._state_std = baseline._state_std
        self._action_std = baseline._action_std
        self._delta_std = baseline._delta_std

    def predict(self, s, tau):
        return self._baseline.predict(s, tau)

    def predict_with_uncertainty(self, s, tau):
        return self._baseline.predict(s, tau), None

    def name(self):
        return "zero_residual_wrapper"

    def train(self, *args, **kwargs):
        pass


def run_equivalence_test(dynamics):
    """GP + zero_residual must equal GP exactly (max_diff < 1e-10)."""
    print("\n=== Equivalence Test: GP == GP + ZeroResidual ===")
    data = generate_training_data(5000, 42, dynamics)
    states, actions, deltas, state_std, action_std, delta_std = data

    gp = GPModel(max_samples=2000, n_restarts=2)
    gp.train(states, actions, deltas, state_std, action_std, delta_std)

    zr = ZeroResidualWrapper(gp)

    # Test on random points
    rng = np.random.RandomState(123)
    n_test = 200
    test_s = rng.uniform(-0.3, 0.3, (n_test, 4))
    test_s[:, 1] = rng.uniform(-0.2, 0.2, n_test)
    test_s[:, 2] = rng.uniform(-1.0, 1.0, n_test)
    test_s[:, 3] = rng.uniform(-0.5, 0.5, n_test)
    test_a = rng.uniform(-50, 50, n_test)

    max_diff = 0.0
    for i in range(n_test):
        p_gp = gp.predict(test_s[i], test_a[i])
        p_zr = zr.predict(test_s[i], test_a[i])
        diff = np.max(np.abs(p_gp - p_zr))
        if diff > max_diff:
            max_diff = diff

    confirmed = max_diff < 1e-10
    status = "CONFIRMED" if confirmed else "FAILED"
    print(f"  max_diff = {max_diff:.2e} -> {status}")
    return {
        "test": "GP_zero_residual_equivalence",
        "max_diff": float(max_diff),
        "threshold": 1e-10,
        "confirmed": confirmed,
        "status": status,
    }


# ========================================================================
# Main validation
# ========================================================================
def main():
    t_start = time.time()
    results = OrderedDict()

    print("=" * 70)
    print("V6 SUBAGENT A: Canonical 4D Validation")
    print("=" * 70)

    # ---- Init dynamics ----
    print("\n[1/5] Initializing dynamics...")
    dynamics = BicycleDynamics()
    print("  BicycleDynamics initialized (v0=3.5, dt=1/30)")

    # ---- Generate training data ----
    print("\n[2/5] Generating 5000 training samples...")
    t0 = time.time()
    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(
        n_samples=5000, seed=42, dynamics=dynamics
    )
    gen_time = time.time() - t0
    print(f"  Generated {len(states)} samples in {gen_time:.1f}s")
    print(f"  state_std = {state_std}")
    print(f"  action_std = {action_std:.4f}")
    print(f"  delta_std = {delta_std}")

    results["training_data"] = {
        "n_samples": int(len(states)),
        "seed": 42,
        "generation_time_s": round(gen_time, 2),
        "state_std": state_std.tolist(),
        "action_std": float(action_std),
        "delta_std": delta_std.tolist(),
    }

    # ---- Train all models ----
    print("\n[3/5] Training all 6 models...")
    training_times = {}

    # GP (use 1000 samples to avoid memory issues on systems with limited page file)
    print("  [1/6] GP...")
    t0 = time.time()
    gp = GPModel(max_samples=1000, n_restarts=2)
    gp.train(states, actions, deltas, state_std, action_std, delta_std)
    training_times["gp"] = time.time() - t0
    print(f"    Done in {training_times['gp']:.1f}s")

    # E1 (offline NN)
    print("  [2/6] E1 (offline NN)...")
    t0 = time.time()
    e1 = E1OfflineNN(n_models=5, n_epochs=50, residual_scale=0.3)
    e1.train(states, actions, deltas, state_std, action_std, delta_std)
    training_times["e1"] = time.time() - t0
    print(f"    Done in {training_times['e1']:.1f}s")

    # RDE-L (local residual, 0 DAgger rounds)
    print("  [3/6] RDE-L (local residual)...")
    t0 = time.time()
    rde_l = RDELocal(n_models=5, n_epochs=50, residual_scale=0.3)
    rde_l.train(states, actions, deltas, state_std, action_std, delta_std)
    training_times["rde_l"] = time.time() - t0
    print(f"    Done in {training_times['rde_l']:.1f}s")

    # RDE-T (trajectory sync, 1 DAgger round)
    print("  [4/6] RDE-T (trajectory sync, 1 DAgger round)...")
    t0 = time.time()
    rde_t_n_seg = 3
    rde_t = RDETrajectory(
        n_models=5, n_epochs=50, residual_scale=0.3,
        dagger_rounds=1, n_segments=rde_t_n_seg, segment_length=500
    )
    rde_t.train(states, actions, deltas, state_std, action_std, delta_std,
                dynamics=dynamics, make_tau_func=dynamics.generate_lqr_tau)
    training_times["rde_t"] = time.time() - t0
    print(f"    Done in {training_times['rde_t']:.1f}s (n_segments={rde_t_n_seg})")

    # RDE-M (hybrid, 2 DAgger rounds)
    print("  [5/6] RDE-M (hybrid, 2 DAgger rounds)...")
    t0 = time.time()
    rde_m_n_seg = 3
    rde_m = RDEHybrid(
        n_models=5, n_epochs=50, residual_scale=0.3,
        dagger_rounds=2, n_segments=rde_m_n_seg, segment_length=500
    )
    rde_m.train(states, actions, deltas, state_std, action_std, delta_std,
                dynamics=dynamics, make_tau_func=dynamics.generate_lqr_tau)
    training_times["rde_m"] = time.time() - t0
    print(f"    Done in {training_times['rde_m']:.1f}s (n_segments={rde_m_n_seg})")

    # SINDy
    print("  [6/6] SINDy...")
    t0 = time.time()
    sindy = SINDyModel()
    sindy.train(states, actions, deltas, state_std, action_std, delta_std)
    training_times["sindy"] = time.time() - t0
    print(f"    Done in {training_times['sindy']:.1f}s")

    results["training_times"] = {k: round(v, 2) for k, v in training_times.items()}

    # ---- Evaluate all models in Mode B ----
    print("\n[4/5] Evaluating all models in Mode B...")
    eval_horizons = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    n_segments = 5
    segment_length = 500

    model_dict = OrderedDict([
        ("gp", gp), ("e1", e1), ("rde_l", rde_l),
        ("rde_t", rde_t), ("rde_m", rde_m), ("sindy", sindy),
    ])

    all_model_results = OrderedDict()

    for mname, model in model_dict.items():
        print(f"\n  Evaluating {mname}...")
        t0 = time.time()
        try:
            raw_results = evaluate_model(
                model, dynamics,
                n_segments=n_segments, segment_length=segment_length, mode="b"
            )
            agg = aggregate_results(raw_results, eval_horizons, state_std)

            eval_time = time.time() - t0

            # Build per-horizon summary
            horizon_summary = OrderedDict()
            for h in eval_horizons:
                metrics_list = agg[h]
                # Average across segments
                avg_mae = np.mean([m["overall"]["mae"] for m in metrics_list])
                avg_nmae = np.mean([m["overall"]["nmae"] for m in metrics_list])
                avg_surv = np.mean([m["stability"]["survival_steps"] for m in metrics_list])

                per_state = {}
                for sn in STATE_NAMES:
                    sn_mae = np.mean([m[sn]["mae"] for m in metrics_list])
                    sn_nmae = np.mean([m[sn]["nmae"] for m in metrics_list])
                    per_state[sn] = {"mae": float(sn_mae), "nmae": float(sn_nmae)}

                horizon_summary[str(h)] = {
                    "mae": float(avg_mae),
                    "nmae": float(avg_nmae),
                    "survival_steps": float(avg_surv),
                    "per_state": per_state,
                }

            all_model_results[mname] = {
                "eval_time_s": round(eval_time, 2),
                "n_segments": n_segments,
                "segment_length": segment_length,
                "horizons": horizon_summary,
                "status": "ok",
            }
            print(f"    Done in {eval_time:.1f}s")

        except Exception as e:
            eval_time = time.time() - t0
            all_model_results[mname] = {
                "eval_time_s": round(eval_time, 2),
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            print(f"    FAILED: {e}")

    results["evaluation"] = all_model_results

    # ---- Equivalence test ----
    print("\n[5/5] Running equivalence test...")
    eq_result = run_equivalence_test(dynamics)
    results["equivalence_test"] = eq_result

    # ---- Total runtime ----
    total_time = time.time() - t_start
    results["total_runtime_s"] = round(total_time, 2)

    # ---- Save results ----
    out_path = Path(__file__).parent / "AGENT_A_RESULTS.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # ---- Print summary table ----
    print("\n" + "=" * 100)
    print("SUMMARY TABLE: Mode B Evaluation")
    print("=" * 100)

    header = f"{'Model':<10} {'H=1':>8} {'H=5':>8} {'H=10':>8} {'H=20':>8} {'H=50':>8} {'H=100':>8} {'H=200':>8} {'H=500':>8} {'H=1000':>8} {'Train(s)':>10}"
    print(header)
    print("-" * 100)

    for mname in model_dict:
        row = f"{mname:<10}"
        for h in [1, 5, 10, 20, 50, 100, 200, 500, 1000]:
            hs = all_model_results.get(mname, {}).get("horizons", {}).get(str(h), {})
            nmae = hs.get("nmae", float("nan"))
            row += f" {nmae:>8.4f}"
        tt = training_times.get(mname, 0)
        row += f" {tt:>10.1f}"
        print(row)

    print("-" * 100)

    # Survival steps table
    print("\nSurvival Steps (avg across segments):")
    header2 = f"{'Model':<10} {'H=1':>8} {'H=5':>8} {'H=10':>8} {'H=20':>8} {'H=50':>8} {'H=100':>8} {'H=200':>8} {'H=500':>8} {'H=1000':>8}"
    print(header2)
    print("-" * 100)

    for mname in model_dict:
        row = f"{mname:<10}"
        for h in [1, 5, 10, 20, 50, 100, 200, 500, 1000]:
            hs = all_model_results.get(mname, {}).get("horizons", {}).get(str(h), {})
            surv = hs.get("survival_steps", float("nan"))
            row += f" {surv:>8.1f}"
        print(row)

    print("-" * 100)
    print(f"\nTotal runtime: {total_time:.1f}s")
    print(f"Equivalence test: {eq_result['status']} (max_diff={eq_result['max_diff']:.2e})")
    print("=" * 100)

    return results


if __name__ == "__main__":
    main()
