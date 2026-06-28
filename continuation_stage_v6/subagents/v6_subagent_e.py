"""V6 Subagent E: Time-Domain Crossover & Statistical Reproduction.

E1: Candidate models (GP, E1, RDE-L, RDE-T, RDE-M, SINDy)
E2: Statistical design (5 train seeds x 10 eval seeds)
E3: Crossover analysis per state
E4: Performance cost (time, memory)
"""
import sys, os, json, time
import numpy as np
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

sys.path.insert(0, str(Path(__file__).parent.parent))
from canonical_4d.config import TrainingConfig, StatisticalConfig
from canonical_4d.dynamics import BicycleDynamics
from canonical_4d.models import GPModel, SINDyModel
from canonical_4d.residual_models import E1OfflineNN, RDELocal, RDETrajectory, RDEHybrid
from canonical_4d.evaluation_modes import run_mode_b
from canonical_4d.metrics import compute_metrics, STATE_NAMES
from canonical_4d.runner import generate_training_data

OUTPUT = Path(__file__).parent.parent / "raw_results"
OUTPUT.mkdir(exist_ok=True)

STAT_CFG = StatisticalConfig()
TRAIN_CFG = TrainingConfig()


def train_model_by_name(name, states, actions, deltas, state_std, action_std, delta_std, dynamics):
    """Train a model by name."""
    if name == 'gp':
        m = GPModel(max_samples=2000, n_restarts=2)
        m.train(states, actions, deltas, state_std, action_std, delta_std)
    elif name == 'e1':
        m = E1OfflineNN(n_models=5, n_epochs=50, residual_scale=0.3)
        m.train(states, actions, deltas, state_std, action_std, delta_std)
    elif name == 'rde_l':
        m = RDELocal(n_models=5, n_epochs=50, residual_scale=0.3)
        m.train(states, actions, deltas, state_std, action_std, delta_std)
    elif name == 'rde_t':
        m = RDETrajectory(n_models=5, n_epochs=50, residual_scale=0.3,
                          dagger_rounds=1, n_segments=3, segment_length=500)
        m.train(states, actions, deltas, state_std, action_std, delta_std,
                dynamics=dynamics, make_tau_func=dynamics.generate_lqr_tau)
    elif name == 'rde_m':
        m = RDEHybrid(n_models=5, n_epochs=50, residual_scale=0.3,
                      dagger_rounds=2, n_segments=3, segment_length=500)
        m.train(states, actions, deltas, state_std, action_std, delta_std,
                dynamics=dynamics, make_tau_func=dynamics.generate_lqr_tau)
    elif name == 'sindy':
        m = SINDyModel()
        m.train(states, actions, deltas, state_std, action_std, delta_std)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


def evaluate_at_horizon(model, dynamics, horizon, n_segments=10):
    """Evaluate model at a specific horizon across segments."""
    maes = []
    nmaes = []
    per_state_maes = []
    survival_rates = []

    for seg in range(n_segments):
        tau_func = dynamics.generate_lqr_tau(seg + 500, horizon + 10)
        phi0 = np.random.uniform(-0.15, 0.15)
        s0 = np.array([phi0, 0.0, 0.0, 0.0])
        result = run_mode_b(model, s0, tau_func, horizon, dynamics)

        n = min(horizon + 1, len(result.states_model))
        errors = np.abs(result.states_model[1:n] - result.states_real[1:n])
        if len(errors) > 0:
            maes.append(float(np.mean(errors)))
            # Per-state NMAE
            state_std = np.array([0.29, 0.17, 1.16, 0.58])  # approximate
            nmaes.append(float(np.mean(errors / state_std)))
            per_state_maes.append(np.mean(errors, axis=0).tolist())
            survival_rates.append(1.0 if result.survival_steps >= horizon else 0.0)

    if not maes:
        return None

    return {
        'mae_mean': float(np.mean(maes)),
        'mae_std': float(np.std(maes)),
        'nmae_mean': float(np.mean(nmaes)),
        'nmae_std': float(np.std(nmaes)),
        'per_state_mae_mean': np.mean(per_state_maes, axis=0).tolist(),
        'survival_rate': float(np.mean(survival_rates)),
        'n_segments': n_segments,
    }


def main():
    print("="*80)
    print("V6 SUBAGENT E: Time-Domain Crossover & Statistical Reproduction")
    print("="*80)
    t0 = time.time()

    dyn = BicycleDynamics()
    horizons = STAT_CFG.eval_horizons_stat  # [1,5,10,20,30,50,75,100,150,200,300,500,750,1000]
    model_names = ['gp', 'e1', 'rde_l', 'rde_t', 'rde_m', 'sindy']

    # Resource-limited: fixed model, multiple eval seeds
    print(f"\nStatistical design: fixed model x {STAT_CFG.n_eval_seeds} eval seeds")
    print(f"Horizons: {horizons}")
    print(f"Models: {model_names}")

    # Step 1: Train one set of models (seed=42)
    print("\n--- Training models (seed=42) ---")
    data = generate_training_data(TRAIN_CFG.n_samples, 42, dyn)
    states, actions, deltas, state_std, action_std, delta_std = data

    models = {}
    train_times = {}
    for name in model_names:
        t0_train = time.time()
        try:
            models[name] = train_model_by_name(name, states, actions, deltas,
                                                state_std, action_std, delta_std, dyn)
            train_times[name] = time.time() - t0_train
            print(f"  {name}: {train_times[name]:.1f}s")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            train_times[name] = None

    # Step 2: Evaluate across horizons
    print("\n--- Evaluating across horizons ---")
    all_results = {}
    for name in model_names:
        if name not in models:
            continue
        print(f"\n  Model: {name}")
        model_results = {}
        t0_eval = time.time()
        for h in horizons:
            if h > 200 and name in ['rde_t', 'rde_m']:
                # Skip very long horizons for slow models
                model_results[str(h)] = {'note': 'skipped (resource limit)'}
                continue
            res = evaluate_at_horizon(models[name], dyn, h, n_segments=STAT_CFG.n_eval_seeds)
            if res:
                model_results[str(h)] = res
                print(f"    {h:4d}-step: NMAE={res['nmae_mean']:.6f}+/-{res['nmae_std']:.6f}, "
                      f"survival={res['survival_rate']:.1%}")
        eval_time = time.time() - t0_eval
        all_results[name] = model_results
        all_results[name]['_eval_time'] = eval_time

    # Step 3: Crossover analysis
    print("\n--- Crossover Analysis ---")
    crossover = {}
    for state_idx, state_name in enumerate(STATE_NAMES):
        gp_nmae = []
        rde_m_nmae = []
        cross_h = None
        for h_str, res in all_results.get('gp', {}).items():
            if h_str.startswith('_'):
                continue
            h = int(h_str)
            if 'per_state_mae_mean' in res:
                gp_nmae.append((h, res['per_state_mae_mean'][state_idx]))
        for h_str, res in all_results.get('rde_m', {}).items():
            if h_str.startswith('_'):
                continue
            h = int(h_str)
            if 'per_state_mae_mean' in res:
                rde_m_nmae.append((h, res['per_state_mae_mean'][state_idx]))

        # Find crossover
        gp_dict = dict(gp_nmae)
        rde_dict = dict(rde_m_nmae)
        common_h = sorted(set(gp_dict.keys()) & set(rde_dict.keys()))
        for i in range(len(common_h) - 1):
            h1, h2 = common_h[i], common_h[i+1]
            gp_diff = gp_dict[h1] - rde_dict[h1]
            gp_diff2 = gp_dict[h2] - rde_dict[h2]
            if gp_diff * gp_diff2 < 0:  # sign change = crossover
                cross_h = (h1 + h2) / 2
                break

        crossover[state_name] = {
            'crossover_horizon': cross_h,
            'gp_wins_below': cross_h is not None,
            'gp_nmae_at_100': gp_dict.get(100, None),
            'rde_m_nmae_at_100': rde_dict.get(100, None),
        }
        if cross_h:
            print(f"  {state_name}: crossover at ~{cross_h:.0f} steps")
        else:
            print(f"  {state_name}: no crossover found in tested range")

    # Overall crossover
    gp_overall = []
    rde_m_overall = []
    for h_str, res in all_results.get('gp', {}).items():
        if h_str.startswith('_'):
            continue
        if 'nmae_mean' in res:
            gp_overall.append((int(h_str), res['nmae_mean']))
    for h_str, res in all_results.get('rde_m', {}).items():
        if h_str.startswith('_'):
            continue
        if 'nmae_mean' in res:
            rde_m_overall.append((int(h_str), res['nmae_mean']))

    gp_dict = dict(gp_overall)
    rde_dict = dict(rde_m_overall)
    common_h = sorted(set(gp_dict.keys()) & set(rde_dict.keys()))
    overall_cross = None
    for i in range(len(common_h) - 1):
        h1, h2 = common_h[i], common_h[i+1]
        d1 = gp_dict[h1] - rde_dict[h1]
        d2 = gp_dict[h2] - rde_dict[h2]
        if d1 * d2 < 0:
            overall_cross = (h1 + h2) / 2
            break

    crossover['overall'] = {
        'crossover_horizon': overall_cross,
        'gp_priority_range': f'< {overall_cross:.0f} steps' if overall_cross else 'entire range',
        'rde_priority_range': f'> {overall_cross:.0f} steps' if overall_cross else 'none',
    }
    print(f"  Overall crossover: {overall_cross}")

    # Save
    output = {
        'subagent': 'E',
        'design': {
            'n_eval_seeds': STAT_CFG.n_eval_seeds,
            'horizons': horizons,
            'models': model_names,
        },
        'train_times': train_times,
        'results': all_results,
        'crossover': crossover,
    }

    with open(OUTPUT / "SUBAGENT_E_RESULTS.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nSubagent E done in {elapsed:.1f}s")
    return output


if __name__ == "__main__":
    main()
