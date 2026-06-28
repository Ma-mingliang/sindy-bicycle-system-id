"""Agent E: Multi-step rollout evaluation for V8 7D world model.

Evaluates GP, E1, RDE-L at horizons [1, 5, 10, 20, 50, 100, 200]
using Mode B (open-loop) on 5 test segments of length 200.
Compares with V6 4D results from FINAL_EXECUTION_OUTPUT_V6.md.
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
sys.path.insert(0, r'D:\系统辨识作业\sindy_bicycle\continuation_stage_v8')

import numpy as np
import json
import time
from canonical_7d import (
    load_7d_data, DataConfig, get_test_segments,
    GPModel, E1OfflineNN, RDELocal,
    evaluate_model_on_segments, aggregate_results,
    compute_metrics, STATE_NAMES_7D, STATE_DIM
)
from canonical_7d.residual_models import NNEnsemble7D
from canonical_7d.models import BaseModel


# Speed-optimized E1 that reuses a pre-trained GP
class FastE1OfflineNN(BaseModel):
    """E1: GP + offline NN residual, sharing pre-trained GP baseline."""

    def __init__(self, baseline_gp, n_models=3, n_epochs=30, residual_scale=0.3):
        self._baseline = baseline_gp
        self.n_models = n_models
        self.n_epochs = n_epochs
        self.residual_scale = residual_scale
        self._ensemble = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std,
              nn_max_samples=10000):
        self._state_std = state_std.copy()
        self._action_std = action_std
        self._delta_std = delta_std.copy()
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0

        # Subsample for residual computation (GP predict is expensive per-sample)
        n = len(states)
        rng = np.random.RandomState(42)
        if n > nn_max_samples:
            idx = rng.choice(n, nn_max_samples, replace=False)
        else:
            idx = np.arange(n)

        sub_states = states[idx]
        sub_actions = actions[idx]
        sub_deltas = deltas[idx]

        residuals = np.empty_like(sub_deltas)
        for i in range(len(sub_states)):
            s_next_base = self._baseline.predict(sub_states[i], sub_actions[i])
            residuals[i] = (sub_deltas[i] - (s_next_base - sub_states[i])) / self._delta_std

        train_inputs = np.column_stack([
            sub_states / self._state_std,
            sub_actions.reshape(-1, 1) / self._action_std
        ])
        self._ensemble = NNEnsemble7D(self.n_models, self.n_epochs)
        self._ensemble.train(train_inputs, residuals)

    def predict(self, s, tau):
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        delta_nn = self._ensemble.predict_mean(s_norm, a_norm)
        return s_next_base + delta_nn * self._delta_std * self.residual_scale

    def predict_with_uncertainty(self, s, tau):
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        mean_nn, std_nn = self._ensemble.predict_mean_std(s_norm, a_norm)
        s_next = s_next_base + mean_nn * self._delta_std * self.residual_scale
        uncertainty = std_nn * self._delta_std * self.residual_scale
        return s_next, uncertainty

    def name(self):
        return 'fast_e1_offline_nn'


# Speed-optimized RDE-L that reuses a pre-trained GP
class FastRDELocal(BaseModel):
    """RDE-L: Local dynamics residual, sharing pre-trained GP baseline."""

    def __init__(self, baseline_gp, n_models=3, n_epochs=30, residual_scale=0.3):
        self._baseline = baseline_gp
        self.n_models = n_models
        self.n_epochs = n_epochs
        self.residual_scale = residual_scale
        self._ensemble = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std,
              nn_max_samples=10000):
        self._state_std = state_std.copy()
        self._action_std = action_std
        self._delta_std = delta_std.copy()
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0

        # Subsample for residual computation
        n = len(states)
        rng = np.random.RandomState(42)
        if n > nn_max_samples:
            idx = rng.choice(n, nn_max_samples, replace=False)
        else:
            idx = np.arange(n)

        sub_states = states[idx]
        sub_actions = actions[idx]
        sub_deltas = deltas[idx]

        residuals = np.empty_like(sub_deltas)
        for i in range(len(sub_states)):
            s_next_base = self._baseline.predict(sub_states[i], sub_actions[i])
            residuals[i] = (sub_deltas[i] - (s_next_base - sub_states[i])) / self._delta_std

        train_inputs = np.column_stack([
            sub_states / self._state_std,
            sub_actions.reshape(-1, 1) / self._action_std
        ])
        self._ensemble = NNEnsemble7D(self.n_models, self.n_epochs)
        self._ensemble.train(train_inputs, residuals)

    def predict(self, s, tau):
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        delta_nn = self._ensemble.predict_mean(s_norm, a_norm)
        return s_next_base + delta_nn * self._delta_std * self.residual_scale

    def predict_with_uncertainty(self, s, tau):
        s_next_base = self._baseline.predict(s, tau)
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        mean_nn, std_nn = self._ensemble.predict_mean_std(s_norm, a_norm)
        s_next = s_next_base + mean_nn * self._delta_std * self.residual_scale
        uncertainty = std_nn * self._delta_std * self.residual_scale
        return s_next, uncertainty

    def name(self):
        return 'fast_rde_local'


def train_models(data):
    """Train GP, E1, RDE-L models on 7D data (optimized: shared GP baseline)."""
    states = data['train_states']
    actions = data['train_actions']
    deltas = data['train_deltas']
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    models = {}

    # GP (trained once, shared by E1 and RDE-L)
    t0 = time.time()
    gp = GPModel(max_samples=2000, n_restarts=2)
    gp.train(states, actions, deltas, state_std, action_std, delta_std)
    models['gp'] = gp
    gp_time = time.time() - t0
    print(f"  GP trained in {gp_time:.1f}s")

    # E1 (offline NN) - reuses GP baseline
    t0 = time.time()
    e1 = FastE1OfflineNN(gp, n_models=3, n_epochs=20, residual_scale=0.3)
    e1.train(states, actions, deltas, state_std, action_std, delta_std, nn_max_samples=8000)
    models['e1'] = e1
    print(f"  E1 trained in {time.time()-t0:.1f}s (NN only, GP shared)")

    # RDE-L - reuses GP baseline
    t0 = time.time()
    rde_l = FastRDELocal(gp, n_models=3, n_epochs=20, residual_scale=0.3)
    rde_l.train(states, actions, deltas, state_std, action_std, delta_std, nn_max_samples=8000)
    models['rde_l'] = rde_l
    print(f"  RDE-L trained in {time.time()-t0:.1f}s (NN only, GP shared)")

    return models


def aggregate_across_segments(metrics_list, state_std):
    """Aggregate per-segment metrics into mean/std across segments."""
    n = len(metrics_list)
    if n == 0:
        return {}

    # Collect overall metrics
    overall_keys = ['nmae', 'mae', 'rmse', 'nrmse', 'endpoint_error']
    overall_agg = {}
    for key in overall_keys:
        vals = []
        for m in metrics_list:
            if 'overall' in m and key in m['overall']:
                v = m['overall'][key]
                if not np.isnan(v):
                    vals.append(v)
        if vals:
            overall_agg[key] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'n_valid': len(vals),
            }
        else:
            overall_agg[key] = {'mean': float('nan'), 'std': float('nan'),
                                'min': float('nan'), 'max': float('nan'), 'n_valid': 0}

    # Collect per-state NMAE
    state_nmae = {}
    for name in STATE_NAMES_7D:
        vals = []
        for m in metrics_list:
            if name in m and 'nmae' in m[name]:
                v = m[name]['nmae']
                if not np.isnan(v):
                    vals.append(v)
        if vals:
            state_nmae[name] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
            }
        else:
            state_nmae[name] = {'mean': float('nan'), 'std': float('nan')}

    # Collect per-state MAE
    state_mae = {}
    for name in STATE_NAMES_7D:
        vals = []
        for m in metrics_list:
            if name in m and 'mae' in m[name]:
                v = m[name]['mae']
                if not np.isnan(v):
                    vals.append(v)
        if vals:
            state_mae[name] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
            }
        else:
            state_mae[name] = {'mean': float('nan'), 'std': float('nan')}

    # Collect survival info
    failure_counts = []
    for m in metrics_list:
        if 'stability' in m:
            failure_counts.append(1 if m['stability'].get('failure_before_horizon', False) else 0)

    survival_rate = 1.0 - (np.mean(failure_counts) if failure_counts else 0.0)

    return {
        'overall': overall_agg,
        'per_state_nmae': state_nmae,
        'per_state_mae': state_mae,
        'survival_rate': float(survival_rate),
        'n_segments': n,
    }


def main():
    print("=" * 70)
    print("AGENT E: Multi-step Rollout Evaluation (V8 7D)")
    print("=" * 70)
    sys.stdout.flush()

    # 1. Load data
    print("\n[1/5] Loading 7D data...")
    sys.stdout.flush()
    t0 = time.time()
    cfg = DataConfig(data_path='D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz')
    data = load_7d_data(cfg)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Train samples: {len(data['train_states'])}")
    print(f"  Test samples:  {len(data['test_states'])}")
    print(f"  State std:     {np.array2string(data['state_std'], precision=4)}")
    print(f"  Action std:    {data['action_std']:.6f}")
    sys.stdout.flush()

    # 2. Get test segments
    print("\n[2/5] Getting test segments...")
    sys.stdout.flush()
    segments = get_test_segments(data, n_segments=5, segment_length=200)
    print(f"  Got {len(segments)} segments, each {len(segments[0]['actions'])} steps")
    for i, seg in enumerate(segments):
        print(f"  Segment {i}: start_idx={seg['start_idx']}, "
              f"states shape={seg['states'].shape}, actions shape={seg['actions'].shape}")
    sys.stdout.flush()

    # 3. Train models
    print("\n[3/5] Training models (GP, E1, RDE-L)...")
    print("  E1 and RDE-L share the GP baseline (no redundant GP training)")
    sys.stdout.flush()
    t_train_start = time.time()
    models = train_models(data)
    print(f"  Total training time: {time.time()-t_train_start:.1f}s")
    sys.stdout.flush()

    # 4. Evaluate at multiple horizons
    print("\n[4/5] Evaluating multi-step rollout (Mode B)...")
    sys.stdout.flush()
    horizons = [1, 5, 10, 20, 50, 100, 200]
    state_std = data['state_std']

    all_results = {}
    for model_name, model in models.items():
        print(f"\n  --- {model_name.upper()} ---")
        sys.stdout.flush()
        t0 = time.time()
        raw_results = evaluate_model_on_segments(model, segments, mode='b')
        print(f"  Rollout time: {time.time()-t0:.1f}s")
        sys.stdout.flush()

        aggregated = aggregate_results(raw_results, horizons, state_std)
        model_report = {}
        for h in horizons:
            metrics_list = aggregated[h]
            agg = aggregate_across_segments(metrics_list, state_std)
            model_report[str(h)] = agg
            nmae = agg['overall']['nmae']['mean']
            mae = agg['overall']['mae']['mean']
            rmse = agg['overall']['rmse']['mean']
            surv = agg['survival_rate']
            ep_err = agg['overall']['endpoint_error']['mean']
            print(f"  H={h:>4d}: NMAE={nmae:.6f}  MAE={mae:.6f}  RMSE={rmse:.6f}  "
                  f"survival={surv:.1%}  ep_err={ep_err:.6f}")
            sys.stdout.flush()

        all_results[model_name] = model_report

    # 5. V6 4D comparison
    print("\n[5/5] Comparing with V6 4D results...")
    sys.stdout.flush()
    v6_4d_nmae = {
        'gp':  {'1': 0.0001, '5': 0.0003, '10': 0.0004, '20': 0.0006,
                '50': 0.0027, '100': 0.0195, '200': 1.77},
        'e1':  {'1': 0.0001, '5': 0.0019, '10': 0.0021, '20': 0.0031,
                '50': 0.0126, '100': 0.0896, '200': 5.32},
        'rde_l': {'1': 0.0001, '5': 0.0015, '10': 0.0017, '20': 0.0028,
                  '50': 0.0126, '100': 0.0897, '200': 4.93},
    }

    comparison = {}
    for model_name in ['gp', 'e1', 'rde_l']:
        comparison[model_name] = {}
        for h_str in [str(h) for h in horizons]:
            v8_nmae = all_results[model_name][h_str]['overall']['nmae']['mean']
            v6_nmae = v6_4d_nmae.get(model_name, {}).get(h_str, float('nan'))
            ratio = v8_nmae / v6_nmae if (v6_nmae and not np.isnan(v6_nmae) and v6_nmae > 0) else float('nan')
            comparison[model_name][h_str] = {
                'v8_7d_nmae': v8_nmae,
                'v6_4d_nmae': v6_nmae,
                'ratio_7d_over_4d': ratio,
            }

    # Print comparison table
    print("\n  NMAE Comparison (V8 7D vs V6 4D, Mode B):")
    header = f"  {'Model':<12s}" + "".join(f"{'H='+str(h):>10s}" for h in horizons)
    print(header)
    print("  " + "-" * (12 + 10 * len(horizons)))
    for model_name in ['gp', 'e1', 'rde_l']:
        row = f"  {'V8-'+model_name.upper():<12s}"
        for h in horizons:
            v = all_results[model_name][str(h)]['overall']['nmae']['mean']
            row += f"{v:>10.6f}"
        print(row)
    print()
    for model_name in ['gp', 'e1', 'rde_l']:
        row = f"  {'V6-'+model_name.upper():<12s}"
        for h in horizons:
            v = v6_4d_nmae.get(model_name, {}).get(str(h), float('nan'))
            row += f"{v:>10.6f}"
        print(row)

    # Build final output
    output = {
        'agent': 'E',
        'task': 'Multi-step Rollout Evaluation',
        'version': 'V8',
        'dimension': '7D',
        'data_info': {
            'n_train': len(data['train_states']),
            'n_test': len(data['test_states']),
            'state_names': STATE_NAMES_7D,
            'state_std': data['state_std'].tolist(),
            'action_std': float(data['action_std']),
        },
        'eval_config': {
            'mode': 'b',
            'horizons': horizons,
            'n_segments': len(segments),
            'segment_length': 200,
        },
        'results': all_results,
        'v6_4d_comparison': {
            'v6_4d_nmae': v6_4d_nmae,
            'detailed': comparison,
        },
        'key_findings': [],
    }

    # Determine key findings
    # 1. Which model is best overall?
    best_model = None
    best_score = float('inf')
    for model_name in ['gp', 'e1', 'rde_l']:
        avg_nmae = np.mean([
            all_results[model_name][str(h)]['overall']['nmae']['mean']
            for h in horizons
            if not np.isnan(all_results[model_name][str(h)]['overall']['nmae']['mean'])
        ])
        if avg_nmae < best_score:
            best_score = avg_nmae
            best_model = model_name

    output['key_findings'].append(f"Best model by average NMAE across horizons: {best_model.upper()} (avg NMAE={best_score:.6f})")

    # 2. Survival analysis
    for model_name in ['gp', 'e1', 'rde_l']:
        surv_200 = all_results[model_name]['200']['survival_rate']
        output['key_findings'].append(f"{model_name.upper()} survival rate at H=200: {surv_200:.1%}")

    # 3. V8 vs V6 comparison
    for model_name in ['gp', 'e1', 'rde_l']:
        v8_h50 = all_results[model_name]['50']['overall']['nmae']['mean']
        v6_h50 = v6_4d_nmae.get(model_name, {}).get('50', float('nan'))
        if not np.isnan(v6_h50) and v6_h50 > 0:
            ratio = v8_h50 / v6_h50
            output['key_findings'].append(
                f"{model_name.upper()} V8/7D vs V6/4D NMAE at H=50: {v8_h50:.6f} / {v6_h50:.6f} = {ratio:.2f}x"
            )

    # 4. Divergence analysis
    for model_name in ['gp', 'e1', 'rde_l']:
        nmae_200 = all_results[model_name]['200']['overall']['nmae']['mean']
        output['key_findings'].append(
            f"{model_name.upper()} NMAE at H=200: {nmae_200:.6f} "
            f"({'STABLE' if nmae_200 < 1.0 else 'DIVERGED'})"
        )

    # Save results
    output_path = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8/raw_results/AGENT_E_MULTISTEP.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
    sys.stdout.flush()

    # Print detailed per-state NMAE for best model
    print(f"\n{'='*70}")
    print(f"DETAILED PER-STATE NMAE for {best_model.upper()} (best model):")
    print(f"{'='*70}")
    header = f"  {'State':<12s}" + "".join(f"{'H='+str(h):>10s}" for h in horizons)
    print(header)
    print("  " + "-" * (12 + 10 * len(horizons)))
    for name in STATE_NAMES_7D:
        row = f"  {name:<12s}"
        for h in horizons:
            v = all_results[best_model][str(h)]['per_state_nmae'][name]['mean']
            row += f"{v:>10.6f}"
        print(row)

    # Print endpoint error for all models
    print(f"\n{'='*70}")
    print("ENDPOINT ERROR (max abs error at final step):")
    print(f"{'='*70}")
    header = f"  {'Model':<12s}" + "".join(f"{'H='+str(h):>10s}" for h in horizons)
    print(header)
    print("  " + "-" * (12 + 10 * len(horizons)))
    for model_name in ['gp', 'e1', 'rde_l']:
        row = f"  {model_name.upper():<12s}"
        for h in horizons:
            v = all_results[model_name][str(h)]['overall']['endpoint_error']['mean']
            row += f"{v:>10.6f}"
        print(row)

    print(f"\n{'='*70}")
    print("DONE. All metrics computed and saved.")
    print(f"{'='*70}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
