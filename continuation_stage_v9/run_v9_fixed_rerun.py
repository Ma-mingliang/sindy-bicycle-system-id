"""V9 Fixed Re-run: Same V9 hyperparameters, but with bug fixes applied.

Tests whether the audit-identified fixes actually improve results:
  1. v9_baseline_ref     -- V9 baseline (reference, from V12_ROOTFIX.json)
  2. v9_fixed_seq        -- Fix: sequential batching (correct multi-step loss)
  3. v9_fixed_seq_contract -- Fix: sequential + contractivity regularization

All three use identical hyperparameters:
  hidden=64, depth=3, tanh, lr=1e-3, n_epochs=200, seed=43,
  rollout_curriculum='1,5,10,20'

Output: raw_results/V9_FIXED_RERUN.json
"""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')

import os
import json
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

try:
    import torch
    if not torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
except ImportError:
    pass

from canonical_node.config_v9 import STATE_NAMES_7D
from canonical_node.data_loader_v9 import load_7d_data, get_test_segments
from canonical_node.neural_ode_v9 import NeuralODEV9, NeuralODEConfig
from canonical_node.neural_ode_v12 import NeuralODEV12, V12Config
from canonical_node.evaluation_v9 import multi_step_evaluate

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
V9_PATH = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9'
RESULTS_PATH = os.path.join(V9_PATH, 'raw_results')
RESULTS_FILE = os.path.join(RESULTS_PATH, 'V9_FIXED_RERUN.json')
OLD_RESULTS_FILE = os.path.join(RESULTS_PATH, 'V12_ROOTFIX.json')

HORIZONS = [1, 10, 50, 100, 200, 500]
N_SEGMENTS = 5
SEG_LEN = 1100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [SAVED] {path}")


def prepare_v12_data(data: dict, segments: list) -> dict:
    """Prepare training/validation data with episode boundaries for V12."""
    train_states = data['train_states']
    train_actions = data['train_actions']
    train_deltas = data['train_deltas']

    # Compute episode boundaries within training data
    train_mask = data['train_mask']
    ep_starts_orig = data['episode_starts']
    ep_ends_orig = data['episode_ends']
    train_indices = np.where(train_mask)[0]
    orig_to_train = {}
    for ti, oi in enumerate(train_indices):
        orig_to_train[oi] = ti
    episode_boundaries = []
    for es, ee in zip(ep_starts_orig, ep_ends_orig):
        if es in orig_to_train and ee - 1 in orig_to_train:
            t_start = orig_to_train[es]
            t_end = orig_to_train[ee - 1] + 1
            if t_end - t_start >= 2:
                episode_boundaries.append((t_start, t_end))

    # Validation: last 10% of training data
    n_val = max(int(len(train_states) * 0.1), 1000)
    val_states = train_states[-n_val:]
    val_actions = train_actions[-n_val:]
    val_deltas = train_deltas[-n_val:]
    train_states = train_states[:-n_val]
    train_actions = train_actions[:-n_val]
    train_deltas = train_deltas[:-n_val]

    train_len = len(train_states)
    episode_boundaries = [(s, min(e, train_len))
                          for s, e in episode_boundaries
                          if s < train_len and e > 0 and min(e, train_len) - s >= 2]

    return {
        'train_states': train_states,
        'train_actions': train_actions,
        'train_deltas': train_deltas,
        'episode_boundaries': episode_boundaries,
        'val_states': val_states,
        'val_actions': val_actions,
        'val_deltas': val_deltas,
    }


def build_config_dict(cfg) -> dict:
    if hasattr(cfg, '__dataclass_fields__'):
        return {k: v for k, v in cfg.__dict__.items()}
    return dict(cfg)


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # 1. V9 baseline reference -- load from existing V12_ROOTFIX.json
    {
        'name': 'v9_baseline_ref',
        'model_class': 'v9',
        'config': NeuralODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            n_epochs=200, seed=43,
            rollout_curriculum='1,5,10,20',
        ),
        'skip_training': True,  # Use existing result
    },
    # 2. V9 fixed: sequential batching only (same hyperparams as V9)
    {
        'name': 'v9_fixed_seq',
        'model_class': 'v12',
        'config': V12Config(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            n_epochs=200, seed=43,
            rollout_curriculum='1,5,10,20',
            integration_method='euler',
            use_adaptive_curriculum=False,
            use_contractivity=False,
            max_segment_len=20,
        ),
    },
    # 3. V9 fixed: sequential + contractivity (addresses survival collapse)
    {
        'name': 'v9_fixed_seq_contract',
        'model_class': 'v12',
        'config': V12Config(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            n_epochs=200, seed=43,
            rollout_curriculum='1,5,10,20',
            integration_method='euler',
            use_adaptive_curriculum=False,
            use_contractivity=True,
            lambda_contractive=0.05,
            contractive_warmup_epochs=20,
            contractive_ramp_epochs=30,
            max_segment_len=20,
        ),
    },
    # 4. Tier 1: sequential + contractivity + survival-aware loss
    {
        'name': 'v9_tier1_survival',
        'model_class': 'v12',
        'config': V12Config(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            n_epochs=200, seed=43,
            rollout_curriculum='1,5,10,20',
            integration_method='euler',
            use_adaptive_curriculum=False,
            use_contractivity=True,
            lambda_contractive=0.05,
            contractive_warmup_epochs=20,
            contractive_ramp_epochs=30,
            max_segment_len=20,
            use_survival_loss=True,
            lambda_survival=0.1,
            survival_limit_scale=0.95,
        ),
    },
]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_experiment(split_data: dict, segments: list, state_std: np.ndarray,
                   data: dict, exp_def: dict) -> dict:
    name = exp_def['name']
    model_class = exp_def['model_class']
    config = exp_def['config']

    print(f"\n{'='*60}")
    print(f"  [{name}] model_class={model_class}")
    print(f"  config: {build_config_dict(config)}")
    print(f"{'='*60}", flush=True)

    # Skip training for baseline reference (use existing result)
    if exp_def.get('skip_training'):
        print(f"  [SKIP] Loading existing result from V12_ROOTFIX.json")
        with open(OLD_RESULTS_FILE) as f:
            old_data = json.load(f)
        if 'v9_baseline' in old_data:
            return old_data['v9_baseline']
        else:
            print(f"  [WARN] v9_baseline not found in old results, training fresh")
            exp_def = {**exp_def, 'skip_training': False}

    # Instantiate model
    if model_class == 'v9':
        model = NeuralODEV9(config)
    elif model_class == 'v12':
        model = NeuralODEV12(config)
    else:
        raise ValueError(f"Unknown model_class: {model_class}")

    # Train
    print(f"  Training {name}...", flush=True)
    t0 = time.time()

    if model_class == 'v12':
        model.train(
            split_data['train_states'],
            split_data['train_actions'],
            split_data['train_deltas'],
            state_std,
            data['action_std'],
            data['delta_std'],
            val_states=split_data['val_states'],
            val_actions=split_data['val_actions'],
            val_deltas=split_data['val_deltas'],
            episode_boundaries=split_data.get('episode_boundaries'),
        )
    else:
        model.train(
            split_data['train_states'],
            split_data['train_actions'],
            split_data['train_deltas'],
            state_std,
            data['action_std'],
            data['delta_std'],
        )

    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")

    # Evaluate
    print(f"  Evaluating at horizons {HORIZONS}...", flush=True)
    eval_result = multi_step_evaluate(model, segments, state_std, HORIZONS)

    # Save checkpoint
    ckpt_dir = os.path.join(V9_PATH, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'V9_FIXED_{name}.pt')
    model.save(ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    # Build result dict
    nmae = {}
    per_state_nmae = {}
    survival = {}
    for h in HORIZONS:
        nmae[h] = eval_result[h]['nmae_mean']
        survival[h] = eval_result[h]['survival_rate']
        per_state_nmae[h] = eval_result[h]['per_state_nmae']

    result = {
        'nmae': {str(k): v for k, v in nmae.items()},
        'per_state_nmae': {str(k): v for k, v in per_state_nmae.items()},
        'survival_rate': {str(k): v for k, v in survival.items()},
        'training_time_seconds': train_time,
        'config_dict': build_config_dict(config),
    }

    # Print summary
    print(f"\n  [{name}] Results:")
    print(f"  {'Horizon':>8} {'NMAE':>10} {'Survival':>10}")
    print(f"  {'-'*30}")
    for h in HORIZONS:
        print(f"  {h:>8} {nmae[h]:>10.4f} {survival[h]:>10.0%}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("="*60)
    print("  V9 Fixed Re-run Experiment")
    print("  Same V9 hyperparameters, bug fixes applied")
    print("="*60)

    # Load data
    print("\n[1/4] Loading data...", flush=True)
    data = load_7d_data()
    print(f"  Train: {data['train_states'].shape}, Test: {data['test_states'].shape}")

    # Get test segments
    print("[2/4] Extracting test segments...", flush=True)
    segments = get_test_segments(data, n_segments=N_SEGMENTS, segment_length=SEG_LEN)
    print(f"  Got {len(segments)} segments, each {segments[0]['states'].shape[0]} steps")

    # Prepare V12 data (with episode boundaries)
    print("[3/4] Preparing data splits...", flush=True)
    split_data = prepare_v12_data(data, segments)
    print(f"  Train: {split_data['train_states'].shape[0]}, Val: {split_data['val_states'].shape[0]}")
    print(f"  Episode boundaries: {len(split_data['episode_boundaries'])}")

    # Run experiments
    print("[4/4] Running experiments...", flush=True)
    results = {}
    for exp_def in EXPERIMENTS:
        name = exp_def['name']
        result = run_experiment(split_data, segments, data['state_std'], data, exp_def)
        results[name] = result
        # Incremental save
        save_json(RESULTS_FILE, results)

    # Comparison table
    print("\n" + "="*70)
    print("  COMPARISON TABLE")
    print("="*70)
    print(f"  {'Horizon':>8}", end='')
    for name in results:
        print(f"  {name:>20}", end='')
    print()
    print(f"  {'':>8}", end='')
    for name in results:
        print(f"  {'NMAE(Surv)':>20}", end='')
    print()
    print(f"  {'-'*70}")

    for h in HORIZONS:
        print(f"  {h:>8}", end='')
        for name in results:
            nmae_val = results[name]['nmae'].get(str(h), results[name]['nmae'].get(h, float('nan')))
            surv_val = results[name]['survival_rate'].get(str(h), results[name]['survival_rate'].get(h, float('nan')))
            print(f"  {nmae_val:.3f}({surv_val:.0%}){'':>4}", end='')
        print()

    # Improvement analysis
    print("\n" + "="*70)
    print("  IMPROVEMENT vs V9 BASELINE")
    print("="*70)
    if 'v9_baseline_ref' in results:
        base = results['v9_baseline_ref']
        for name in results:
            if name == 'v9_baseline_ref':
                continue
            print(f"\n  {name} vs v9_baseline_ref:")
            for h in HORIZONS:
                base_nmae = base['nmae'].get(str(h), base['nmae'].get(h, 0))
                curr_nmae = results[name]['nmae'].get(str(h), results[name]['nmae'].get(h, 0))
                base_surv = base['survival_rate'].get(str(h), base['survival_rate'].get(h, 0))
                curr_surv = results[name]['survival_rate'].get(str(h), results[name]['survival_rate'].get(h, 0))
                nmae_change = (curr_nmae - base_nmae) / max(base_nmae, 1e-10) * 100
                surv_change = curr_surv - base_surv
                nmae_arrow = "DOWN" if nmae_change < 0 else "UP"
                surv_arrow = "UP" if surv_change > 0 else ("DOWN" if surv_change < 0 else "SAME")
                print(f"    H={h:>3}: NMAE {base_nmae:.4f} -> {curr_nmae:.4f} ({nmae_change:+.1f}% {nmae_arrow}), "
                      f"Surv {base_surv:.0%} -> {curr_surv:.0%} ({surv_arrow})")

    save_json(RESULTS_FILE, results)
    print(f"\nDone. Results saved to {RESULTS_FILE}")
