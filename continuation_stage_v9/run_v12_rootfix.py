"""V12 Root-Fix Experiment Runner.

Compares V12 bug fixes against V9 baseline to isolate the impact of each fix:

  1. Sequential batch sampling (fixes shuffled DataLoader)
  2. Extended rollout curriculum (H up to 100, was 20)
  3. RK4 integration (reduces truncation error)
  4. Combined: adaptive curriculum + contractivity regularization

Configs:
  v9_baseline         -- V9 reproduced reference
  v12_sequential_only -- V12 with only the DataLoader fix (euler, same curriculum)
  v12_extended_rollout-- V12 sequential + extended curriculum (H=100, adaptive)
  v12_rk4             -- V12 sequential + RK4 (same curriculum as V9)
  v12_full            -- V12 all fixes combined

Output: raw_results/V12_ROOTFIX.json (incremental save after each experiment)
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

# Force CPU if no GPU available; detect and use GPU if present.
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
V12_PATH = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9'
RESULTS_PATH = os.path.join(V12_PATH, 'raw_results')
RESULTS_FILE = os.path.join(RESULTS_PATH, 'V12_ROOTFIX.json')

HORIZONS = [1, 10, 50, 100, 200, 500]
N_SEGMENTS = 5
SEG_LEN = 1100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_json(path: str, data: dict) -> None:
    """Write data to JSON, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [SAVED] {path}")


def prepare_data(data: dict, segments: list):
    """Prepare training and validation data from the full dataset.

    Uses the FULL training data (106K samples) for training — NOT test segments.
    Test segments are used only for evaluation via multi_step_evaluate().

    For V12 validation: use last 10% of training data as held-out validation
    (the V12 train method also supports val_states/val_actions/val_deltas).

    IMPORTANT: Uses data['train_deltas'] from the data loader (which correctly
    computes deltas as next_obs - obs from the original contiguous data).
    DO NOT recompute deltas from consecutive state differences — training
    episodes are shuffled, so consecutive states may span episode boundaries.
    """
    train_states = data['train_states']
    train_actions = data['train_actions']
    train_deltas = data['train_deltas']  # Use pre-computed deltas from data loader

    # Compute episode boundaries within training data for SequentialSegmentDataset
    # Training data is organized by episodes (shuffled), so we need to find where
    # each episode starts and ends within the training array.
    train_mask = data['train_mask']
    ep_starts_orig = data['episode_starts']
    ep_ends_orig = data['episode_ends']
    # Map original indices to training data indices
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

    # Adjust episode boundaries to exclude validation data
    train_len = len(train_states)
    episode_boundaries = [(s, min(e, train_len))
                          for s, e in episode_boundaries
                          if s < train_len and e > 0 and min(e, train_len) - s >= 2]

    return {
        'train_states': train_states,
        'train_actions': train_actions,
        'episode_boundaries': episode_boundaries,
        'train_deltas': train_deltas,
        'val_states': val_states,
        'val_actions': val_actions,
        'val_deltas': val_deltas,
    }


def build_config_dict(cfg) -> dict:
    """Extract a JSON-serialisable dict from a config dataclass."""
    if hasattr(cfg, '__dataclass_fields__'):
        return {k: v for k, v in cfg.__dict__.items()}
    return dict(cfg)


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # 1. V9 baseline (reproduced) -- reference
    {
        'name': 'v9_baseline',
        'model_class': 'v9',
        'config': NeuralODEConfig(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            n_epochs=200, seed=43,
            rollout_curriculum='1,5,10,20',
        ),
    },
    # 2. V12 sequential batch ONLY (euler, no extended rollout, no contractivity)
    #    Isolates the impact of fixing the DataLoader shuffle bug
    {
        'name': 'v12_sequential_only',
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
    # 3. V12 sequential + extended rollout (up to 100 steps)
    {
        'name': 'v12_extended_rollout',
        'model_class': 'v12',
        'config': V12Config(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            n_epochs=400, seed=43,
            rollout_curriculum='1,5,10,20,50,100',
            integration_method='euler',
            use_adaptive_curriculum=True,
            use_contractivity=False,
            max_segment_len=100,
        ),
    },
    # 4. V12 sequential + RK4 integration
    {
        'name': 'v12_rk4',
        'model_class': 'v12',
        'config': V12Config(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            n_epochs=200, seed=43,
            rollout_curriculum='1,5,10,20',
            integration_method='rk4',
            use_adaptive_curriculum=False,
            use_contractivity=False,
            max_segment_len=20,
        ),
    },
    # 5. V12 FULL (sequential + extended + RK4 + contractivity)
    {
        'name': 'v12_full',
        'model_class': 'v12',
        'config': V12Config(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            n_epochs=400, seed=43,
            rollout_curriculum='1,5,10,20,50,100',
            integration_method='rk4',
            use_adaptive_curriculum=True,
            use_contractivity=True,
            lambda_contractive=0.05,
            contractive_warmup_epochs=20,
            max_segment_len=100,
        ),
    },
]


# ---------------------------------------------------------------------------
# Run a single experiment
# ---------------------------------------------------------------------------

def run_experiment(
    split_data: dict,
    segments: list,
    state_std: np.ndarray,
    exp_def: dict,
) -> dict:
    """Train and evaluate one experiment config.

    Returns a result dict with NMAE, per-state NMAE, survival rate, and timing.
    """
    name = exp_def['name']
    model_class = exp_def['model_class']
    config = exp_def['config']

    print(f"\n{'='*60}")
    print(f"  [{name}] model_class={model_class}")
    print(f"  config: {build_config_dict(config)}")
    print(f"{'='*60}", flush=True)

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
        # V9: pass validation data as well (train method accepts optional val args)
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
        )

    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")

    # Evaluate
    print(f"  Evaluating at horizons {HORIZONS}...", flush=True)
    eval_result = multi_step_evaluate(model, segments, state_std, HORIZONS)

    # Save checkpoint
    ckpt_dir = os.path.join(V12_PATH, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'V12_ROOTFIX_{name}.pt')
    model.save(ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    # Build result dict
    nmae = {}
    per_state_nmae = {}
    survival = {}
    for h in HORIZONS:
        nmae[h] = eval_result[h]['nmae_mean']
        survival[h] = eval_result[h]['survival_rate']
        per_state_nmae[h] = {
            sn: eval_result[h]['per_state_nmae'][sn]['mean']
            for sn in STATE_NAMES_7D
        }

    result = {
        'nmae': nmae,
        'per_state_nmae': per_state_nmae,
        'survival_rate': survival,
        'training_time_seconds': round(train_time, 1),
        'config_dict': build_config_dict(config),
    }

    # Print quick summary
    summary_parts = []
    for h in [1, 10, 50, 100, 200, 500]:
        summary_parts.append(f"H={h}:{nmae[h]:.4f}")
    print(f"  [{name}] {', '.join(summary_parts)}  ({train_time:.0f}s)")

    return result


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def format_comparison_table(results: dict) -> str:
    """Build a formatted comparison table string."""
    lines = []
    sep = '-' * 90

    # Header
    hdr = f"{'Config':<30}"
    for h in HORIZONS:
        hdr += f" {'H=' + str(h):>8}"
    hdr += f" {'Time':>8}"
    lines.append(sep)
    lines.append(hdr)
    lines.append(sep)

    # Rows
    for exp_def in EXPERIMENTS:
        name = exp_def['name']
        r = results.get(name, {})
        if 'error' in r:
            lines.append(f"{name:<30} ERROR: {r['error'][:50]}")
            continue

        row = f"{name:<30}"
        for h in HORIZONS:
            val = r.get('nmae', {}).get(h, float('nan'))
            row += f" {val:>8.4f}"
        t = r.get('training_time_seconds', 0)
        row += f" {t:>7.0f}s"
        lines.append(row)

    lines.append(sep)

    # Survival rates
    lines.append("")
    lines.append("Survival Rates:")
    lines.append(sep)
    surv_hdr = f"{'Config':<30}"
    for h in HORIZONS:
        surv_hdr += f" {'H=' + str(h):>8}"
    lines.append(surv_hdr)
    lines.append(sep)

    for exp_def in EXPERIMENTS:
        name = exp_def['name']
        r = results.get(name, {})
        if 'error' in r:
            continue

        row = f"{name:<30}"
        for h in HORIZONS:
            val = r.get('survival_rate', {}).get(h, float('nan'))
            row += f" {val:>7.0%}"
        lines.append(row)

    lines.append(sep)

    # Per-state NMAE at H=100
    lines.append("")
    lines.append("Per-State NMAE at H=100 (lower is better):")
    lines.append(sep)
    ps_hdr = f"{'Config':<30}"
    for sn in STATE_NAMES_7D:
        ps_hdr += f" {sn:>8}"
    lines.append(ps_hdr)
    lines.append(sep)

    for exp_def in EXPERIMENTS:
        name = exp_def['name']
        r = results.get(name, {})
        if 'error' in r:
            continue

        psn = r.get('per_state_nmae', {}).get(100, {})
        row = f"{name:<30}"
        for sn in STATE_NAMES_7D:
            val = psn.get(sn, float('nan'))
            row += f" {val:>8.4f}"
        lines.append(row)

    lines.append(sep)

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global data  # needed by run_experiment to access action_std / delta_std

    print("=" * 70)
    print("V12 Root-Fix Experiment: Isolating Each Bug Fix")
    print("=" * 70)
    t_start = time.time()

    # ------------------------------------------------------------------
    # [0] Load data
    # ------------------------------------------------------------------
    print("\n[0] Loading data...")
    data = load_7d_data()
    segments = get_test_segments(data, n_segments=N_SEGMENTS, segment_length=SEG_LEN)
    state_std = data['state_std']

    print(f"  Train samples: {len(data['train_states'])}")
    print(f"  Test segments: {len(segments)}")
    print(f"  State std: {state_std}")

    # ------------------------------------------------------------------
    # [1] Prepare training data (use FULL training set, not test segments)
    # ------------------------------------------------------------------
    print("\n[1] Preparing training data (full dataset)...")
    split_data = prepare_data(data, segments)
    print(f"  Train samples: {len(split_data['train_states'])}")
    print(f"  Val samples:   {len(split_data['val_states'])}")

    # ------------------------------------------------------------------
    # [2] Run experiments
    # ------------------------------------------------------------------
    print(f"\n[2] Running {len(EXPERIMENTS)} experiments...")
    results = {}

    for i, exp_def in enumerate(EXPERIMENTS):
        print(f"\n--- Experiment {i + 1}/{len(EXPERIMENTS)} ---")
        try:
            results[exp_def['name']] = run_experiment(
                split_data, segments, state_std, exp_def
            )
        except Exception as e:
            import traceback
            print(f"  FAILED {exp_def['name']}: {e}")
            traceback.print_exc()
            results[exp_def['name']] = {'error': str(e)}

        # Incremental save after each experiment (crash recovery)
        save_json(RESULTS_FILE, results)

    # ------------------------------------------------------------------
    # [3] Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    table = format_comparison_table(results)
    print(table)

    # Store table in results for persistence
    results['comparison_table'] = table
    save_json(RESULTS_FILE, results)

    total_time = time.time() - t_start
    print(f"\nTotal experiment time: {total_time / 60:.1f} min")
    print(f"Results saved to: {RESULTS_FILE}")
    print("Done.")


if __name__ == '__main__':
    main()
