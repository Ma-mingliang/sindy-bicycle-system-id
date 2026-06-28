"""Repair Experiments: Test solutions for the accuracy-stability tradeoff.

Hypotheses:
  H1: Soft contractivity (lambda=0.01) finds a better NMAE-survival sweet spot.
  H2: Inference-time state clipping restores survival while preserving NMAE gains.
  H3: Combining soft contractivity + clipping achieves both goals.

Configurations:
  1. soft_contract      -- lambda=0.01, warmup=30, ramp=50
  2. clip_seq           -- Load v9_fixed_seq checkpoint, add inference-time clipping
  3. soft_contract_clip -- Train with lambda=0.01, evaluate with clipping

Output: raw_results/REPAIR_EXPERIMENTS.json
"""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')

import os
import json
import time
import warnings
import copy
import numpy as np

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

try:
    import torch
    if not torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
except ImportError:
    pass

from canonical_node.config_v9 import STATE_NAMES_7D, PHYSICAL_LIMITS
from canonical_node.data_loader_v9 import load_7d_data, get_test_segments
from canonical_node.neural_ode_v12 import NeuralODEV12, V12Config
from canonical_node.evaluation_v9 import multi_step_evaluate

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
V9_PATH = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9'
RESULTS_PATH = os.path.join(V9_PATH, 'raw_results')
RESULTS_FILE = os.path.join(RESULTS_PATH, 'REPAIR_EXPERIMENTS.json')
PREV_RESULTS_FILE = os.path.join(RESULTS_PATH, 'V9_FIXED_RERUN.json')

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
# Inference-time clipping wrapper
# ---------------------------------------------------------------------------

class ClippedModel:
    """Wraps a NeuralODE model to apply state clipping at each prediction step.

    After each predict() call, clips the output state to physical limits
    scaled by `clip_scale`. This prevents trajectory explosion during
    long rollouts without modifying the trained model.
    """

    def __init__(self, base_model, clip_scale: float = 0.95):
        self._base = base_model
        self._clip_scale = clip_scale
        self._limits = self._compute_limits()

    def _compute_limits(self):
        """Compute clipping bounds in physical units."""
        limits = np.array([
            PHYSICAL_LIMITS['e_y'],
            PHYSICAL_LIMITS['e_psi'],
            PHYSICAL_LIMITS['v'],
            PHYSICAL_LIMITS['theta'],
            PHYSICAL_LIMITS['theta_dot'],
            PHYSICAL_LIMITS['delta'],
            PHYSICAL_LIMITS['delta_dot'],
        ])
        return limits * self._clip_scale

    def predict(self, s, tau):
        """Predict with post-step clipping."""
        s_next = self._base.predict(s, tau)
        # Clip to physical limits
        s_next = np.clip(s_next, -self._limits, self._limits)
        return s_next

    def name(self):
        return f"clipped({self._base.name()})"

    def save(self, path):
        self._base.save(path)

    def load(self, path):
        self._base.load(path)


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # 1. Soft contractivity: lambda=0.01 (5x weaker than v9_fixed_seq_contract)
    {
        'name': 'soft_contract',
        'model_class': 'v12',
        'config': V12Config(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            n_epochs=200, seed=43,
            rollout_curriculum='1,5,10,20',
            integration_method='euler',
            use_adaptive_curriculum=False,
            use_contractivity=True,
            lambda_contractive=0.01,       # 5x weaker
            contractive_warmup_epochs=30,   # longer warmup
            contractive_ramp_epochs=50,     # longer ramp
            max_segment_len=20,
        ),
        'eval_clip': False,
    },
    # 2. Inference-time clipping on v9_fixed_seq (no retraining)
    {
        'name': 'clip_seq',
        'model_class': 'v12',
        'load_checkpoint': 'V9_FIXED_v9_fixed_seq.pt',
        'eval_clip': True,
        'clip_scale': 0.95,
    },
    # 3. Soft contractivity + inference-time clipping
    {
        'name': 'soft_contract_clip',
        'model_class': 'v12',
        'config': V12Config(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            n_epochs=200, seed=43,
            rollout_curriculum='1,5,10,20',
            integration_method='euler',
            use_adaptive_curriculum=False,
            use_contractivity=True,
            lambda_contractive=0.01,
            contractive_warmup_epochs=30,
            contractive_ramp_epochs=50,
            max_segment_len=20,
        ),
        'eval_clip': True,
        'clip_scale': 0.95,
    },
    # 4. Very soft contractivity: lambda=0.005 (10x weaker)
    {
        'name': 'very_soft_contract',
        'model_class': 'v12',
        'config': V12Config(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            n_epochs=200, seed=43,
            rollout_curriculum='1,5,10,20',
            integration_method='euler',
            use_adaptive_curriculum=False,
            use_contractivity=True,
            lambda_contractive=0.005,      # 10x weaker
            contractive_warmup_epochs=30,
            contractive_ramp_epochs=50,
            max_segment_len=20,
        ),
        'eval_clip': False,
    },
    # 5. Very soft contractivity + clipping
    {
        'name': 'very_soft_contract_clip',
        'model_class': 'v12',
        'config': V12Config(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            n_epochs=200, seed=43,
            rollout_curriculum='1,5,10,20',
            integration_method='euler',
            use_adaptive_curriculum=False,
            use_contractivity=True,
            lambda_contractive=0.005,
            contractive_warmup_epochs=30,
            contractive_ramp_epochs=50,
            max_segment_len=20,
        ),
        'eval_clip': True,
        'clip_scale': 0.95,
    },
]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_experiment(split_data: dict, segments: list, state_std: np.ndarray,
                   data: dict, exp_def: dict) -> dict:
    name = exp_def['name']

    print(f"\n{'='*60}")
    print(f"  [{name}]")
    print(f"{'='*60}", flush=True)

    # Load checkpoint or train
    if 'load_checkpoint' in exp_def:
        ckpt_path = os.path.join(V9_PATH, 'checkpoints', exp_def['load_checkpoint'])
        print(f"  Loading checkpoint: {ckpt_path}")
        config = V12Config(
            hidden=64, depth=3, activation='tanh', lr=1e-3,
            n_epochs=200, seed=43,
            rollout_curriculum='1,5,10,20',
            integration_method='euler',
            max_segment_len=20,
        )
        model = NeuralODEV12(config)
        model.load(ckpt_path)
        train_time = 0.0
    else:
        config = exp_def['config']
        print(f"  config: {build_config_dict(config)}")
        model = NeuralODEV12(config)

        print(f"  Training {name}...", flush=True)
        t0 = time.time()
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
        train_time = time.time() - t0
        print(f"  Training completed in {train_time:.1f}s")

        # Save checkpoint
        ckpt_dir = os.path.join(V9_PATH, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'REPAIR_{name}.pt')
        model.save(ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    # Apply clipping wrapper if requested
    eval_model = model
    if exp_def.get('eval_clip'):
        clip_scale = exp_def.get('clip_scale', 0.95)
        eval_model = ClippedModel(model, clip_scale=clip_scale)
        print(f"  Evaluation: clipping enabled (scale={clip_scale})")

    # Evaluate
    print(f"  Evaluating at horizons {HORIZONS}...", flush=True)
    eval_result = multi_step_evaluate(eval_model, segments, state_std, HORIZONS)

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
        'config_dict': build_config_dict(config) if 'config' in exp_def else {'loaded': exp_def.get('load_checkpoint')},
        'eval_clipped': exp_def.get('eval_clip', False),
        'clip_scale': exp_def.get('clip_scale', None),
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
    print("  Repair Experiments")
    print("  Testing solutions for accuracy-stability tradeoff")
    print("="*60)

    # Load data
    print("\n[1/4] Loading data...", flush=True)
    data = load_7d_data()
    print(f"  Train: {data['train_states'].shape}, Test: {data['test_states'].shape}")

    # Get test segments
    print("[2/4] Extracting test segments...", flush=True)
    segments = get_test_segments(data, n_segments=N_SEGMENTS, segment_length=SEG_LEN)
    print(f"  Got {len(segments)} segments, each {segments[0]['states'].shape[0]} steps")

    # Prepare V12 data
    print("[3/4] Preparing data splits...", flush=True)
    split_data = prepare_v12_data(data, segments)
    print(f"  Train: {split_data['train_states'].shape[0]}, Val: {split_data['val_states'].shape[0]}")
    print(f"  Episode boundaries: {len(split_data['episode_boundaries'])}")

    # Load previous results for comparison
    prev_results = {}
    if os.path.exists(PREV_RESULTS_FILE):
        with open(PREV_RESULTS_FILE) as f:
            prev_results = json.load(f)
        print(f"  Loaded previous results from {PREV_RESULTS_FILE}")

    # Run experiments
    print("[4/4] Running experiments...", flush=True)
    results = {}

    # Include previous results for comparison
    for k, v in prev_results.items():
        results[k] = v

    for exp_def in EXPERIMENTS:
        name = exp_def['name']
        result = run_experiment(split_data, segments, data['state_std'], data, exp_def)
        results[name] = result
        # Incremental save
        save_json(RESULTS_FILE, results)

    # Comparison table
    print("\n" + "="*80)
    print("  COMPARISON TABLE")
    print("="*80)
    print(f"  {'Config':>25} {'H=1':>8} {'H=10':>8} {'H=50':>8} {'H=100':>8} {'H=200':>8} {'H=500':>8} {'S@500':>6}")
    print(f"  {'-'*80}")

    for name in results:
        nmae_vals = results[name]['nmae']
        surv_vals = results[name]['survival_rate']
        h1 = nmae_vals.get('1', nmae_vals.get(1, float('nan')))
        h10 = nmae_vals.get('10', nmae_vals.get(10, float('nan')))
        h50 = nmae_vals.get('50', nmae_vals.get(50, float('nan')))
        h100 = nmae_vals.get('100', nmae_vals.get(100, float('nan')))
        h200 = nmae_vals.get('200', nmae_vals.get(200, float('nan')))
        h500 = nmae_vals.get('500', nmae_vals.get(500, float('nan')))
        s500 = surv_vals.get('500', surv_vals.get(500, float('nan')))
        marker = " *" if name in [e['name'] for e in EXPERIMENTS] else ""
        print(f"  {name:>25} {h1:>8.4f} {h10:>8.4f} {h50:>8.4f} {h100:>8.4f} {h200:>8.4f} {h500:>8.4f} {s500:>5.0%}{marker}")

    # Acceptance criteria check
    print("\n" + "="*80)
    print("  ACCEPTANCE CRITERIA CHECK")
    print("="*80)
    if 'v9_baseline_ref' in results:
        base = results['v9_baseline_ref']
        base_nmae_50 = base['nmae'].get('50', base['nmae'].get(50, 0))
        base_nmae_500 = base['nmae'].get('500', base['nmae'].get(500, 0))
        base_nmae_1 = base['nmae'].get('1', base['nmae'].get(1, 0))

        for name in results:
            if name == 'v9_baseline_ref':
                continue
            r = results[name]
            nmae_1 = r['nmae'].get('1', r['nmae'].get(1, 0))
            nmae_50 = r['nmae'].get('50', r['nmae'].get(50, 0))
            nmae_500 = r['nmae'].get('500', r['nmae'].get(500, 0))
            surv_500 = r['survival_rate'].get('500', r['survival_rate'].get(500, 0))

            crit_50 = nmae_50 < base_nmae_50
            crit_500 = nmae_500 < base_nmae_500
            crit_surv = surv_500 >= 1.0
            crit_1 = nmae_1 < 0.020

            all_pass = crit_50 and crit_500 and crit_surv and crit_1
            status = "PASS" if all_pass else "FAIL"

            print(f"\n  {name}: {status}")
            print(f"    H=50  NMAE < {base_nmae_50:.3f}: {nmae_50:.4f} {'PASS' if crit_50 else 'FAIL'}")
            print(f"    H=500 NMAE < {base_nmae_500:.3f}: {nmae_500:.4f} {'PASS' if crit_500 else 'FAIL'}")
            print(f"    Surv@500 = 100%: {surv_500:.0%} {'PASS' if crit_surv else 'FAIL'}")
            print(f"    H=1   NMAE < 0.020: {nmae_1:.4f} {'PASS' if crit_1 else 'FAIL'}")

    save_json(RESULTS_FILE, results)
    print(f"\nDone. Results saved to {RESULTS_FILE}")
