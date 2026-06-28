"""
Advanced V9 Repair Experiments
==============================
Tests:
1. Multi-seed validation (seeds 43, 44, 45)
2. Various contractivity strengths
3. Extended curriculum (400 epochs)
"""

import sys
import os
import json
import time
import numpy as np
import torch
from datetime import datetime

# Add paths
V9_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, V9_PATH)
sys.path.insert(0, os.path.join(V9_PATH, '..', 'continuation_stage_v8'))

from canonical_node.data_loader_v9 import load_7d_data, get_test_segments
from canonical_node.config_v9 import PHYSICAL_LIMITS, N_EVAL_SEGMENTS, SEGMENT_LENGTH, ROLLOUT_HORIZONS
from canonical_node.neural_ode_v12 import NeuralODEV12, V12Config
from canonical_node.evaluation_v9 import multi_step_evaluate

HORIZONS = ROLLOUT_HORIZONS
DT = 1/30
N_SEGMENTS = N_EVAL_SEGMENTS
SEG_LEN = SEGMENT_LENGTH


def prepare_data(data: dict, segments: list) -> dict:
    """Prepare training/validation data with episode boundaries."""
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

    # Validation: last 10% of training data
    n_val = max(int(len(train_states) * 0.1), 1000)
    val_states = train_states[-n_val:]
    val_actions = train_actions[-n_val:]
    val_deltas = train_deltas[-n_val:]
    train_states = train_states[:-n_val]
    train_actions = train_actions[:-n_val]
    train_deltas = train_deltas[:-n_val]

    # Clip boundaries to training length
    train_len = len(train_states)
    episode_boundaries = [(s, min(e, train_len))
                          for s, e in episode_boundaries
                          if s < train_len and e > 0 and min(e, train_len) - s >= 2]

    return {
        'train_states': train_states,
        'train_actions': train_actions,
        'train_deltas': train_deltas,
        'val_states': val_states,
        'val_actions': val_actions,
        'val_deltas': val_deltas,
        'episode_boundaries': episode_boundaries,
    }


def run_experiment(name, config, split_data, state_std, action_std, delta_std,
                   segments, seed=43):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"  [{name}] seed={seed}")
    print(f"{'='*60}")

    config.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = NeuralODEV12(config)

    t0 = time.time()
    model.train(
        split_data['train_states'],
        split_data['train_actions'],
        split_data['train_deltas'],
        state_std,
        action_std,
        delta_std,
        val_states=split_data['val_states'],
        val_actions=split_data['val_actions'],
        val_deltas=split_data['val_deltas'],
        episode_boundaries=split_data.get('episode_boundaries'),
    )
    train_time = time.time() - t0

    eval_result = multi_step_evaluate(model, segments, state_std, HORIZONS)

    ckpt_dir = os.path.join(V9_PATH, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'ADV_{name}_seed{seed}.pt')
    model.save(ckpt_path)

    # Extract results per horizon
    nmae = {}
    survival = {}
    for h in HORIZONS:
        nmae[h] = eval_result[h]['nmae_mean']
        survival[h] = eval_result[h]['survival_rate']

    print(f"\n  [{name}] Results:")
    print(f"  {'Horizon':>8} {'NMAE':>10} {'Survival':>10}")
    print(f"  {'-'*30}")
    for h in HORIZONS:
        print(f"  {h:>8} {nmae[h]:>10.4f} {survival[h]:>10.0%}")

    return {
        'name': name,
        'seed': seed,
        'nmae': {str(k): v for k, v in nmae.items()},
        'survival': {str(k): v for k, v in survival.items()},
        'train_time': train_time,
        'checkpoint': ckpt_path,
    }


def main():
    print("=" * 60)
    print("  V9 Advanced Repair Experiments")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading data...", flush=True)
    data = load_7d_data()
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']
    print(f"  Train: {data['train_states'].shape}, Test: {data['test_states'].shape}")

    print("[2/3] Extracting test segments...", flush=True)
    segments = get_test_segments(data, n_segments=N_SEGMENTS, segment_length=SEG_LEN)
    print(f"  Got {len(segments)} segments, each {segments[0]['states'].shape[0]} steps")

    split_data = prepare_data(data, segments)
    print(f"  Train: {split_data['train_states'].shape[0]}, Val: {split_data['val_states'].shape[0]}")

    results = []

    # --------------------------------------------------------
    # Exp 1: Multi-seed WITH contractivity (lambda=0.05)
    # --------------------------------------------------------
    print("\n[3/3] Running experiments...", flush=True)

    contract_config = V12Config(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        n_epochs=200, batch_size=256, dt=DT,
        rollout_curriculum='1,5,10,20',
        integration_method='euler',
        use_adaptive_curriculum=False,
        use_contractivity=True,
        lambda_contractive=0.05,
        contractive_warmup_epochs=20,
        contractive_ramp_epochs=30,
        max_segment_len=20,
    )

    for seed in [43, 44, 45]:
        r = run_experiment(
            "contract_005", contract_config,
            split_data, state_std, action_std, delta_std,
            segments, seed=seed
        )
        results.append(r)

    # --------------------------------------------------------
    # Exp 2: Multi-seed WITHOUT contractivity
    # --------------------------------------------------------
    no_contract_config = V12Config(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        n_epochs=200, batch_size=256, dt=DT,
        rollout_curriculum='1,5,10,20',
        integration_method='euler',
        use_adaptive_curriculum=False,
        use_contractivity=False,
        max_segment_len=20,
    )

    for seed in [43, 44, 45]:
        r = run_experiment(
            "no_contract", no_contract_config,
            split_data, state_std, action_std, delta_std,
            segments, seed=seed
        )
        results.append(r)

    # --------------------------------------------------------
    # Exp 3: Extended curriculum (400 epochs, H=50)
    # --------------------------------------------------------
    ext_config = V12Config(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        n_epochs=400, batch_size=256, dt=DT,
        rollout_curriculum='1,5,10,20,50',
        integration_method='euler',
        use_adaptive_curriculum=False,
        use_contractivity=True,
        lambda_contractive=0.05,
        contractive_warmup_epochs=40,
        contractive_ramp_epochs=60,
        max_segment_len=50,
    )

    r = run_experiment(
        "ext_400ep_h50", ext_config,
        split_data, state_std, action_std, delta_std,
        segments, seed=43
    )
    results.append(r)

    # --------------------------------------------------------
    # Exp 4: Weak contractivity (lambda=0.02)
    # --------------------------------------------------------
    weak_config = V12Config(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        n_epochs=200, batch_size=256, dt=DT,
        rollout_curriculum='1,5,10,20',
        integration_method='euler',
        use_adaptive_curriculum=False,
        use_contractivity=True,
        lambda_contractive=0.02,
        contractive_warmup_epochs=20,
        contractive_ramp_epochs=30,
        max_segment_len=20,
    )

    r = run_experiment(
        "weak_contract_002", weak_config,
        split_data, state_std, action_std, delta_std,
        segments, seed=43
    )
    results.append(r)

    # --------------------------------------------------------
    # Exp 5: Strong contractivity (lambda=0.1)
    # --------------------------------------------------------
    strong_config = V12Config(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        n_epochs=200, batch_size=256, dt=DT,
        rollout_curriculum='1,5,10,20',
        integration_method='euler',
        use_adaptive_curriculum=False,
        use_contractivity=True,
        lambda_contractive=0.10,
        contractive_warmup_epochs=20,
        contractive_ramp_epochs=30,
        max_segment_len=20,
    )

    r = run_experiment(
        "strong_contract_010", strong_config,
        split_data, state_std, action_std, delta_std,
        segments, seed=43
    )
    results.append(r)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiments': results,
    }

    # Summary table
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Config':<30} {'Seed':<6} {'H=1':<8} {'H=10':<8} {'H=50':<8} {'H=100':<8} {'H=200':<8} {'H=500':<8} {'Surv500':<8}")
    print("-" * 90)

    for r in results:
        nmae = r['nmae']
        surv = r['survival']
        print(f"{r['name']:<30} {r['seed']:<6} "
              f"{nmae.get('1',0):.3f}  {nmae.get('10',0):.3f}  "
              f"{nmae.get('50',0):.3f}  {nmae.get('100',0):.3f}  "
              f"{nmae.get('200',0):.3f}  {nmae.get('500',0):.3f}  "
              f"{surv.get('500',0):.1%}")

    # Comparison
    print("\n" + "=" * 90)
    print("COMPARISON WITH V9 BASELINE (H=50: 0.548, H=500: 0.998, Surv: 100%)")
    print("=" * 90)
    baseline = {'1': 0.006, '10': 0.066, '50': 0.548, '100': 0.672, '200': 0.719, '500': 0.998}

    for r in results:
        nmae = r['nmae']
        surv = r['survival']
        print(f"\n  {r['name']} (seed={r['seed']}):")
        for h in ['50', '200', '500']:
            if h in nmae:
                imp = (baseline[h] - nmae[h]) / baseline[h] * 100
                status = "BETTER" if imp > 0 else "WORSE"
                print(f"    H={h}: {nmae[h]:.3f} vs {baseline[h]:.3f} ({imp:+.1f}% {status})")
        s500 = surv.get('500', 0)
        print(f"    Surv@500: {s500:.1%}")

    os.makedirs(os.path.join(V9_PATH, 'raw_results'), exist_ok=True)
    output_path = os.path.join(V9_PATH, 'raw_results', 'ADVANCED_EXPERIMENTS.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
