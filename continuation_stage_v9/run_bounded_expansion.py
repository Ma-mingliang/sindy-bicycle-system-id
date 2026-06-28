"""Bounded Expansion Constraint Experiment for V12 Neural ODE.

Tests the hypothesis that bounded expansion (allowing limited local instability)
can improve accuracy while maintaining stability, compared to global contractivity.

Key difference from contractivity:
- Contractivity: penalizes ALL positive eigenvalues (forces global contraction)
- Bounded expansion: only penalizes eigenvalues above a threshold beta (allows limited expansion)

L_expand = mean(ReLU(lambda_max(J_sym) - 2*beta))

This follows Bachar (2026) and Slotine's contraction theory, allowing the model
to express locally unstable dynamics (physically real) while preventing exponential divergence.

References:
- Bachar 2026: Residual analysis of Neural ODE stability
- Slotine: Contraction theory and bounded expansion
"""

import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Tuple

V9_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, V9_PATH)
sys.path.insert(0, os.path.join(V9_PATH, '..', 'continuation_stage_v8'))

from canonical_node.data_loader_v9 import load_7d_data, get_test_segments
from canonical_node.config_v9 import PHYSICAL_LIMITS, N_EVAL_SEGMENTS, SEGMENT_LENGTH, ROLLOUT_HORIZONS
from canonical_node.neural_ode_v12 import (
    NeuralODEV12, V12Config, ODEFunc,
    _integrate, _contractivity_loss,
    _multi_step_rollout_loss,
    _AdaptiveCurriculumSchedulerV12,
    compute_jacobian, max_eigenvalue_symmetric,
)
from canonical_node.evaluation_v9 import multi_step_evaluate

HORIZONS = ROLLOUT_HORIZONS
DT = 1/30
N_SEGMENTS = N_EVAL_SEGMENTS
SEG_LEN = SEGMENT_LENGTH
STATE_DIM = 7


def _bounded_expansion_loss(model: ODEFunc, s: torch.Tensor, a: torch.Tensor,
                           beta: float = 0.1):
    """Compute bounded expansion loss: penalize eigenvalues above threshold beta.

    Unlike contractivity (which penalizes ALL positive eigenvalues), this only
    penalizes eigenvalues that exceed beta, allowing limited local instability.

    L_expand = mean(ReLU(lambda_max(J_sym) - 2*beta))

    Args:
        model: ODEFunc dynamics network
        s: [batch, STATE_DIM] with requires_grad=True
        a: [batch, ACTION_DIM]
        beta: expansion bound (allow eigenvalues up to beta)

    Returns:
        loss: scalar bounded expansion loss
        max_eigs: max eigenvalue per sample [batch] (detached, for diagnostics)
    """
    batch_size = s.shape[0]
    max_eigs = torch.zeros(batch_size, device=s.device)

    for i in range(batch_size):
        s_i = s[i:i+1]
        a_i = a[i:i+1]
        jac = compute_jacobian(model, s_i, a_i)
        max_eig, _ = max_eigenvalue_symmetric(jac)
        max_eigs[i] = max_eig

    # Only penalize eigenvalues above beta
    loss = torch.relu(max_eigs - 2 * beta).mean()

    return loss, max_eigs.detach()


@dataclass
class BoundedExpansionConfig(V12Config):
    """V12 config with bounded expansion instead of global contractivity."""
    use_bounded_expansion: bool = True
    beta_expansion: float = 0.1  # Allow eigenvalues up to this bound
    lambda_bounded: float = 0.05
    bounded_warmup_epochs: int = 20
    bounded_ramp_epochs: int = 30

    # Override defaults
    use_contractivity: bool = False  # Disable global contractivity
    n_epochs: int = 200
    rollout_curriculum: str = '1,5,10,20'
    max_segment_len: int = 20
    use_adaptive_curriculum: bool = False


class BoundedExpansionNeuralODE(NeuralODEV12):
    """Neural ODE with Bounded Expansion Constraint."""

    def __init__(self, config: BoundedExpansionConfig):
        super().__init__(config)
        self.be_config = config

    def _get_bounded_lambda(self, epoch: int) -> float:
        """Get bounded expansion lambda with warmup."""
        cfg = self.be_config
        if not cfg.use_bounded_expansion:
            return 0.0
        if epoch < cfg.bounded_warmup_epochs:
            return 0.0
        ramp_epoch = epoch - cfg.bounded_warmup_epochs
        if ramp_epoch < cfg.bounded_ramp_epochs:
            alpha = ramp_epoch / cfg.bounded_ramp_epochs
            return cfg.lambda_bounded * alpha
        return cfg.lambda_bounded

    def train(self, states, actions, deltas, state_std, action_std, delta_std,
              val_states=None, val_actions=None, val_deltas=None,
              episode_boundaries=None):
        """Train with bounded expansion constraint."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        cfg = self.config
        be_cfg = self.be_config

        # Store normalization
        self._state_std = state_std.copy()
        self._action_std = action_std
        self._delta_std = delta_std.copy()
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0
        self._std_ratio = float(state_std[4] / state_std[3]) if state_std[3] > 1e-10 else 1.0

        # Normalize
        train_s = states / self._state_std
        train_a = actions.reshape(-1, 1) / self._action_std
        train_dsdot = deltas / self._state_std

        # Dataset
        dataset = self._build_dataset(train_s, train_a, train_dsdot, cfg.max_segment_len,
                                      episode_boundaries=episode_boundaries)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
        )

        # Validation
        has_val = val_states is not None and len(val_states) > 0
        if has_val:
            val_s = torch.FloatTensor(val_states / self._state_std)
            val_a = torch.FloatTensor(val_actions.reshape(-1, 1) / self._action_std)
            val_y = torch.FloatTensor(val_deltas / self._state_std)
        else:
            n_val = max(int(len(train_s) * 0.1), 64)
            val_s = torch.FloatTensor(train_s[-n_val:])
            val_a = torch.FloatTensor(train_a[-n_val:])
            val_y = torch.FloatTensor(train_dsdot[-n_val:])
            has_val = True

        # Model
        self._model = ODEFunc(cfg.hidden, cfg.depth, cfg.activation)
        opt = torch.optim.Adam(self._model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)

        # Curriculum
        curriculum_milestones = [int(x) for x in cfg.rollout_curriculum.split(',')]
        curriculum = _AdaptiveCurriculumSchedulerV12(
            thresholds_str=cfg.adaptive_thresholds,
            ema_decay=cfg.ema_decay,
            eval_interval=cfg.eval_interval,
            min_epochs_per_level=cfg.min_epochs_per_level,
            max_epochs_per_level=cfg.max_epochs_per_level,
            n_epochs=cfg.n_epochs,
            lambda_multi_start=cfg.lambda_multi_start,
            lambda_multi_end=cfg.lambda_multi_end,
            warmup_epochs=5,
            start_level=1,
        )

        print(f"    Bounded Expansion: beta={be_cfg.beta_expansion}, "
              f"lambda={be_cfg.lambda_bounded}")

        # Training loop
        self._model.train()
        for epoch in range(cfg.n_epochs):
            epoch_loss = 0.0
            epoch_single = 0.0
            epoch_multi = 0.0
            epoch_bounded = 0.0
            n_batches = 0

            rollout_steps = curriculum.get_rollout_steps(epoch)
            lambda_multi = curriculum.get_lambda_multi(epoch)
            eff_lambda_be = self._get_bounded_lambda(epoch)

            dataset.set_max_horizon(max(rollout_steps, 2))
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
            )

            # Validation
            validation_errors = {}
            if has_val and (epoch + 1) % cfg.eval_interval == 0:
                validation_errors = self._compute_validation_errors(
                    val_s, val_a, val_y, curriculum_milestones
                )
                curriculum.update(epoch, validation_errors)
            else:
                curriculum.update(epoch, None)

            for states_batch, actions_batch, deltas_batch in loader:
                B = states_batch.shape[0]
                H_single = states_batch.shape[1] - 1

                s_flat = states_batch[:, :H_single].reshape(-1, STATE_DIM)
                a_flat = actions_batch.reshape(-1, 1)
                y_flat = deltas_batch.reshape(-1, STATE_DIM)

                pred = self._model(s_flat, a_flat)
                loss_single = nn.functional.mse_loss(pred, y_flat)

                # Multi-step loss
                loss_multi = torch.tensor(0.0, device=s_flat.device)
                if rollout_steps > 1 and H_single >= rollout_steps:
                    n_roll_segs = min(B, 32)
                    loss_multi = torch.tensor(0.0, device=s_flat.device)
                    for seg_i in range(n_roll_segs):
                        seg_states = states_batch[seg_i]
                        seg_actions = actions_batch[seg_i]
                        seg_mse, _ = _multi_step_rollout_loss(
                            self._model, seg_states, seg_actions, self._dt,
                            rollout_steps, cfg.integration_method, cfg.sub_steps,
                        )
                        loss_multi = loss_multi + seg_mse
                    loss_multi = loss_multi / n_roll_segs

                # Consistency loss
                loss_consistency = torch.tensor(0.0, device=s_flat.device)
                if cfg.lambda_consistency > 0 and hasattr(self, '_std_ratio'):
                    s_next_pred = s_flat + pred
                    theta_pred = s_next_pred[:, 3]
                    theta_gt = s_flat[:, 3] + s_flat[:, 4] * self._dt * self._std_ratio
                    loss_consistency = nn.functional.mse_loss(theta_pred, theta_gt)

                # BOUNDED EXPANSION LOSS (replaces contractivity)
                loss_bounded = torch.tensor(0.0, device=s_flat.device)
                if eff_lambda_be > 0:
                    n_be = min(32, len(s_flat))
                    s_be = s_flat[:n_be].requires_grad_(True)
                    a_be = a_flat[:n_be].requires_grad_(True)
                    loss_bounded, _ = _bounded_expansion_loss(
                        self._model, s_be, a_be, beta=be_cfg.beta_expansion
                    )

                # Total loss
                loss = (
                    loss_single
                    + lambda_multi * loss_multi
                    + cfg.lambda_consistency * loss_consistency
                    + eff_lambda_be * loss_bounded
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()

                epoch_loss += loss.item()
                epoch_single += loss_single.item()
                epoch_multi += loss_multi.item()
                epoch_bounded += loss_bounded.item()
                n_batches += 1

            scheduler.step()

            if epoch % 50 == 0 or epoch == cfg.n_epochs - 1:
                print(f"    Epoch {epoch}: loss={epoch_loss/max(n_batches,1):.6f} "
                      f"(single={epoch_single/max(n_batches,1):.6f}, "
                      f"multi={epoch_multi/max(n_batches,1):.6f}, "
                      f"be={epoch_bounded/max(n_batches,1):.6f})")

        self._model.eval()


def prepare_data(data, segments):
    """Prepare training/validation data."""
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

    model = BoundedExpansionNeuralODE(config)

    t0 = time.time()
    model.train(
        split_data['train_states'],
        split_data['train_actions'],
        split_data['train_deltas'],
        state_std, action_std, delta_std,
        val_states=split_data['val_states'],
        val_actions=split_data['val_actions'],
        val_deltas=split_data['val_deltas'],
        episode_boundaries=split_data.get('episode_boundaries'),
    )
    train_time = time.time() - t0

    eval_result = multi_step_evaluate(model, segments, state_std, HORIZONS)

    ckpt_dir = os.path.join(V9_PATH, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'BE_{name}_seed{seed}.pt')
    model.save(ckpt_path)

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
    print("  Bounded Expansion Experiments")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n[1/3] Loading data...", flush=True)
    data = load_7d_data()
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    print("[2/3] Extracting test segments...", flush=True)
    segments = get_test_segments(data, n_segments=N_SEGMENTS, segment_length=SEG_LEN)

    split_data = prepare_data(data, segments)
    print(f"  Train: {split_data['train_states'].shape[0]}, Val: {split_data['val_states'].shape[0]}")

    results = []

    print("\n[3/3] Running experiments...", flush=True)

    # --------------------------------------------------------
    # Exp 1: Bounded expansion beta=0.1
    # --------------------------------------------------------
    be_config_1 = BoundedExpansionConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        n_epochs=200, batch_size=256, dt=DT,
        rollout_curriculum='1,5,10,20',
        integration_method='euler',
        use_adaptive_curriculum=False,
        max_segment_len=20,
        use_bounded_expansion=True,
        beta_expansion=0.1,
        lambda_bounded=0.05,
        bounded_warmup_epochs=20,
        bounded_ramp_epochs=30,
    )

    for seed in [43, 44, 45]:
        r = run_experiment("be_beta01", be_config_1,
                          split_data, state_std, action_std, delta_std,
                          segments, seed=seed)
        results.append(r)

    # --------------------------------------------------------
    # Exp 2: Bounded expansion beta=0.2 (more permissive)
    # --------------------------------------------------------
    be_config_2 = BoundedExpansionConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        n_epochs=200, batch_size=256, dt=DT,
        rollout_curriculum='1,5,10,20',
        integration_method='euler',
        use_adaptive_curriculum=False,
        max_segment_len=20,
        use_bounded_expansion=True,
        beta_expansion=0.2,
        lambda_bounded=0.05,
        bounded_warmup_epochs=20,
        bounded_ramp_epochs=30,
    )

    for seed in [43, 44, 45]:
        r = run_experiment("be_beta02", be_config_2,
                          split_data, state_std, action_std, delta_std,
                          segments, seed=seed)
        results.append(r)

    # --------------------------------------------------------
    # Exp 3: Bounded expansion beta=0.05 (more restrictive)
    # --------------------------------------------------------
    be_config_3 = BoundedExpansionConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        n_epochs=200, batch_size=256, dt=DT,
        rollout_curriculum='1,5,10,20',
        integration_method='euler',
        use_adaptive_curriculum=False,
        max_segment_len=20,
        use_bounded_expansion=True,
        beta_expansion=0.05,
        lambda_bounded=0.05,
        bounded_warmup_epochs=20,
        bounded_ramp_epochs=30,
    )

    r = run_experiment("be_beta005", be_config_3,
                      split_data, state_std, action_std, delta_std,
                      segments, seed=43)
    results.append(r)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiments': results,
    }

    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Config':<25} {'Seed':<6} {'H=1':<8} {'H=10':<8} {'H=50':<8} {'H=100':<8} {'H=500':<8} {'Surv500':<8}")
    print("-" * 90)

    for r in results:
        nmae = r['nmae']
        surv = r['survival']
        print(f"{r['name']:<25} {r['seed']:<6} "
              f"{nmae.get('1',0):.3f}  {nmae.get('10',0):.3f}  "
              f"{nmae.get('50',0):.3f}  {nmae.get('100',0):.3f}  "
              f"{nmae.get('500',0):.3f}  {surv.get('500',0):.1%}")

    baseline = {'1': 0.006, '10': 0.066, '50': 0.548, '100': 0.672, '200': 0.719, '500': 0.998}
    print("\n" + "=" * 90)
    print("COMPARISON WITH V9 BASELINE")
    print("=" * 90)

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
    output_path = os.path.join(V9_PATH, 'raw_results', 'BOUNDED_EXPANSION.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
