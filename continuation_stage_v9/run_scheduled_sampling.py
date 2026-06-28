"""Scheduled Sampling Experiment for V12 Neural ODE.

Tests the hypothesis that reducing train-eval gap via scheduled sampling
can improve long-horizon prediction while maintaining stability.

Scheduled Sampling strategy:
- During single-step training, with probability p_use_pred, replace the
  input state with a one-step model prediction from the previous state
- p_use_pred starts at 0 (pure teacher forcing) and increases linearly
  to p_max over training
- This trains the model to handle its own errors, bridging the gap
  between single-step training and multi-step evaluation

References:
- Bengio et al. 2015: Scheduled Sampling for Sequence Prediction
- DreamerV3 (Hafner 2023): Uses similar strategy for world model training
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

# Add paths
V9_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, V9_PATH)
sys.path.insert(0, os.path.join(V9_PATH, '..', 'continuation_stage_v8'))

from canonical_node.data_loader_v9 import load_7d_data, get_test_segments
from canonical_node.config_v9 import PHYSICAL_LIMITS, N_EVAL_SEGMENTS, SEGMENT_LENGTH, ROLLOUT_HORIZONS
from canonical_node.neural_ode_v12 import (
    NeuralODEV12, V12Config, ODEFunc,
    SequentialSegmentDataset, Segment,
    _integrate, _contractivity_loss,
    _multi_step_rollout_loss,
    _AdaptiveCurriculumSchedulerV12,
)
from canonical_node.evaluation_v9 import multi_step_evaluate

HORIZONS = ROLLOUT_HORIZONS
DT = 1/30
N_SEGMENTS = N_EVAL_SEGMENTS
SEG_LEN = SEGMENT_LENGTH
STATE_DIM = 7
ACTION_DIM = 1


@dataclass
class ScheduledSamplingConfig(V12Config):
    """V12 config with scheduled sampling support."""
    # Scheduled sampling parameters
    use_scheduled_sampling: bool = True
    ss_start_prob: float = 0.0      # Initial probability of using model prediction
    ss_end_prob: float = 0.5        # Final probability
    ss_warmup_epochs: int = 50      # Epochs to ramp from start to end
    ss_noise_scale: float = 0.01    # Scale of noise added when using teacher forcing

    # Override defaults for this experiment
    n_epochs: int = 200
    rollout_curriculum: str = '1,5,10,20'
    max_segment_len: int = 20
    use_adaptive_curriculum: bool = False


class ScheduledSamplingNeuralODE(NeuralODEV12):
    """Neural ODE with Scheduled Sampling for reduced train-eval gap."""

    def __init__(self, config: ScheduledSamplingConfig):
        super().__init__(config)
        self.ss_config = config

    def _get_ss_prob(self, epoch: int) -> float:
        """Get scheduled sampling probability for current epoch."""
        if not self.ss_config.use_scheduled_sampling:
            return 0.0
        if epoch < self.ss_config.ss_warmup_epochs:
            # Linear ramp from start to end
            alpha = epoch / self.ss_config.ss_warmup_epochs
            return self.ss_config.ss_start_prob + alpha * (
                self.ss_config.ss_end_prob - self.ss_config.ss_start_prob
            )
        return self.ss_config.ss_end_prob

    def train(self, states: np.ndarray, actions: np.ndarray, deltas: np.ndarray,
              state_std: np.ndarray, action_std: float, delta_std: np.ndarray,
              val_states: Optional[np.ndarray] = None,
              val_actions: Optional[np.ndarray] = None,
              val_deltas: Optional[np.ndarray] = None,
              episode_boundaries: Optional[List[Tuple[int, int]]] = None) -> None:
        """Train with scheduled sampling.

        Key modification: During single-step loss computation, with probability
        ss_prob, replace input states with model's own predictions (plus noise).
        """
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        cfg = self.config
        ss_cfg = self.ss_config

        # Store normalization parameters
        self._state_std = state_std.copy()
        self._action_std = action_std
        self._delta_std = delta_std.copy()
        self._state_std[self._state_std < 1e-10] = 1.0
        self._delta_std[self._delta_std < 1e-10] = 1.0
        self._std_ratio = float(state_std[4] / state_std[3]) if state_std[3] > 1e-10 else 1.0

        # Normalize training data
        train_s = states / self._state_std
        train_a = actions.reshape(-1, 1) / self._action_std
        train_dsdot = deltas / self._state_std

        # Build dataset
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

        # Initialize model
        self._model = ODEFunc(cfg.hidden, cfg.depth, cfg.activation)

        # Optimizer
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

        print(f"    Scheduled Sampling: start_prob={ss_cfg.ss_start_prob}, "
              f"end_prob={ss_cfg.ss_end_prob}, warmup={ss_cfg.ss_warmup_epochs}")

        # ---- Training loop ----
        self._model.train()
        for epoch in range(cfg.n_epochs):
            epoch_loss = 0.0
            epoch_single = 0.0
            epoch_multi = 0.0
            epoch_contractive = 0.0
            n_batches = 0

            rollout_steps = curriculum.get_rollout_steps(epoch)
            lambda_multi = curriculum.get_lambda_multi(epoch)
            eff_lambda_con = self._get_contractive_lambda(epoch)
            ss_prob = self._get_ss_prob(epoch)

            dataset.set_max_horizon(max(rollout_steps, 2))
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
            )

            # Validation for adaptive curriculum
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

                # === SCHEDULED SAMPLING SINGLE-STEP LOSS ===
                # Instead of using ground truth states directly, sometimes use
                # model's own predictions as input (with noise)
                s_flat = states_batch[:, :H_single].reshape(-1, STATE_DIM)
                a_flat = actions_batch.reshape(-1, ACTION_DIM)
                y_flat = deltas_batch.reshape(-1, STATE_DIM)

                if ss_prob > 0 and H_single > 1:
                    # Reshape to [B, H, STATE_DIM] for sequential processing
                    s_seq = states_batch[:, :H_single]  # [B, H, 7]
                    a_seq = actions_batch  # [B, H, 1]

                    # For each step, with probability ss_prob, use model prediction
                    pred_list = []
                    target_list = []
                    s_cur = s_seq[:, 0:1]  # [B, 1, 7] - first state always ground truth

                    for step in range(H_single):
                        s_input = s_cur[:, 0]  # [B, 7]
                        a_input = a_seq[:, step]  # [B, 1]

                        # Model prediction for this step
                        pred_step = self._model(s_input, a_input)  # [B, 7]
                        pred_list.append(pred_step)
                        target_list.append(deltas_batch[:, step])

                        # Next state: with prob ss_prob use model prediction,
                        # with prob (1-ss_prob) use ground truth
                        s_next_gt = s_seq[:, step + 1:step + 2] if step + 1 < H_single else None

                        if s_next_gt is not None:
                            # Model's predicted next state
                            s_next_pred = (s_input + pred_step).unsqueeze(1)  # [B, 1, 7]

                            # Random mask for which samples use prediction
                            use_pred = torch.rand(B, 1, 1) < ss_prob  # [B, 1, 1]
                            use_pred = use_pred.float().to(s_input.device)

                            # Blend: use_pred * s_next_pred + (1-use_pred) * s_next_gt
                            s_cur = use_pred * s_next_pred + (1 - use_pred) * s_next_gt

                            # Add small noise to ground truth to improve robustness
                            if ss_cfg.ss_noise_scale > 0:
                                noise = torch.randn_like(s_cur) * ss_cfg.ss_noise_scale
                                s_cur = s_cur + noise * (1 - use_pred)  # Only add noise to GT

                    pred = torch.stack(pred_list, dim=1).reshape(-1, STATE_DIM)  # [B*H, 7]
                    y = torch.stack(target_list, dim=1).reshape(-1, STATE_DIM)  # [B*H, 7]
                    loss_single = nn.functional.mse_loss(pred, y)

                    # Also need to flatten s_flat for consistency/jacobian losses
                    s_flat = s_seq.reshape(-1, STATE_DIM)
                    a_flat = a_seq.reshape(-1, ACTION_DIM)
                    y_flat = deltas_batch.reshape(-1, STATE_DIM)
                    pred = self._model(s_flat, a_flat)  # Re-compute for consistency
                else:
                    # Standard single-step loss (no scheduled sampling)
                    pred = self._model(s_flat, a_flat)
                    loss_single = nn.functional.mse_loss(pred, y_flat)

                # === MULTI-STEP ROLLOUT LOSS (same as V12) ===
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

                # === CONSISTENCY LOSS ===
                loss_consistency = torch.tensor(0.0, device=s_flat.device)
                if cfg.lambda_consistency > 0 and hasattr(self, '_std_ratio'):
                    s_next_pred = s_flat + pred
                    theta_pred = s_next_pred[:, 3]
                    theta_gt = s_flat[:, 3] + s_flat[:, 4] * self._dt * self._std_ratio
                    loss_consistency = nn.functional.mse_loss(theta_pred, theta_gt)

                # === CONTRACTIVITY LOSS ===
                loss_contractive = torch.tensor(0.0, device=s_flat.device)
                if eff_lambda_con > 0:
                    n_con = min(32, len(s_flat))
                    s_con = s_flat[:n_con].requires_grad_(True)
                    a_con = a_flat[:n_con].requires_grad_(True)
                    loss_contractive, _ = _contractivity_loss(self._model, s_con, a_con)

                # === TOTAL LOSS ===
                loss = (
                    loss_single
                    + lambda_multi * loss_multi
                    + cfg.lambda_consistency * loss_consistency
                    + eff_lambda_con * loss_contractive
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()

                epoch_loss += loss.item()
                epoch_single += loss_single.item()
                epoch_multi += loss_multi.item()
                epoch_contractive += loss_contractive.item()
                n_batches += 1

            scheduler.step()

            if epoch % 50 == 0 or epoch == cfg.n_epochs - 1:
                print(f"    Epoch {epoch}: loss={epoch_loss/max(n_batches,1):.6f} "
                      f"(single={epoch_single/max(n_batches,1):.6f}, "
                      f"multi={epoch_multi/max(n_batches,1):.6f}), "
                      f"ss_prob={ss_prob:.3f}, rollout={rollout_steps}")

        self._model.eval()


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

    model = ScheduledSamplingNeuralODE(config)

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
    ckpt_path = os.path.join(ckpt_dir, f'SS_{name}_seed{seed}.pt')
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
    print("  Scheduled Sampling Experiments")
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
    # Exp 1: Scheduled Sampling (prob 0.0 -> 0.3) + contractivity
    # --------------------------------------------------------
    ss_config_1 = ScheduledSamplingConfig(
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
        # Scheduled sampling
        use_scheduled_sampling=True,
        ss_start_prob=0.0,
        ss_end_prob=0.3,
        ss_warmup_epochs=100,
        ss_noise_scale=0.01,
    )

    for seed in [43, 44, 45]:
        r = run_experiment("ss_03_contract", ss_config_1,
                          split_data, state_std, action_std, delta_std,
                          segments, seed=seed)
        results.append(r)

    # --------------------------------------------------------
    # Exp 2: Scheduled Sampling (prob 0.0 -> 0.5) + contractivity
    # --------------------------------------------------------
    ss_config_2 = ScheduledSamplingConfig(
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
        # Scheduled sampling - more aggressive
        use_scheduled_sampling=True,
        ss_start_prob=0.0,
        ss_end_prob=0.5,
        ss_warmup_epochs=80,
        ss_noise_scale=0.01,
    )

    for seed in [43, 44, 45]:
        r = run_experiment("ss_05_contract", ss_config_2,
                          split_data, state_std, action_std, delta_std,
                          segments, seed=seed)
        results.append(r)

    # --------------------------------------------------------
    # Exp 3: Scheduled Sampling (prob 0.0 -> 0.3) WITHOUT contractivity
    # --------------------------------------------------------
    ss_config_3 = ScheduledSamplingConfig(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        n_epochs=200, batch_size=256, dt=DT,
        rollout_curriculum='1,5,10,20',
        integration_method='euler',
        use_adaptive_curriculum=False,
        use_contractivity=False,
        max_segment_len=20,
        # Scheduled sampling
        use_scheduled_sampling=True,
        ss_start_prob=0.0,
        ss_end_prob=0.3,
        ss_warmup_epochs=100,
        ss_noise_scale=0.01,
    )

    for seed in [43, 44, 45]:
        r = run_experiment("ss_03_nocontract", ss_config_3,
                          split_data, state_std, action_std, delta_std,
                          segments, seed=seed)
        results.append(r)

    # --------------------------------------------------------
    # Exp 4: No scheduled sampling + contractivity (baseline comparison)
    # --------------------------------------------------------
    baseline_config = ScheduledSamplingConfig(
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
        # No scheduled sampling
        use_scheduled_sampling=False,
    )

    r = run_experiment("baseline_contract", baseline_config,
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
    print("COMPARISON WITH V9 BASELINE (H=50: 0.548, H=500: 0.998, Surv: 100%)")
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
    output_path = os.path.join(V9_PATH, 'raw_results', 'SCHEDULED_SAMPLING.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
