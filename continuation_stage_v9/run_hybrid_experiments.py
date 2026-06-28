"""Hybrid Experiments: Blend V12 sequential and contractive models.

Key insight from repair experiments:
  - v9_fixed_seq: NMAE excellent (H=50: 0.385, H=500: 0.476) but 0% survival
  - v9_fixed_seq_contract: NMAE worse (H=50: 0.647, H=500: 0.958) but 100% survival
  - Contractivity trades accuracy for stability

Hypothesis: Blending predictions from both models can achieve
better NMAE than contractive alone while maintaining 100% survival.

Configs:
  1. blend_0.5     -- 50% seq + 50% contract
  2. blend_0.7_seq -- 70% seq + 30% contract
  3. blend_0.3_seq -- 30% seq + 70% contract
  4. blend_adaptive -- Use seq for H<50, contract for H>=50

Output: raw_results/HYBRID_EXPERIMENTS.json
"""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')

import os
import json
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

from canonical_node.config_v9 import STATE_NAMES_7D, PHYSICAL_LIMITS
from canonical_node.data_loader_v9 import load_7d_data, get_test_segments
from canonical_node.neural_ode_v12 import NeuralODEV12, V12Config
from canonical_node.evaluation_v9 import multi_step_evaluate, check_survival

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
V9_PATH = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9'
RESULTS_PATH = os.path.join(V9_PATH, 'raw_results')
RESULTS_FILE = os.path.join(RESULTS_PATH, 'HYBRID_EXPERIMENTS.json')
PREV_RESULTS_FILE = os.path.join(RESULTS_PATH, 'REPAIR_EXPERIMENTS.json')

HORIZONS = [1, 10, 50, 100, 200, 500]
N_SEGMENTS = 5
SEG_LEN = 1100


# ---------------------------------------------------------------------------
# Blended Model
# ---------------------------------------------------------------------------

class BlendedModel:
    """Blends predictions from two models: s_next = alpha * s_seq + (1-alpha) * s_contract."""

    def __init__(self, model_seq, model_contract, alpha: float):
        self._seq = model_seq
        self._contract = model_contract
        self._alpha = alpha

    def predict(self, s, tau):
        s_seq = self._seq.predict(s, tau)
        s_contract = self._contract.predict(s, tau)
        return self._alpha * s_seq + (1 - self._alpha) * s_contract

    def name(self):
        return f"blend({self._alpha:.1f})"


class AdaptiveBlendedModel:
    """Uses sequential model for short horizons, contractive for long.

    Switches at a threshold horizon. For rollout, we track step count.
    """

    def __init__(self, model_seq, model_contract, switch_horizon: int = 50):
        self._seq = model_seq
        self._contract = model_contract
        self._switch_h = switch_horizon
        self._step = 0

    def predict(self, s, tau):
        if self._step < self._switch_h:
            result = self._seq.predict(s, tau)
        else:
            result = self._contract.predict(s, tau)
        self._step += 1
        return result

    def reset(self):
        self._step = 0

    def name(self):
        return f"adaptive(switch@{self._switch_h})"


class SafetyBlendedModel:
    """Uses sequential model normally, switches to contractive when near limits.

    Checks if current state is within `safety_margin` fraction of physical limits.
    If so, uses contractive model (safer). Otherwise uses sequential model (more accurate).
    """

    def __init__(self, model_seq, model_contract, safety_margin: float = 0.7):
        self._seq = model_seq
        self._contract = model_contract
        self._margin = safety_margin
        self._limits = np.array([
            PHYSICAL_LIMITS['e_y'],
            PHYSICAL_LIMITS['e_psi'],
            PHYSICAL_LIMITS['v'],
            PHYSICAL_LIMITS['theta'],
            PHYSICAL_LIMITS['theta_dot'],
            PHYSICAL_LIMITS['delta'],
            PHYSICAL_LIMITS['delta_dot'],
        ])
        self._thresholds = self._limits * safety_margin

    def predict(self, s, tau):
        # Check if any state dimension is near its limit
        near_limit = np.any(np.abs(s) > self._thresholds)
        if near_limit:
            return self._contract.predict(s, tau)
        else:
            return self._seq.predict(s, tau)

    def name(self):
        return f"safety_blended(margin={self._margin})"


# ---------------------------------------------------------------------------
# Load models from checkpoints
# ---------------------------------------------------------------------------

def load_v12_model(ckpt_name: str) -> NeuralODEV12:
    """Load a V12 model from checkpoint."""
    ckpt_path = os.path.join(V9_PATH, 'checkpoints', ckpt_name)
    config = V12Config(
        hidden=64, depth=3, activation='tanh', lr=1e-3,
        n_epochs=200, seed=43,
        rollout_curriculum='1,5,10,20',
        integration_method='euler',
        max_segment_len=20,
    )
    model = NeuralODEV12(config)
    model.load(ckpt_path)
    return model


# ---------------------------------------------------------------------------
# Custom evaluation for adaptive model
# ---------------------------------------------------------------------------

def evaluate_adaptive_model(model, segments, state_std, horizons,
                             survival_mode='physical'):
    """Evaluate with step counter reset per horizon."""
    from canonical_node.evaluation_v9 import compute_nmae, compute_survival
    results = {}

    for h in horizons:
        nmae_list = []
        survival_list = []
        per_state_nmae = {name: [] for name in STATE_NAMES_7D}

        for seg in segments:
            s0 = seg['states'][0].copy()
            actions_seg = seg['actions']
            real_states = seg['states']

            n = min(h, len(actions_seg))
            predicted = [s0.copy()]
            s_cur = s0.copy()
            survived = True

            # Reset step counter for adaptive model
            if hasattr(model, 'reset'):
                model.reset()

            for step in range(n):
                try:
                    s_next = model.predict(s_cur, actions_seg[step])
                    if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                        survived = False
                        break
                    if not check_survival(s_next, mode=survival_mode):
                        survived = False
                        break
                    predicted.append(s_next.copy())
                    s_cur = s_next
                except Exception:
                    survived = False
                    break

            predicted = np.array(predicted)
            real = real_states[:len(predicted)]
            n_valid = min(len(predicted), len(real)) - 1

            if n_valid > 0:
                nmae = compute_nmae(predicted[1:n_valid+1], real[1:n_valid+1], state_std)
                nmae_list.append(nmae['overall'])
                for name in STATE_NAMES_7D:
                    per_state_nmae[name].append(nmae[name])
            else:
                nmae_list.append(float('nan'))

            survival_list.append(survived and n_valid >= n - 1)

        valid_nmae = [x for x in nmae_list if not np.isnan(x)]
        results[h] = {
            'nmae_mean': float(np.nanmean(valid_nmae)) if valid_nmae else float('nan'),
            'nmae_std': float(np.nanstd(valid_nmae)) if valid_nmae else float('nan'),
            'survival_rate': compute_survival(survival_list),
            'n_valid': len(valid_nmae),
            'per_state_nmae': {
                name: {
                    'mean': float(np.nanmean(per_state_nmae[name])) if per_state_nmae[name] else float('nan'),
                    'std': float(np.nanstd(per_state_nmae[name])) if per_state_nmae[name] else float('nan'),
                }
                for name in STATE_NAMES_7D
            },
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("="*60)
    print("  Hybrid Experiments")
    print("  Blending sequential + contractive models")
    print("="*60)

    # Load data
    print("\n[1/3] Loading data...", flush=True)
    data = load_7d_data()
    print(f"  Train: {data['train_states'].shape}, Test: {data['test_states'].shape}")

    # Get test segments
    print("[2/3] Extracting test segments...", flush=True)
    segments = get_test_segments(data, n_segments=N_SEGMENTS, segment_length=SEG_LEN)
    print(f"  Got {len(segments)} segments, each {segments[0]['states'].shape[0]} steps")

    # Load models
    print("[3/3] Loading models...", flush=True)
    model_seq = load_v12_model('V9_FIXED_v9_fixed_seq.pt')
    model_contract = load_v12_model('V9_FIXED_v9_fixed_seq_contract.pt')
    print("  Loaded v9_fixed_seq and v9_fixed_seq_contract checkpoints")

    state_std = data['state_std']

    # Load previous results
    all_results = {}
    for f in [PREV_RESULTS_FILE, os.path.join(RESULTS_PATH, 'V9_FIXED_RERUN.json')]:
        if os.path.exists(f):
            with open(f) as fh:
                prev = json.load(fh)
                all_results.update(prev)

    # Define hybrid experiments
    experiments = [
        ('blend_0.5', BlendedModel(model_seq, model_contract, alpha=0.5)),
        ('blend_0.7_seq', BlendedModel(model_seq, model_contract, alpha=0.7)),
        ('blend_0.3_seq', BlendedModel(model_seq, model_contract, alpha=0.3)),
        ('blend_0.9_seq', BlendedModel(model_seq, model_contract, alpha=0.9)),
        ('safety_blend_0.7', SafetyBlendedModel(model_seq, model_contract, safety_margin=0.7)),
        ('safety_blend_0.5', SafetyBlendedModel(model_seq, model_contract, safety_margin=0.5)),
        ('safety_blend_0.3', SafetyBlendedModel(model_seq, model_contract, safety_margin=0.3)),
    ]

    for name, model in experiments:
        print(f"\n{'='*60}")
        print(f"  [{name}] {model.name()}")
        print(f"{'='*60}", flush=True)

        if isinstance(model, AdaptiveBlendedModel):
            eval_result = evaluate_adaptive_model(model, segments, state_std, HORIZONS)
        else:
            eval_result = multi_step_evaluate(model, segments, state_std, HORIZONS)

        nmae = {}
        survival = {}
        per_state_nmae = {}
        for h in HORIZONS:
            nmae[h] = eval_result[h]['nmae_mean']
            survival[h] = eval_result[h]['survival_rate']
            per_state_nmae[h] = eval_result[h]['per_state_nmae']

        result = {
            'nmae': {str(k): v for k, v in nmae.items()},
            'per_state_nmae': {str(k): v for k, v in per_state_nmae.items()},
            'survival_rate': {str(k): v for k, v in survival.items()},
        }
        all_results[name] = result

        print(f"\n  [{name}] Results:")
        print(f"  {'Horizon':>8} {'NMAE':>10} {'Survival':>10}")
        print(f"  {'-'*30}")
        for h in HORIZONS:
            print(f"  {h:>8} {nmae[h]:>10.4f} {survival[h]:>10.0%}")

    # Save
    os.makedirs(RESULTS_PATH, exist_ok=True)
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  [SAVED] {RESULTS_FILE}")

    # Comparison table
    print("\n" + "="*90)
    print("  FULL COMPARISON TABLE")
    print("="*90)
    print(f"  {'Config':>25} {'H=1':>8} {'H=10':>8} {'H=50':>8} {'H=100':>8} {'H=200':>8} {'H=500':>8} {'S@500':>6}")
    print(f"  {'-'*90}")

    key_configs = ['v9_baseline_ref', 'v9_fixed_seq', 'v9_fixed_seq_contract',
                   'soft_contract', 'blend_0.5', 'blend_0.7_seq', 'blend_0.3_seq',
                   'blend_0.9_seq', 'safety_blend_0.7', 'safety_blend_0.5', 'safety_blend_0.3']
    for name in key_configs:
        if name not in all_results:
            continue
        r = all_results[name]
        nmae_vals = r['nmae']
        surv_vals = r['survival_rate']
        h1 = nmae_vals.get('1', nmae_vals.get(1, float('nan')))
        h10 = nmae_vals.get('10', nmae_vals.get(10, float('nan')))
        h50 = nmae_vals.get('50', nmae_vals.get(50, float('nan')))
        h100 = nmae_vals.get('100', nmae_vals.get(100, float('nan')))
        h200 = nmae_vals.get('200', nmae_vals.get(200, float('nan')))
        h500 = nmae_vals.get('500', nmae_vals.get(500, float('nan')))
        s500 = surv_vals.get('500', surv_vals.get(500, float('nan')))
        print(f"  {name:>25} {h1:>8.4f} {h10:>8.4f} {h50:>8.4f} {h100:>8.4f} {h200:>8.4f} {h500:>8.4f} {s500:>5.0%}")

    # Find best config
    print("\n" + "="*90)
    print("  BEST CONFIG ANALYSIS")
    print("="*90)

    best_configs = {}
    for name in all_results:
        r = all_results[name]
        s500 = r['survival_rate'].get('500', r['survival_rate'].get(500, 0))
        if s500 < 1.0:
            continue  # Skip configs that don't achieve 100% survival
        h50 = r['nmae'].get('50', r['nmae'].get(50, float('inf')))
        h500 = r['nmae'].get('500', r['nmae'].get(500, float('inf')))
        h1 = r['nmae'].get('1', r['nmae'].get(1, float('inf')))
        # Composite score: weighted average of NMAE at key horizons
        score = 0.2 * h1 + 0.4 * h50 + 0.4 * h500
        best_configs[name] = {'score': score, 'h1': h1, 'h50': h50, 'h500': h500, 'surv': s500}

    if best_configs:
        sorted_configs = sorted(best_configs.items(), key=lambda x: x[1]['score'])
        print(f"\n  Ranked by composite score (0.2*H1 + 0.4*H50 + 0.4*H500):")
        for i, (name, info) in enumerate(sorted_configs[:5]):
            print(f"  {i+1}. {name:>25} score={info['score']:.4f} "
                  f"(H1={info['h1']:.4f}, H50={info['h50']:.4f}, H500={info['h500']:.4f}, Surv={info['surv']:.0%})")

    print(f"\nDone. Results saved to {RESULTS_FILE}")
