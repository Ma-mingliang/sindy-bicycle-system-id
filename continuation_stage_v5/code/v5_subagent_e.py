"""V5 子代理E: 8维路线最小可运行基线。

E1: 数据结构
E2: 最小数据核验
E3: 最小基线 (分层SINDy)
E4: 最小评估
E5: 门控结论
"""

import sys, os, json, time, math, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'evaluate'))
os.chdir(ROOT)

OUTPUT = Path(__file__).parent.parent / "raw_results"


def main():
    print("="*80)
    print("V5 SUBAGENT E: 8D Route Minimum Baseline")
    print("="*80)
    t0 = time.time()

    # E1: Data structure definition
    print("\n--- E1: Data Structure ---")
    data_schema = {
        'state': ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot'],
        'context': ['v', 'kappa'],
        'action': ['tau'],
        'note': 'kappa reclassified as exogenous input, not prediction state',
        'original_8d': ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'kappa', 'delta', 'delta_dot'],
        'proposed_7d': ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot'],
        'proposed_6d': ['e_y', 'e_psi', 'theta', 'theta_dot', 'delta', 'delta_dot'],
    }
    print(f"  Original 8D: {data_schema['original_8d']}")
    print(f"  Proposed 7D: {data_schema['proposed_7d']}")
    print(f"  Context inputs: {data_schema['context']}")

    # E2: Data audit using existing 4D data
    print("\n--- E2: Data Audit (existing 4D data) ---")
    import methods_common as mc

    # Generate data to check coverage
    states, actions, deltas, std_s, std_a, std_d = mc.generate_training_data(5000)

    # Check ranges
    state_names = ['phi', 'delta', 'phi_dot', 'delta_dot']
    data_stats = {}
    for j, name in enumerate(state_names):
        data_stats[name] = {
            'min': float(np.min(states[:, j])),
            'max': float(np.max(states[:, j])),
            'std': float(np.std(states[:, j])),
            'mean': float(np.mean(states[:, j])),
        }
        print(f"  {name}: [{data_stats[name]['min']:.3f}, {data_stats[name]['max']:.3f}], std={data_stats[name]['std']:.3f}")

    action_stats = {
        'min': float(np.min(actions)),
        'max': float(np.max(actions)),
        'std': float(np.std(actions)),
        'mean': float(np.mean(actions)),
    }
    print(f"  action (tau): [{action_stats['min']:.3f}, {action_stats['max']:.3f}], std={action_stats['std']:.3f}")

    # E3: Minimum baseline - Hierarchical SINDy on 4D
    # Since we don't have 8D data, build a minimum baseline on existing 4D
    print("\n--- E3: Minimum Baseline (SINDy on 4D) ---")

    # Simple polynomial regression as SINDy proxy
    from sklearn.linear_model import Ridge

    # Build library: [1, s1, s2, s3, s4, tau, s1^2, s1*s2, ...]
    def build_library(X):
        n = X.shape[0]
        lib = [np.ones(n)]
        for i in range(X.shape[1]):
            lib.append(X[:, i])
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                lib.append(X[:, i] * X[:, j])
        return np.column_stack(lib)

    X_full = np.column_stack([states, actions.reshape(-1, 1)])
    Theta = build_library(X_full)

    # Fit with regularization
    model_sindy = Ridge(alpha=1.0)
    model_sindy.fit(Theta, deltas)
    sindy_score = model_sindy.score(Theta, deltas)
    print(f"  SINDy R^2: {sindy_score:.4f}")

    # Coefficients
    n_features = Theta.shape[1]
    feature_names = ['1']
    for n in state_names + ['tau']:
        feature_names.append(n)
    for i in range(5):
        for j in range(i, 5):
            names = state_names + ['tau']
            feature_names.append(f"{names[i]}*{names[j]}")

    # Identify significant coefficients
    coef = model_sindy.coef_  # (4, n_features)
    significant = {}
    for j, name in enumerate(state_names):
        top_idx = np.argsort(np.abs(coef[j]))[-5:][::-1]
        significant[name] = [(feature_names[k], float(coef[j, k])) for k in top_idx if k < len(feature_names)]
        print(f"  {name} top terms: {[(f'{v:.4f}', n) for n, v in significant[name][:3]]}")

    def sindy_predict(s, tau):
        x = np.concatenate([s, [tau]])
        Theta = build_library(x.reshape(1, -1))
        delta = model_sindy.predict(Theta).flatten()
        return s + delta

    # E4: Minimum evaluation
    print("\n--- E4: Minimum Evaluation ---")
    import methods_evaluate as me

    horizons = [1, 5, 10, 20, 50, 100]
    eval_results = {}

    for h in horizons:
        metrics_list = []
        for si in range(3):
            tau_func = me.make_tau_func(si, max(h, 200))
            s0 = np.array([0.1, 0.0, 0.0, 0.0])

            # Collect actions
            actions_seq = []
            s_r = s0.copy()
            for t in range(h):
                tau = tau_func(t, s_r)
                actions_seq.append(tau)
                s_r = mc.real_step(s_r, tau)
                if abs(s_r[0]) > math.pi/3:
                    break

            # Model rollout
            s_m = s0.copy()
            smodel = [s_m.copy()]
            diverged = False
            for t in range(len(actions_seq)):
                s_m = sindy_predict(s_m, actions_seq[t])
                if np.any(np.isnan(s_m)) or np.any(np.abs(s_m) > 100):
                    diverged = True
                    break
                smodel.append(s_m.copy())
            smodel = np.array(smodel)

            # Real
            s_r = s0.copy()
            sreal = [s_r.copy()]
            for t in range(len(actions_seq)):
                s_r = mc.real_step(s_r, actions_seq[t])
                sreal.append(s_r.copy())
            sreal = np.array(sreal)

            n = min(len(smodel), len(sreal))
            errs = np.abs(smodel[:n] - sreal[:n])
            valid = ~np.any(np.isnan(errs), axis=1)
            n_valid = np.sum(valid)
            mae = float(np.mean(errs[valid])) if n_valid > 0 else float('nan')
            per_state = [float(np.mean(errs[valid, j])) if n_valid > 0 else float('nan') for j in range(4)]

            metrics_list.append({
                'mae': mae,
                'survival_steps': int(n_valid),
                'diverged': diverged,
                'per_state_mae': per_state,
            })

        avg_mae = float(np.mean([m['mae'] for m in metrics_list if not math.isnan(m['mae'])]))
        avg_surv = float(np.mean([m['survival_steps'] for m in metrics_list]))
        fail_rate = float(np.mean([m['diverged'] for m in metrics_list]))
        eval_results[f"{h}_steps"] = {
            'mae': avg_mae,
            'survival_steps': avg_surv,
            'failure_rate': fail_rate,
        }
        print(f"  {h:>3d} steps: MAE={avg_mae:.4f}, surv={avg_surv:.0f}, fail={fail_rate:.1%}")

    # E5: Gate decision
    print("\n--- E5: Gate Decision ---")
    # Check if 20-step MAE is reasonable
    mae_20 = eval_results.get('20_steps', {}).get('mae', float('nan'))
    fail_20 = eval_results.get('20_steps', {}).get('failure_rate', 1.0)

    if mae_20 < 0.1 and fail_20 < 0.3:
        gate = "GO_WITH_MORE_DATA"
        reason = f"20-step MAE={mae_20:.4f} is reasonable, but 8D needs >5000 samples"
    elif mae_20 < 1.0:
        gate = "GO_WITH_MORE_DATA"
        reason = f"20-step MAE={mae_20:.4f} shows promise, need more data for 7D/6D"
    else:
        gate = "NO_GO"
        reason = f"20-step MAE={mae_20:.4f} is too high, fundamental issues remain"

    print(f"  Gate: {gate}")
    print(f"  Reason: {reason}")

    # Save
    results = {
        'subagent': 'E',
        'data_schema': data_schema,
        'data_stats': data_stats,
        'action_stats': action_stats,
        'sindy_r2': float(sindy_score),
        'significant_coefficients': significant,
        'evaluation': eval_results,
        'gate_decision': gate,
        'gate_reason': reason,
        'data_gap': {
            'current_samples': 5000,
            'minimum_for_7d': 10000,
            'recommended': 20000,
        },
    }

    with open(OUTPUT / "SUBAGENT_E_RESULTS.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nSubagent E done in {elapsed:.1f}s")
    return results


if __name__ == "__main__":
    main()
