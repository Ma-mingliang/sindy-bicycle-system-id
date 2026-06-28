"""V5 子代理F: 统计复现与模型选择。

F1: 候选模型
F2: 统计设计 (5训练种子 x 3评估种子)
F3: 指标
F4: 模型选择 (用途化推荐)
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


def generate_data(n=5000, seed=42):
    import methods_common as mc
    return mc.generate_training_data(n, seed)


def train_gp(states, actions, deltas, std_s, std_a, std_d, max_samples=1500, seed=42):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    rng = np.random.RandomState(seed)
    n = len(states)
    idx = rng.choice(n, min(max_samples, n), replace=False)
    X = np.column_stack([states[idx]/std_s, actions[idx].reshape(-1,1)/std_a])
    Y = deltas[idx] / std_d
    gps = []
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    for col in range(4):
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-6)
        gp.fit(X, Y[:, col])
        gps.append(gp)
    return gps


def gp_pred(gps, s, tau, std_s, std_a, std_d):
    s_n = s / std_s
    a_n = tau / std_a
    x = np.concatenate([s_n, [a_n]]).reshape(1, -1)
    d_n = np.array([gp.predict(x)[0] for gp in gps])
    return s + d_n * std_d


def train_nn(models_list, gps, states, actions, deltas, std_s, std_a, std_d,
             n_models=5, n_epochs=50, dagger_rounds=0, seed_base=42, rs=0.3):
    import torch
    import methods_nn as nn_mod
    DEVICE = nn_mod.DEVICE
    ResidualNet = nn_mod.ResidualNet

    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next = gp_pred(gps, states[i], actions[i], std_s, std_a, std_d)
        residuals[i] = (deltas[i] - (s_next - states[i])) / std_d

    train_x = np.column_stack([states/std_s, actions.reshape(-1,1)/std_a])
    all_x = [train_x.copy()]
    all_r = [residuals.copy()]

    models = []
    actual_rounds = max(1, dagger_rounds)
    for rd in range(actual_rounds):
        tx = np.vstack(all_x)
        tr = np.vstack(all_r)
        for i in range(n_models):
            torch.manual_seed(seed_base + i * 42)
            np.random.seed(seed_base + i * 42)
            model = ResidualNet().to(DEVICE)
            ttx = torch.FloatTensor(tx).to(DEVICE)
            tty = torch.FloatTensor(tr).to(DEVICE)
            ds = torch.utils.data.TensorDataset(ttx, tty)
            loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            crit = torch.nn.MSELoss()
            model.train()
            for ep in range(n_epochs):
                for xb, yb in loader:
                    loss = crit(model(xb), yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            model.eval()
            models.append(model)

        if rd < dagger_rounds - 1:
            import methods_evaluate as me
            import methods_common as mc
            new_x, new_r = [], []
            for seg in range(2):
                tau_func = me.make_tau_func(300+seg, 200)
                phi0 = np.random.uniform(-0.2, 0.2)
                s0 = np.array([phi0, 0.0, 0.0, 0.0])
                s_base = s0.copy()
                s_real = s0.copy()
                for step in range(200):
                    tau = tau_func(step, s_real)
                    s_real = mc.real_step(s_real, tau)
                    s_n = s_base / std_s
                    a_n = tau / std_a
                    st = torch.FloatTensor(s_n).unsqueeze(0).to(DEVICE)
                    at = torch.FloatTensor([a_n]).unsqueeze(0).to(DEVICE)
                    preds = []
                    for m in models:
                        with torch.no_grad():
                            p = m(st, at).cpu().numpy()[0]
                        preds.append(p)
                    d_nn = np.mean(preds, axis=0)
                    s_base = gp_pred(gps, s_base, tau, std_s, std_a, std_d) + d_nn * std_d * rs
                    if abs(s_real[0]) > math.pi/3:
                        break
                    new_x.append(np.concatenate([s_n, [a_n]]))
                    s_for_gp = s_base - d_nn * std_d * rs
                    s_next_base = gp_pred(gps, s_for_gp, tau, std_s, std_a, std_d)
                    new_r.append(((s_real - s_for_gp) - (s_next_base - s_for_gp)) / std_d)
            if new_x:
                all_x.append(np.array(new_x))
                all_r.append(np.array(new_r))

    return models


def predict_with_nn(models, gps, s, tau, std_s, std_a, std_d, rs=0.3):
    import torch
    import methods_nn as nn_mod
    DEVICE = nn_mod.DEVICE
    s_next = gp_pred(gps, s, tau, std_s, std_a, std_d)
    s_n = s / std_s
    a_n = tau / std_a
    st = torch.FloatTensor(s_n).unsqueeze(0).to(DEVICE)
    at = torch.FloatTensor([a_n]).unsqueeze(0).to(DEVICE)
    preds = []
    for m in models:
        with torch.no_grad():
            p = m(st, at).cpu().numpy()[0]
        preds.append(p)
    d_nn = np.mean(preds, axis=0)
    return s_next + d_nn * std_d * rs


def evaluate_model(predict_fn, std_s, n_seeds=3, horizons=[10, 50, 100]):
    import methods_common as mc
    import methods_evaluate as me
    results = {}
    for h in horizons:
        metrics = []
        for si in range(n_seeds):
            tau_func = me.make_tau_func(si, max(h, 500))
            s0 = np.array([0.1, 0.0, 0.0, 0.0])
            actions_seq = []
            s_r = s0.copy()
            for t in range(h):
                tau = tau_func(t, s_r)
                actions_seq.append(tau)
                s_r = mc.real_step(s_r, tau)
                if abs(s_r[0]) > math.pi/3:
                    break
            s_m = s0.copy()
            smodel = [s_m.copy()]
            for t in range(len(actions_seq)):
                s_m = predict_fn(s_m, actions_seq[t])
                if np.any(np.isnan(s_m)) or np.any(np.abs(s_m) > 100):
                    break
                smodel.append(s_m.copy())
            smodel = np.array(smodel)
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
            nmae = float(np.mean(errs[valid]/std_s)) if n_valid > 0 else float('nan')
            per_state = [float(np.mean(errs[valid,j])) if n_valid > 0 else float('nan') for j in range(4)]
            metrics.append({'mae': mae, 'nmae': nmae, 'survival': int(n_valid),
                           'per_state': per_state})
        avg = {k: float(np.mean([m[k] for m in metrics if not math.isnan(m['mae'])]))
               for k in ['mae', 'nmae', 'survival']}
        avg['per_state'] = [float(np.mean([m['per_state'][j] for m in metrics if not math.isnan(m['mae'])]))
                           for j in range(4)]
        results[f'{h}_steps'] = avg
    return results


def main():
    print("="*80)
    print("V5 SUBAGENT F: Statistical Reproduction & Model Selection")
    print("="*80)
    t0 = time.time()

    states, actions, deltas, std_s, std_a, std_d = generate_data(5000)
    print(f"Data: {len(states)} samples")

    n_train_seeds = 3
    n_eval_seeds = 3
    horizons = [10, 50, 100]

    models_config = {
        'GP_pure': {'type': 'gp'},
        'GP_RDE_L': {'type': 'rde_l'},
        'GP_RDE_T': {'type': 'rde_t'},
        'GP_RDE_M': {'type': 'rde_m'},
    }

    all_results = {}

    for mname, mconf in models_config.items():
        print(f"\n--- {mname} ---")
        seed_results = []
        for train_seed in range(n_train_seeds):
            base_seed = train_seed * 100 + 42
            gps = train_gp(states, actions, deltas, std_s, std_a, std_d, seed=base_seed)

            if mconf['type'] == 'gp':
                pred_fn = lambda s, t, g=gps: gp_pred(g, s, t, std_s, std_a, std_d)
            else:
                dagger_map = {'rde_l': 0, 'rde_t': 1, 'rde_m': 2}
                nn_models = train_nn([], gps, states, actions, deltas, std_s, std_a, std_d,
                                     n_models=5, n_epochs=50,
                                     dagger_rounds=dagger_map[mconf['type']],
                                     seed_base=base_seed)
                pred_fn = lambda s, t, m=nn_models, g=gps: predict_with_nn(m, g, s, t, std_s, std_a, std_d)

            eval_r = evaluate_model(pred_fn, std_s, n_seeds=n_eval_seeds, horizons=horizons)
            seed_results.append(eval_r)

        # Aggregate across training seeds
        agg = {}
        for h in horizons:
            key = f'{h}_steps'
            maes = [sr[key]['mae'] for sr in seed_results if not math.isnan(sr[key]['mae'])]
            nmaes = [sr[key]['nmae'] for sr in seed_results if not math.isnan(sr[key]['nmae'])]
            survs = [sr[key]['survival'] for sr in seed_results]
            per_state = [sr[key]['per_state'] for sr in seed_results if not math.isnan(sr[key]['mae'])]
            agg[key] = {
                'mae_mean': float(np.mean(maes)) if maes else float('nan'),
                'mae_std': float(np.std(maes)) if maes else float('nan'),
                'nmae_mean': float(np.mean(nmaes)) if nmaes else float('nan'),
                'nmae_std': float(np.std(nmaes)) if nmaes else float('nan'),
                'survival_mean': float(np.mean(survs)),
                'per_state_mean': [float(np.mean([ps[j] for ps in per_state])) for j in range(4)] if per_state else [float('nan')]*4,
            }
        all_results[mname] = agg
        print(f"  100-step NMAE: {agg['100_steps']['nmae_mean']:.4f} +/- {agg['100_steps']['nmae_std']:.4f}")

    # Model selection by use case
    print("\n--- Model Selection ---")
    selection = {}

    # Short-term accuracy: lowest 10-step NMAE
    best_short = min(all_results.keys(), key=lambda k: all_results[k].get('10_steps', {}).get('nmae_mean', 999))
    selection['short_term_accuracy'] = best_short
    print(f"  Short-term accuracy: {best_short}")

    # Long-term stability: lowest 100-step NMAE
    best_long = min(all_results.keys(), key=lambda k: all_results[k].get('100_steps', {}).get('nmae_mean', 999))
    selection['long_term_stability'] = best_long
    print(f"  Long-term stability: {best_long}")

    # Best survival
    best_surv = max(all_results.keys(), key=lambda k: all_results[k].get('100_steps', {}).get('survival_mean', 0))
    selection['best_survival'] = best_surv
    print(f"  Best survival: {best_surv}")

    output = {
        'subagent': 'F',
        'statistical_design': {
            'train_seeds': n_train_seeds,
            'eval_seeds': n_eval_seeds,
            'horizons': horizons,
        },
        'results': all_results,
        'model_selection': selection,
    }

    with open(OUTPUT / "SUBAGENT_F_RESULTS.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nSubagent F done in {elapsed:.1f}s")
    return output


if __name__ == "__main__":
    main()
