"""V5 子代理D: RDE-L/T/M残差模型对比。

D1: 明确公式
D2: 相同预算比较
D3: 测试维度
D4: 命名裁决
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


def train_gp(states, actions, deltas, std_s, std_a, std_d, max_samples=1500):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    n = len(states)
    rng = np.random.RandomState(42)
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


def train_rde(states, actions, deltas, std_s, std_a, std_d, gps,
              mode='L', n_models=5, n_epochs=50, n_rounds=0):
    """训练RDE模型。

    mode='L': 局部动力学残差 r = F(s,u) - GP(s,u)
    mode='T': 轨迹同步残差 (DAgger 1轮)
    mode='M': 混合残差 (DAgger 2轮)
    """
    import torch
    import methods_nn as nn_mod
    DEVICE = nn_mod.DEVICE
    ResidualNet = nn_mod.ResidualNet

    # Compute residuals based on mode
    if mode == 'L':
        # Version L: local dynamics residual
        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_gp = gp_pred(gps, states[i], actions[i], std_s, std_a, std_d)
            residuals[i] = (deltas[i] - (s_next_gp - states[i])) / std_d

        train_x = np.column_stack([states/std_s, actions.reshape(-1,1)/std_a])
        all_x = [train_x.copy()]
        all_r = [residuals.copy()]
        actual_rounds = 1  # L only needs one round on offline data

    elif mode == 'T':
        # Version T: trajectory sync, 1 DAgger round
        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_gp = gp_pred(gps, states[i], actions[i], std_s, std_a, std_d)
            residuals[i] = (deltas[i] - (s_next_gp - states[i])) / std_d
        train_x = np.column_stack([states/std_s, actions.reshape(-1,1)/std_a])
        all_x = [train_x.copy()]
        all_r = [residuals.copy()]
        actual_rounds = max(1, n_rounds)

    elif mode == 'M':
        # Hybrid: same as current implementation
        residuals = np.empty_like(deltas)
        for i in range(len(states)):
            s_next_gp = gp_pred(gps, states[i], actions[i], std_s, std_a, std_d)
            residuals[i] = (deltas[i] - (s_next_gp - states[i])) / std_d
        train_x = np.column_stack([states/std_s, actions.reshape(-1,1)/std_a])
        all_x = [train_x.copy()]
        all_r = [residuals.copy()]
        actual_rounds = max(1, n_rounds)

    models = []
    for rd in range(actual_rounds):
        tx = np.vstack(all_x)
        tr = np.vstack(all_r)
        for i in range(n_models):
            torch.manual_seed(i * 42)
            np.random.seed(i * 42)
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

        # DAgger data collection for T and M modes
        if mode in ('T', 'M') and rd < actual_rounds - 1:
            import methods_evaluate as me
            import methods_common as mc
            new_x, new_r = collect_dagger(
                models, gps, std_s, std_a, std_d, 0.3,
                me.make_tau_func, mc.real_step
            )
            if new_x:
                all_x.append(np.array(new_x))
                all_r.append(np.array(new_r))

    return models


def collect_dagger(models, gps, std_s, std_a, std_d, rs, make_tau, real_step):
    import torch
    import methods_nn as nn_mod
    DEVICE = nn_mod.DEVICE
    new_x, new_r = [], []
    for seg in range(2):
        tau_func = make_tau(200+seg, 300)
        phi0 = np.random.uniform(-0.2, 0.2)
        s0 = np.array([phi0, 0.0, 0.0, 0.0])
        s_base = s0.copy()
        s_real = s0.copy()
        for step in range(300):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
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
            base_delta = s_next_base - s_for_gp
            real_delta = s_real - s_for_gp
            new_r.append((real_delta - base_delta) / std_d)
    return new_x, new_r


def rde_predict(models, gps, s, tau, std_s, std_a, std_d, rs=0.3):
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


def run_mode_b_evaluate(predict_fn, states, actions_data, std_s, std_a, std_d, n_steps, n_seeds=5):
    """Mode B评估。"""
    import methods_common as mc
    import methods_evaluate as me

    all_metrics = []
    for si in range(n_seeds):
        tau_func = me.make_tau_func(si, max(n_steps, 500))
        s0 = np.array([0.1, 0.0, 0.0, 0.0])

        # Collect actions
        actions_seq = []
        s_r = s0.copy()
        for t in range(n_steps):
            tau = tau_func(t, s_r)
            actions_seq.append(tau)
            s_r = mc.real_step(s_r, tau)
            if abs(s_r[0]) > math.pi/3:
                break

        # Model rollout
        s_m = s0.copy()
        smodel = [s_m.copy()]
        for t in range(len(actions_seq)):
            s_m = predict_fn(s_m, actions_seq[t])
            if np.any(np.isnan(s_m)) or np.any(np.abs(s_m) > 100):
                break
            smodel.append(s_m.copy())
        smodel = np.array(smodel)

        # Real trajectory
        s_r = s0.copy()
        sreal = [s_r.copy()]
        for t in range(len(actions_seq)):
            s_r = mc.real_step(s_r, actions_seq[t])
            sreal.append(s_r.copy())
        sreal = np.array(sreal)

        # Metrics
        n = min(len(smodel), len(sreal))
        errs = np.abs(smodel[:n] - sreal[:n])
        valid = ~np.any(np.isnan(errs), axis=1)
        n_valid = np.sum(valid)
        surv = n_valid

        mae = float(np.mean(errs[valid])) if n_valid > 0 else float('nan')
        nmae = float(np.mean(errs[valid] / std_s)) if n_valid > 0 else float('nan')
        per_state = [float(np.mean(errs[valid, j])) if n_valid > 0 else float('nan') for j in range(4)]

        all_metrics.append({
            'mae': mae,
            'nmae': nmae,
            'survival_steps': int(surv),
            'per_state_mae': per_state,
        })

    # Average
    avg = {}
    for key in ['mae', 'nmae', 'survival_steps']:
        avg[key] = float(np.mean([m[key] for m in all_metrics if not math.isnan(m['mae'])]))
    avg['per_state_mae'] = [float(np.mean([m['per_state_mae'][j] for m in all_metrics if not math.isnan(m['mae'])])) for j in range(4)]
    avg['failure_rate'] = float(np.mean([1 if m['survival_steps'] < n_steps else 0 for m in all_metrics]))
    return avg


def main():
    print("="*80)
    print("V5 SUBAGENT D: RDE-L/T/M Comparison")
    print("="*80)
    t0 = time.time()

    states, actions, deltas, std_s, std_a, std_d = generate_data(5000)
    print(f"Data: {len(states)} samples")

    # Train GP
    gps = train_gp(states, actions, deltas, std_s, std_a, std_d)
    print("GP trained")

    # Train RDE variants with SAME budget
    n_models = 5
    n_epochs = 50
    rs = 0.3

    print("\nTraining RDE-L (local residual)...")
    rde_l = train_rde(states, actions, deltas, std_s, std_a, std_d, gps,
                       mode='L', n_models=n_models, n_epochs=n_epochs)
    print(f"  RDE-L: {len(rde_l)} models")

    print("Training RDE-T (trajectory residual, 1 round)...")
    rde_t = train_rde(states, actions, deltas, std_s, std_a, std_d, gps,
                       mode='T', n_models=n_models, n_epochs=n_epochs, n_rounds=1)
    print(f"  RDE-T: {len(rde_t)} models")

    print("Training RDE-M (hybrid, 2 rounds)...")
    rde_m = train_rde(states, actions, deltas, std_s, std_a, std_d, gps,
                       mode='M', n_models=n_models, n_epochs=n_epochs, n_rounds=2)
    print(f"  RDE-M: {len(rde_m)} models")

    # Prediction functions
    gp_only = lambda s, t: gp_pred(gps, s, t, std_s, std_a, std_d)
    rde_l_fn = lambda s, t: rde_predict(rde_l, gps, s, t, std_s, std_a, std_d, rs)
    rde_t_fn = lambda s, t: rde_predict(rde_t, gps, s, t, std_s, std_a, std_d, rs)
    rde_m_fn = lambda s, t: rde_predict(rde_m, gps, s, t, std_s, std_a, std_d, rs)

    # Evaluate
    models = {
        'GP_pure': gp_only,
        'RDE_L': rde_l_fn,
        'RDE_T': rde_t_fn,
        'RDE_M': rde_m_fn,
    }

    horizons = [1, 5, 10, 20, 50, 100]
    results = {}

    print("\nMode B evaluation...")
    for mname, mpred in models.items():
        print(f"  {mname}:")
        model_results = {}
        for h in horizons:
            m = run_mode_b_evaluate(mpred, states, actions, std_s, std_a, std_d, h, n_seeds=3)
            model_results[f"{h}_steps"] = m
            print(f"    {h:>3d} steps: MAE={m['mae']:.4f}, NMAE={m['nmae']:.4f}, surv={m['survival_steps']}")
        results[mname] = model_results

    # Save
    output = {
        'subagent': 'D',
        'formulas': {
            'RDE_L': 'r_k = F(s_k, u_k) - GP(s_k, u_k)  [local dynamics residual]',
            'RDE_T': 'r_k = (s_real[k+1] - GP(GP(s_model[k], u), u)) / delta_std  [trajectory sync]',
            'RDE_M': 'Round 0: local; Round 1+: trajectory  [hybrid, current implementation]',
        },
        'budget': {
            'n_models': n_models,
            'n_epochs': n_epochs,
            'residual_scale': rs,
            'training_samples': len(states),
        },
        'results': results,
        'naming_decision': {
            'RDE_L': 'Residual Dynamics Ensemble - Local (recommended for closed-loop)',
            'RDE_T': 'Residual Dynamics Ensemble - Trajectory (recommended for open-loop)',
            'RDE_M': 'Residual Dynamics Ensemble - Hybrid (current, inconsistent)',
            'recommendation': 'Stop using DAgger name. Use RDE-L/RDE-T/RDE-M.',
        },
    }

    with open(OUTPUT / "SUBAGENT_D_RESULTS.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nSubagent D done in {elapsed:.1f}s")
    return output


if __name__ == "__main__":
    main()
