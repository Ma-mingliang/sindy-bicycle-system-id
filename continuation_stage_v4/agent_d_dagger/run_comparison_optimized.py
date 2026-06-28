"""
Optimized comparison: Version L vs Version T.
Trains GP ONCE, shares it between both versions.
Only the DAgger residual collection differs.
"""

import sys
import os
import json
import numpy as np
import math
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CURRENT_DIR)

from methods_common import real_step, generate_training_data
from methods_evaluate import make_tau_func
from version_l_local_residual import GPStandard
import torch
import torch.nn as nn


class ResidualNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )
    def forward(self, s, a=None):
        if a is not None:
            return self.net(torch.cat([s, a], dim=-1))
        return self.net(s)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_ensemble(train_inputs, train_residuals, n_models=3, n_epochs=50):
    """Train NN ensemble on given data."""
    models = []
    for i in range(n_models):
        torch.manual_seed(i * 42)
        np.random.seed(i * 42)
        model = ResidualNet().to(DEVICE)
        train_x = torch.FloatTensor(train_inputs).to(DEVICE)
        train_y = torch.FloatTensor(train_residuals).to(DEVICE)
        ds = torch.utils.data.TensorDataset(train_x, train_y)
        loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = torch.nn.MSELoss()
        model.train()
        for _ in range(n_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                loss = crit(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        models.append(model)
    return models


def predict_ensemble(models, s_norm, a_norm):
    s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
    a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
    preds = []
    for model in models:
        with torch.no_grad():
            preds.append(model(s_t, a_t).cpu().numpy()[0])
    return np.mean(preds, axis=0)


def collect_dagger_L(baseline, models, state_std, action_std, delta_std, residual_scale):
    """Version L: local dynamics residual at MODEL state."""
    new_inputs, new_residuals = [], []
    for seg_i in range(3):
        tf = make_tau_func(100+seg_i, 500)
        phi_init = np.random.uniform(-0.25, 0.25)
        s0 = np.array([phi_init, 0.0, 0.0, 0.0])
        s_model = s0.copy()
        s_real = s0.copy()
        for step in range(500):
            tau = tf(step, s_real)
            s_real = real_step(s_real, tau)
            if abs(s_real[0]) > math.pi/3: break
            s_model_k = s_model.copy()
            s_next_gp = baseline.predict(s_model_k, tau)
            s_next_real = real_step(s_model_k, tau)  # real at MODEL state
            gp_delta = s_next_gp - s_model_k
            real_delta = s_next_real - s_model_k
            residual = (real_delta - gp_delta) / delta_std
            s_norm = s_model_k / state_std
            a_norm = tau / action_std
            delta_nn = predict_ensemble(models, s_norm, a_norm)
            s_model = s_next_gp + delta_nn * delta_std * residual_scale
            new_inputs.append(np.concatenate([s_norm, [a_norm]]))
            new_residuals.append(residual)
    return np.array(new_inputs), np.array(new_residuals)


def collect_dagger_T(baseline, models, state_std, action_std, delta_std, residual_scale):
    """Version T: trajectory synchronization residual."""
    new_inputs, new_residuals = [], []
    for seg_i in range(3):
        tf = make_tau_func(100+seg_i, 500)
        phi_init = np.random.uniform(-0.25, 0.25)
        s0 = np.array([phi_init, 0.0, 0.0, 0.0])
        s_model = s0.copy()
        s_real = s0.copy()
        for step in range(500):
            tau = tf(step, s_real)
            s_real_next = real_step(s_real, tau)
            if abs(s_real_next[0]) > math.pi/3: break
            s_model_k = s_model.copy()
            s_next_gp = baseline.predict(s_model_k, tau)
            residual = (s_real_next - s_next_gp) / delta_std
            s_norm = s_model_k / state_std
            a_norm = tau / action_std
            delta_nn = predict_ensemble(models, s_norm, a_norm)
            s_model = s_next_gp + delta_nn * delta_std * residual_scale
            s_real = s_real_next
            new_inputs.append(np.concatenate([s_norm, [a_norm]]))
            new_residuals.append(residual)
    return np.array(new_inputs), np.array(new_residuals)


def composite_predict(baseline, models, s, tau, state_std, action_std, delta_std, residual_scale):
    """Unified prediction: GP + scaled NN."""
    s_next_gp = baseline.predict(s, tau)
    s_norm = s / state_std
    a_norm = tau / action_std
    delta_nn = predict_ensemble(models, s_norm, a_norm)
    return s_next_gp + delta_nn * delta_std * residual_scale


def run_trajectory_predict(predict_fn, s0, n_steps, tau_func):
    s_model = s0.copy()
    s_real = s0.copy()
    mt = [s0.copy()]; rt = [s0.copy()]
    for i in range(n_steps):
        tau = tau_func(i, s_real)
        s_real = real_step(s_real, tau)
        s_model = predict_fn(s_model, tau)
        if abs(s_real[0]) > math.pi/3:
            mt.append(np.full(4, np.nan)); rt.append(np.full(4, np.nan)); break
        mt.append(s_model.copy()); rt.append(s_real.copy())
    return np.array(mt), np.array(rt)


def main():
    print("=" * 60, flush=True)
    print("Optimized Version L vs Version T Comparison", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()

    # Generate training data
    print("\n[1/5] Generating training data (2000 samples)...", flush=True)
    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(
        n_samples=2000, seed=42
    )
    print(f"  Done: {len(states)} samples ({time.time()-t0:.1f}s)", flush=True)

    # Train GP ONCE
    t1 = time.time()
    print(f"\n[2/5] Training GP baseline (shared)...", flush=True)
    gp = GPStandard(max_samples=2000)
    gp.train(states, actions, deltas, state_std, action_std, delta_std)
    print(f"  Done ({time.time()-t1:.1f}s)", flush=True)

    # Round 0 residuals (same for both versions at expert states)
    print(f"\n[3/5] Computing Round 0 residuals...", flush=True)
    residuals_r0 = np.empty_like(deltas)
    for i in range(len(states)):
        s_next = gp.predict(states[i], actions[i])
        residuals_r0[i] = (deltas[i] - (s_next - states[i])) / delta_std

    inputs_base = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])

    N_MODELS = 3
    N_EPOCHS = 50
    ROUNDS = 2
    RESIDUAL_SCALE = 0.3

    # DAgger for Version L
    t2 = time.time()
    print(f"\n[4/5] DAgger rounds for Version L...", flush=True)
    all_in_L = [inputs_base.copy()]
    all_re_L = [residuals_r0.copy()]
    for ri in range(ROUNDS):
        print(f"  Round {ri}: training ensemble...", flush=True)
        ti = np.vstack(all_in_L); tr = np.vstack(all_re_L)
        models_L = train_ensemble(ti, tr, N_MODELS, N_EPOCHS)
        if ri < ROUNDS - 1:
            print(f"  Round {ri}: collecting Version L data...", flush=True)
            ni, nr = collect_dagger_L(gp, models_L, state_std, action_std, delta_std, RESIDUAL_SCALE)
            if len(ni) > 0:
                all_in_L.append(ni); all_re_L.append(nr)
            print(f"  Round {ri}: collected {len(ni)} samples ({time.time()-t2:.1f}s)", flush=True)
    print(f"  Version L done ({time.time()-t2:.1f}s)", flush=True)

    # DAgger for Version T
    t3 = time.time()
    print(f"\n[4/5] DAgger rounds for Version T...", flush=True)
    all_in_T = [inputs_base.copy()]
    all_re_T = [residuals_r0.copy()]
    for ri in range(ROUNDS):
        print(f"  Round {ri}: training ensemble...", flush=True)
        ti = np.vstack(all_in_T); tr = np.vstack(all_re_T)
        models_T = train_ensemble(ti, tr, N_MODELS, N_EPOCHS)
        if ri < ROUNDS - 1:
            print(f"  Round {ri}: collecting Version T data...", flush=True)
            ni, nr = collect_dagger_T(gp, models_T, state_std, action_std, delta_std, RESIDUAL_SCALE)
            if len(ni) > 0:
                all_in_T.append(ni); all_re_T.append(nr)
            print(f"  Round {ri}: collected {len(ni)} samples ({time.time()-t3:.1f}s)", flush=True)
    print(f"  Version T done ({time.time()-t3:.1f}s)", flush=True)

    # Build predict functions
    def predict_L(s, tau):
        return composite_predict(gp, models_L, s, tau, state_std, action_std, delta_std, RESIDUAL_SCALE)
    def predict_T(s, tau):
        return composite_predict(gp, models_T, s, tau, state_std, action_std, delta_std, RESIDUAL_SCALE)

    # Evaluate
    t4 = time.time()
    print(f"\n[5/5] Evaluating...", flush=True)
    results = {'version_l': {}, 'version_t': {}}

    # Mode A
    print("  Mode A...", flush=True)
    for vname, pfn in [('version_l', predict_L), ('version_t', predict_T)]:
        errs = []
        for si in range(5):
            tf = make_tau_func(200+si, 100)
            s = np.array([0.1*(si-2), 0.0, 0.0, 0.0])
            for step in range(100):
                tau = tf(step, s)
                sn = real_step(s, tau)
                sp = pfn(s, tau)
                errs.append(float(np.mean(np.abs(sp - sn))))
                s = sn
                if abs(s[0]) > math.pi/3: break
        results[vname]['mode_a'] = {'mean_mae': float(np.mean(errs))}

    # Mode B/C/D
    for mode_name in ['mode_b', 'mode_c', 'mode_d']:
        print(f"  {mode_name.upper()}...", flush=True)
        for vname, pfn in [('version_l', predict_L), ('version_t', predict_T)]:
            step_errs = []
            for si in range(5):
                tf = make_tau_func(300+si, 100)
                s0 = np.array([0.1*(si-2), 0.0, 0.0, 0.0])
                if mode_name == 'mode_d':
                    from methods_common import K_lqr
                    s_m = s0.copy(); s_r = s0.copy()
                    rng = np.random.RandomState(42+si)
                    dists = rng.uniform(-0.5, 0.5, 100)
                    mt2 = [s_m.copy()]; rt2 = [s_r.copy()]
                    for step in range(100):
                        x_lr = np.array([s_r[0], s_r[2], s_r[1], s_r[3]])
                        tau_r = float(-K_lqr @ x_lr) + dists[step]
                        x_lm = np.array([s_m[0], s_m[2], s_m[1], s_m[3]])
                        tau_m = float(-K_lqr @ x_lm) + dists[step]
                        s_r = real_step(s_r, tau_r)
                        s_m = pfn(s_m, tau_m)
                        if abs(s_r[0]) > math.pi/3:
                            mt2.append(np.full(4, np.nan)); rt2.append(np.full(4, np.nan)); break
                        mt2.append(s_m.copy()); rt2.append(s_r.copy())
                    mt2 = np.array(mt2); rt2 = np.array(rt2)
                else:
                    mt2, rt2 = run_trajectory_predict(pfn, s0, 100, tf)
                # Compute metrics
                n = min(len(mt2), len(rt2))
                valid = ~(np.any(np.isnan(mt2[:n]), axis=1) | np.any(np.isnan(rt2[:n]), axis=1))
                nv = int(np.sum(valid))
                if nv > 0:
                    idx = np.where(valid)[0]
                    step_errs.append(float(np.mean(np.abs(mt2[idx] - rt2[idx]))))
            valid_e = [e for e in step_errs if not math.isnan(e)]
            results[vname][mode_name] = {
                'mae_100steps': float(np.mean(valid_e)) if valid_e else float('nan'),
                'n_seeds': len(valid_e),
            }

    # Per-state breakdown for mode_b at 100 steps
    print("  Per-state breakdown (Mode B, 100 steps)...", flush=True)
    for vname, pfn in [('version_l', predict_L), ('version_t', predict_T)]:
        all_errs = []
        for si in range(5):
            tf = make_tau_func(300+si, 100)
            s0 = np.array([0.1*(si-2), 0.0, 0.0, 0.0])
            mt2, rt2 = run_trajectory_predict(pfn, s0, 100, tf)
            n = min(len(mt2), len(rt2))
            valid = ~(np.any(np.isnan(mt2[:n]), axis=1) | np.any(np.isnan(rt2[:n]), axis=1))
            nv = int(np.sum(valid))
            if nv > 0:
                idx = np.where(valid)[0]
                all_errs.append(np.abs(mt2[idx] - rt2[idx]))
        if all_errs:
            combined = np.vstack(all_errs)
            state_names = ['phi', 'delta', 'phi_dot', 'delta_dot']
            per_state = {state_names[i]: float(np.mean(combined[:, i])) for i in range(4)}
            results[vname]['per_state_mode_b_100'] = per_state

    # Initial conditions
    print("  Initial conditions...", flush=True)
    inits = {
        'nominal': np.array([0.0, 0.0, 0.0, 0.0]),
        'large_phi': np.array([0.3, 0.0, 0.0, 0.0]),
        'high_vel': np.array([0.0, 0.0, 1.5, 0.0]),
        'combined': np.array([0.2, 0.1, 1.0, 0.5]),
    }
    for vname, pfn in [('version_l', predict_L), ('version_t', predict_T)]:
        ic = {}
        for name, s0 in inits.items():
            tf = make_tau_func(500+hash(name)%100, 100)
            mt2, rt2 = run_trajectory_predict(pfn, s0, 100, tf)
            n = min(len(mt2), len(rt2))
            valid = ~(np.any(np.isnan(mt2[:n]), axis=1) | np.any(np.isnan(rt2[:n]), axis=1))
            nv = int(np.sum(valid))
            if nv > 0:
                idx = np.where(valid)[0]
                ic[name] = {'mae': float(np.mean(np.abs(mt2[idx] - rt2[idx]))), 'survival': nv}
            else:
                ic[name] = {'mae': float('nan'), 'survival': 0}
        results[vname]['initial_conditions'] = ic

    # Action signals
    print("  Action signals...", flush=True)
    act_funcs = {
        'small_sine': lambda step: 5.0*math.sin(2*math.pi*step/50),
        'large_sine': lambda step: 30.0*math.sin(2*math.pi*step/30),
        'step': lambda step: 20.0 if step > 10 else 0.0,
    }
    for vname, pfn in [('version_l', predict_L), ('version_t', predict_T)]:
        ac = {}
        for name, func in act_funcs.items():
            s = np.array([0.1, 0.0, 0.0, 0.0])
            errs = []
            for step in range(100):
                tau = func(step)
                sn = real_step(s, tau)
                sp = pfn(s, tau)
                errs.append(float(np.mean(np.abs(sp - sn))))
                s = sn
                if abs(s[0]) > math.pi/3: break
            ac[name] = float(np.mean(errs)) if errs else float('nan')
        results[vname]['action_signals'] = ac

    # OOD
    print("  OOD states...", flush=True)
    ood = {
        'extreme_phi': np.array([0.5, 0.0, 0.0, 0.0]),
        'extreme_delta': np.array([0.0, 0.3, 0.0, 0.0]),
    }
    for vname, pfn in [('version_l', predict_L), ('version_t', predict_T)]:
        oc = {}
        for name, s0 in ood.items():
            sn = real_step(s0, 10.0)
            sp = pfn(s0, 10.0)
            oc[name] = float(np.mean(np.abs(sp - sn)))
        results[vname]['ood_states'] = oc

    # Save
    results['config'] = {
        'n_models': N_MODELS, 'dagger_rounds': ROUNDS, 'n_epochs': N_EPOCHS,
        'residual_scale': RESIDUAL_SCALE, 'training_samples': len(states),
        'gp_max_samples': 2000, 'seed': 42,
    }
    out_path = os.path.join(CURRENT_DIR, 'LOCAL_VS_TRAJECTORY_RESULTS.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}", flush=True)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for mode in ['mode_a', 'mode_b', 'mode_c', 'mode_d']:
        l = results['version_l'].get(mode, {})
        t = results['version_t'].get(mode, {})
        l_val = l.get('mean_mae', l.get('mae_100steps', float('nan')))
        t_val = t.get('mean_mae', t.get('mae_100steps', float('nan')))
        w = 'L' if (not math.isnan(l_val) and l_val < t_val) else 'T'
        print(f"  {mode.upper():>8s}: L={l_val:.6f}  T={t_val:.6f}  -> Winner: {w}", flush=True)

    print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
