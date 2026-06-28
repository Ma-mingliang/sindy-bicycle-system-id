"""
Fast comparison: Version L vs Version T with reduced parameters for quick validation.
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
from version_l_local_residual import VersionLLocalResidual
from version_t_trajectory_residual import VersionTTrajectoryResidual


def run_trajectory(model, s0, n_steps, tau_func):
    s_model = s0.copy()
    s_real = s0.copy()
    model_traj = [s0.copy()]
    real_traj = [s0.copy()]
    for i in range(n_steps):
        tau = tau_func(i, s_real)
        s_real_new = real_step(s_real, tau)
        s_model = model.predict(s_model, tau)
        if abs(s_real_new[0]) > math.pi / 3:
            model_traj.append(np.full(4, np.nan))
            real_traj.append(np.full(4, np.nan))
            break
        model_traj.append(s_model.copy())
        real_traj.append(s_real_new.copy())
        s_real = s_real_new
    return np.array(model_traj), np.array(real_traj)


def compute_metrics(model_traj, real_traj, state_std):
    n = min(len(model_traj), len(real_traj))
    valid = ~(np.any(np.isnan(model_traj[:n]), axis=1) | np.any(np.isnan(real_traj[:n]), axis=1))
    nv = int(np.sum(valid))
    if nv == 0:
        return {'mae': float('nan'), 'survival_steps': 0, 'per_state_mae': [float('nan')]*4}
    idx = np.where(valid)[0]
    errs = np.abs(model_traj[idx] - real_traj[idx])
    return {
        'mae': float(np.mean(errs)),
        'survival_steps': nv,
        'per_state_mae': [float(np.mean(errs[:, i])) for i in range(4)],
        'normalized_mae': float(np.mean(errs / state_std)),
    }


def main():
    print("=" * 60, flush=True)
    print("Version L vs Version T FAST Comparison", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()

    # Generate training data
    print("\n[1/4] Generating training data (2000 samples)...", flush=True)
    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(
        n_samples=2000, seed=42
    )
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    print(f"\n[2/4] Training Version L...", flush=True)
    model_l = VersionLLocalResidual(n_models=3, dagger_rounds=2, n_epochs=50, residual_scale=0.3)
    model_l.train(states, actions, deltas, state_std, action_std, delta_std)
    print(f"  Done in {time.time()-t1:.1f}s", flush=True)

    t2 = time.time()
    print(f"\n[3/4] Training Version T...", flush=True)
    model_t = VersionTTrajectoryResidual(n_models=3, dagger_rounds=2, n_epochs=50, residual_scale=0.3)
    model_t.train(states, actions, deltas, state_std, action_std, delta_std)
    print(f"  Done in {time.time()-t2:.1f}s", flush=True)

    t3 = time.time()
    print(f"\n[4/4] Evaluating...", flush=True)

    results = {'version_l': {}, 'version_t': {}}
    n_eval = 3

    # Mode A
    print("  Mode A...", flush=True)
    for vname, model in [('version_l', model_l), ('version_t', model_t)]:
        errs = []
        for si in range(n_eval):
            tf = make_tau_func(200+si, 100)
            s = np.array([0.1*(si-1), 0.0, 0.0, 0.0])
            for step in range(100):
                tau = tf(step, s)
                s_next = real_step(s, tau)
                s_pred = model.predict(s, tau)
                errs.append(float(np.mean(np.abs(s_pred - s_next))))
                s = s_next
                if abs(s[0]) > math.pi/3: break
        results[vname]['mode_a'] = {'mean_mae': float(np.mean(errs))}

    # Mode B/C/D at 50 steps
    for mode_name, mode_func in [('mode_b', 'fixed'), ('mode_c', 'mixed'), ('mode_d', 'closedloop')]:
        print(f"  {mode_name.upper()}...", flush=True)
        for vname, model in [('version_l', model_l), ('version_t', model_t)]:
            step_errs = []
            for si in range(n_eval):
                tf = make_tau_func(300+si, 50)
                s0 = np.array([0.1*(si-1), 0.0, 0.0, 0.0])
                if mode_name == 'mode_d':
                    from methods_common import K_lqr
                    s_m = s0.copy(); s_r = s0.copy()
                    rng = np.random.RandomState(42+si)
                    dists = rng.uniform(-0.5, 0.5, 50)
                    target = 0.0
                    mt = [s_m.copy()]; rt = [s_r.copy()]
                    for step in range(50):
                        x_lr = np.array([s_r[0]-target, s_r[2], s_r[1], s_r[3]])
                        tau_r = float(-K_lqr @ x_lr) + dists[step]
                        x_lm = np.array([s_m[0]-target, s_m[2], s_m[1], s_m[3]])
                        tau_m = float(-K_lqr @ x_lm) + dists[step]
                        s_r = real_step(s_r, tau_r)
                        s_m = model.predict(s_m, tau_m)
                        if abs(s_r[0]) > math.pi/3:
                            mt.append(np.full(4, np.nan)); rt.append(np.full(4, np.nan)); break
                        mt.append(s_m.copy()); rt.append(s_r.copy())
                    mt = np.array(mt); rt = np.array(rt)
                else:
                    mt, rt = run_trajectory(model, s0, 50, tf)
                m = compute_metrics(mt, rt, state_std)
                step_errs.append(m['mae'])
            valid_e = [e for e in step_errs if not math.isnan(e)]
            results[vname][mode_name] = {'mae_50steps': float(np.mean(valid_e)) if valid_e else float('nan')}

    # Initial conditions
    print("  Initial conditions...", flush=True)
    inits = {
        'nominal': np.array([0.0, 0.0, 0.0, 0.0]),
        'large_phi': np.array([0.3, 0.0, 0.0, 0.0]),
        'high_vel': np.array([0.0, 0.0, 1.5, 0.0]),
    }
    for vname, model in [('version_l', model_l), ('version_t', model_t)]:
        ic_results = {}
        for ic_name, s0 in inits.items():
            tf = make_tau_func(500+hash(ic_name)%100, 50)
            mt, rt = run_trajectory(model, s0, 50, tf)
            m = compute_metrics(mt, rt, state_std)
            ic_results[ic_name] = m['mae']
        results[vname]['initial_conditions'] = ic_results

    # Action signals
    print("  Action signals...", flush=True)
    actions_test = {
        'small_sine': lambda step: 5.0*math.sin(2*math.pi*step/50),
        'large_sine': lambda step: 30.0*math.sin(2*math.pi*step/30),
        'step': lambda step: 20.0 if step > 10 else 0.0,
    }
    for vname, model in [('version_l', model_l), ('version_t', model_t)]:
        act_results = {}
        for act_name, act_func in actions_test.items():
            s = np.array([0.1, 0.0, 0.0, 0.0])
            errs = []
            for step in range(50):
                tau = act_func(step)
                s_next = real_step(s, tau)
                s_pred = model.predict(s, tau)
                errs.append(float(np.mean(np.abs(s_pred - s_next))))
                s = s_next
                if abs(s[0]) > math.pi/3: break
            act_results[act_name] = float(np.mean(errs)) if errs else float('nan')
        results[vname]['action_signals'] = act_results

    # OOD states
    print("  OOD states...", flush=True)
    ood = {
        'extreme_phi': np.array([0.5, 0.0, 0.0, 0.0]),
        'extreme_delta': np.array([0.0, 0.3, 0.0, 0.0]),
    }
    for vname, model in [('version_l', model_l), ('version_t', model_t)]:
        ood_results = {}
        for ood_name, s0 in ood.items():
            s_next_r = real_step(s0, 10.0)
            s_next_m = model.predict(s0, 10.0)
            ood_results[ood_name] = float(np.mean(np.abs(s_next_m - s_next_r)))
        results[vname]['ood_states'] = ood_results

    # Save
    results['config'] = {
        'n_models': 3, 'dagger_rounds': 2, 'n_epochs': 50,
        'residual_scale': 0.3, 'training_samples': 2000, 'seed': 42,
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
        l_val = l.get('mean_mae', l.get('mae_50steps', float('nan')))
        t_val = t.get('mean_mae', t.get('mae_50steps', float('nan')))
        w = 'L' if (not math.isnan(l_val) and l_val < t_val) else 'T'
        print(f"  {mode.upper():>8s}: L={l_val:.6f}  T={t_val:.6f}  -> {w}", flush=True)

    print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
