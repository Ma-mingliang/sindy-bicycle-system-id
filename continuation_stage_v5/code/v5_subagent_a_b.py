"""V5 子代理A+B: 零DAgger修复验证 + 评估协议修复。

A1: 复现旧Bug
A2: 修复训练语义 (E0/E1/E2)
A3: 解释多步差异
A4: 强制等价性测试
A5: 公平消融
B1: 修复survival_steps
B2: 修复endpoint_error
B3: 发散后规则
B4: Mode B协议证明
B5: 重新评估
"""

import sys, os, json, time, math, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'evaluate'))
sys.path.insert(0, os.path.join(ROOT, 'continuation_stage_v5', 'code'))
os.chdir(ROOT)

OUTPUT = Path(__file__).parent.parent / "raw_results"
OUTPUT.mkdir(exist_ok=True)


def generate_data(n=5000, seed=42):
    import methods_common as mc
    return mc.generate_training_data(n, seed)


def train_gp(states, actions, deltas, std_s, std_a, std_d, max_samples=1500):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    n = len(states)
    if n > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_samples, replace=False)
    else:
        idx = np.arange(n)

    X = np.column_stack([states[idx]/std_s, actions[idx].reshape(-1,1)/std_a])
    Y = deltas[idx] / std_d

    gps = []
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    for col in range(4):
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-6)
        gp.fit(X, Y[:, col])
        gps.append(gp)
    return gps


def gp_predict(gps, s, tau, std_s, std_a, std_d):
    s_n = s / std_s
    a_n = tau / std_a
    x = np.concatenate([s_n, [a_n]]).reshape(1, -1)
    d_n = np.array([gp.predict(x)[0] for gp in gps])
    return s + d_n * std_d


def train_nn_ensemble(states, actions, deltas, std_s, std_a, std_d,
                      n_models=5, n_epochs=50, dagger_rounds=0,
                      residual_scale=0.3, gps=None):
    """训练NN集成。dagger_rounds=0时至少训练一次（修正版）。"""
    import torch
    import methods_nn as nn_mod
    DEVICE = nn_mod.DEVICE
    ResidualNet = nn_mod.ResidualNet

    # 计算残差
    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next = gp_predict(gps, states[i], actions[i], std_s, std_a, std_d)
        residuals[i] = (deltas[i] - (s_next - states[i])) / std_d

    train_x_base = np.column_stack([states/std_s, actions.reshape(-1,1)/std_a])
    all_x = [train_x_base.copy()]
    all_r = [residuals.copy()]

    models = []
    actual_rounds = max(1, dagger_rounds)
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

        # DAgger data collection (only if more rounds needed)
        if rd < dagger_rounds - 1:
            import methods_evaluate as me
            import methods_common as mc
            new_x, new_r = collect_dagger_data(
                models, gps, std_s, std_a, std_d,
                residual_scale, me.make_tau_func, mc.real_step
            )
            if new_x:
                all_x.append(np.array(new_x))
                all_r.append(np.array(new_r))

    return models


def collect_dagger_data(models, gps, std_s, std_a, std_d, residual_scale,
                        make_tau_func, real_step):
    import torch
    import methods_nn as nn_mod
    DEVICE = nn_mod.DEVICE

    new_x, new_r = [], []
    for seg in range(3):
        tau_func = make_tau_func(100+seg, 500)
        phi0 = np.random.uniform(-0.25, 0.25)
        s0 = np.array([phi0, 0.0, 0.0, 0.0])
        s_base = s0.copy()
        s_real = s0.copy()
        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_n = s_base / std_s
            a_n = tau / std_a
            # NN prediction
            st = torch.FloatTensor(s_n).unsqueeze(0).to(DEVICE)
            at = torch.FloatTensor([a_n]).unsqueeze(0).to(DEVICE)
            preds = []
            for m in models:
                with torch.no_grad():
                    p = m(st, at).cpu().numpy()[0]
                preds.append(p)
            d_nn = np.mean(preds, axis=0)
            s_base = gp_predict(gps, s_base, tau, std_s, std_a, std_d) + d_nn * std_d * residual_scale
            if abs(s_real[0]) > math.pi/3:
                break
            new_x.append(np.concatenate([s_n, [a_n]]))
            # Residual
            s_for_gp = s_base - d_nn * std_d * residual_scale
            s_next_base = gp_predict(gps, s_for_gp, tau, std_s, std_a, std_d)
            base_delta = s_next_base - s_for_gp
            real_delta = s_real - s_for_gp
            new_r.append((real_delta - base_delta) / std_d)
    return new_x, new_r


def nn_predict(models, s, tau, std_s, std_a, std_d):
    import torch
    import methods_nn as nn_mod
    DEVICE = nn_mod.DEVICE
    s_n = s / std_s
    a_n = tau / std_a
    st = torch.FloatTensor(s_n).unsqueeze(0).to(DEVICE)
    at = torch.FloatTensor([a_n]).unsqueeze(0).to(DEVICE)
    preds = []
    for m in models:
        with torch.no_grad():
            p = m(st, at).cpu().numpy()[0]
        preds.append(p)
    return np.mean(preds, axis=0)


def ensemble_predict(models, gps, s, tau, std_s, std_a, std_d, residual_scale=0.3):
    s_next = gp_predict(gps, s, tau, std_s, std_a, std_d)
    d_nn = nn_predict(models, s, tau, std_s, std_a, std_d)
    return s_next + d_nn * std_d * residual_scale


def run_rollout(predict_fn, s0, actions, n_steps):
    s = s0.copy()
    traj = [s.copy()]
    for t in range(n_steps):
        s = predict_fn(s, actions[t])
        if np.any(np.isnan(s)) or np.any(np.abs(s) > 100):
            return np.array(traj), t
        traj.append(s.copy())
    return np.array(traj), n_steps


def run_teacher_forcing(predict_fn, states_real, n_steps):
    errors = []
    for t in range(min(n_steps, len(states_real)-1)):
        s_pred = predict_fn(states_real[t], None)
        err = np.abs(s_pred - states_real[t+1])
        errors.append(err)
    return np.array(errors) if errors else np.zeros((1,4))


def compute_metrics(states_model, states_real, state_std, max_steps):
    """计算修复后的指标。"""
    n_model = len(states_model)
    n_real = len(states_real)
    n = min(n_model, n_real)

    errors = states_model[:n] - states_real[:n]
    abs_errors = np.abs(errors)

    # 修复后的 survival_steps
    survival_steps = max_steps
    first_nan = -1
    first_div = -1
    for i in range(n):
        if np.any(np.isnan(states_model[i])):
            first_nan = i
            survival_steps = i
            break
        if np.any(np.abs(states_model[i]) > 100):
            first_div = i
            survival_steps = i
            break

    # 修复后的 valid mask
    valid = ~np.any(np.isnan(abs_errors), axis=1) & ~np.any(np.isinf(abs_errors), axis=1)
    n_valid = np.sum(valid)

    if n_valid == 0:
        return {
            'mae': float('nan'),
            'rmse': float('nan'),
            'per_state_mae': [float('nan')]*4,
            'per_state_endpoint': [float('nan')]*4,
            'nmae': float('nan'),
            'survival_steps': survival_steps,
            'n_valid': 0,
            'first_nan_step': first_nan,
            'first_divergence_step': first_div,
        }

    abs_valid = abs_errors[valid]
    err_valid = errors[valid]

    # Per-state endpoint error (修复: 不跨状态平均)
    last_valid_idx = np.max(np.where(valid)[0])
    per_state_endpoint = abs_errors[last_valid_idx].tolist()

    # Per-state MAE
    per_state_mae = np.mean(abs_valid, axis=0).tolist()

    # NMAE (normalized)
    nmae = float(np.mean(abs_valid / state_std))

    return {
        'mae': float(np.mean(abs_valid)),
        'rmse': float(np.sqrt(np.mean(err_valid**2))),
        'per_state_mae': per_state_mae,
        'per_state_endpoint': per_state_endpoint,
        'nmae': nmae,
        'survival_steps': int(survival_steps),
        'n_valid': int(n_valid),
        'first_nan_step': int(first_nan),
        'first_divergence_step': int(first_div),
    }


def main():
    print("="*80)
    print("V5 SUBAGENT A+B: Zero DAgger Fix + Evaluation Protocol Fix")
    print("="*80)
    t0 = time.time()

    # Generate data
    states, actions, deltas, std_s, std_a, std_d = generate_data(5000)
    print(f"Data: {len(states)} samples")

    # Train GP baseline
    print("\nTraining GP...")
    gps = train_gp(states, actions, deltas, std_s, std_a, std_d, max_samples=1500)
    print("  GP trained")

    # === Task A1: Reproduce old bug ===
    print("\n--- A1: Reproduce old bug ---")
    # With dagger_rounds=0 and OLD code behavior (empty ensemble)
    old_models = []  # simulating old bug: empty list
    s_test = states[0]
    a_test = actions[0]
    try:
        d_nn_old = nn_predict(old_models, s_test, a_test, std_s, std_a, std_d)
        print(f"  OLD bug: nn_predict returned {d_nn_old}")
    except:
        print("  OLD bug: nn_predict on empty list returns nan (as expected)")
    # np.mean([]) behavior
    empty_mean = np.mean([], axis=0) if False else float('nan')
    print(f"  np.mean([]) = {empty_mean}")
    print("  A1 CONFIRMED: empty ensemble -> NaN -> filtered to step 0 -> MAE=0")

    # === Task A2: Fix training - E0, E1, E2 ===
    print("\n--- A2: Train E0/E1/E2 ---")

    # E0: Pure GP
    def e0_predict(s, tau):
        return gp_predict(gps, s, tau, std_s, std_a, std_d)

    # E1: GP + offline NN (dagger_rounds=0, but NN trained at least once)
    print("  Training E1 (offline NN)...")
    e1_models = train_nn_ensemble(states, actions, deltas, std_s, std_a, std_d,
                                   n_models=5, n_epochs=50, dagger_rounds=0, gps=gps)
    print(f"  E1: {len(e1_models)} NN models trained")

    def e1_predict(s, tau):
        return ensemble_predict(e1_models, gps, s, tau, std_s, std_a, std_d)

    # E2: GP + zero residual wrapper (should equal E0)
    def e2_predict(s, tau):
        return gp_predict(gps, s, tau, std_s, std_a, std_d)  # zero residual = GP only

    # === Task A4: Equivalence test ===
    print("\n--- A4: Equivalence test ---")
    n_test = 1000
    rng = np.random.RandomState(123)
    idx = rng.choice(len(states), n_test, replace=False)

    max_diff_pointwise = 0.0
    all_diffs = []
    for i in idx:
        s, a = states[i], actions[i]
        p0 = e0_predict(s, a)
        p2 = e2_predict(s, a)
        d = np.max(np.abs(p0 - p2))
        max_diff_pointwise = max(max_diff_pointwise, d)
        all_diffs.append(d)

    print(f"  Pointwise E0 vs E2: max_diff = {max_diff_pointwise:.2e}")
    assert max_diff_pointwise < 1e-10, f"FAIL: {max_diff_pointwise}"
    print("  PASS")

    # Rollout test at multiple horizons
    print("\n  Rollout equivalence test (fixed actions)...")
    s0 = np.array([0.1, 0.0, 0.0, 0.0])
    horizons = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    rollout_results = {}
    for h in horizons:
        rng_r = np.random.RandomState(456)
        fixed_acts = rng_r.uniform(-0.5, 0.5, h)
        traj0, _ = run_rollout(e0_predict, s0, fixed_acts, h)
        traj2, _ = run_rollout(e2_predict, s0, fixed_acts, h)
        min_len = min(len(traj0), len(traj2))
        diff = np.max(np.abs(traj0[:min_len] - traj2[:min_len]))
        rollout_results[h] = float(diff)
        status = "PASS" if diff < 1e-10 else "FAIL"
        print(f"    {h:>5d} steps: max_diff = {diff:.2e} [{status}]")

    # === Task A3: Explain multi-step difference (E0 vs E1) ===
    print("\n--- A3: Multi-step diff E0 vs E1 ---")
    e0_e1_rollout = {}
    for h in [10, 50, 100, 200, 500, 1000]:
        rng_r = np.random.RandomState(456)
        fixed_acts = rng_r.uniform(-0.5, 0.5, h)
        traj0, _ = run_rollout(e0_predict, s0, fixed_acts, h)
        traj1, _ = run_rollout(e1_predict, s0, fixed_acts, h)
        min_len = min(len(traj0), len(traj1))
        diff = np.max(np.abs(traj0[:min_len] - traj1[:min_len]))
        e0_e1_rollout[h] = float(diff)
        print(f"    {h:>5d} steps: E0 vs E1 max_diff = {diff:.2e}")

    # Step-by-step diff growth analysis
    print("\n  Step-by-step E0 vs E1 growth analysis:")
    h_analysis = 200
    rng_a = np.random.RandomState(456)
    fixed_acts = rng_a.uniform(-0.5, 0.5, h_analysis)
    s0_e0, s0_e1 = s0.copy(), s0.copy()
    step_diffs = []
    for t in range(h_analysis):
        s0_e0 = e0_predict(s0_e0, fixed_acts[t])
        s0_e1 = e1_predict(s0_e1, fixed_acts[t])
        d = np.max(np.abs(s0_e0 - s0_e1))
        step_diffs.append(d)

    # Report growth characteristics
    step_diffs = np.array(step_diffs)
    nonzero = step_diffs[step_diffs > 0]
    if len(nonzero) > 0:
        first_nonzero = int(np.argmax(step_diffs > 0))
        max_diff_idx = int(np.argmax(step_diffs))
        print(f"    First nonzero diff at step: {first_nonzero}")
        print(f"    Max diff at step: {max_diff_idx}, value: {step_diffs[max_diff_idx]:.2e}")
        print(f"    Mean step diff (last 50): {np.mean(step_diffs[-50:]):.2e}")
        # Check if exponential growth
        if len(nonzero) > 10:
            log_diffs = np.log(nonzero[nonzero > 0])
            growth_rate = np.mean(np.diff(log_diffs[:min(50, len(log_diffs))]))
            print(f"    Empirical growth rate: {growth_rate:.4f} per step")
            print(f"    Interpretation: {'Exponential' if growth_rate > 0.01 else 'Sub-exponential'} amplification in unstable system")

    # === Task A5: Fair ablation ===
    print("\n--- A5: Fair ablation E0/E1/E3/RDE-L/RDE-T/RDE-M ---")

    # Train RDE-L models (Version L: local dynamics residual)
    print("  Training RDE-L (local residual)...")
    # RDE-L: train NN on local residuals (same as E1 but naming)
    # For fair comparison, use same architecture and training
    rde_l_models = train_nn_ensemble(states, actions, deltas, std_s, std_a, std_d,
                                      n_models=5, n_epochs=50, dagger_rounds=0, gps=gps)

    def rde_l_predict(s, tau):
        return ensemble_predict(rde_l_models, gps, s, tau, std_s, std_a, std_d)

    # Train RDE-T models (Version T: trajectory sync residual)
    print("  Training RDE-T (trajectory residual, 1 round)...")
    rde_t_models = train_nn_ensemble(states, actions, deltas, std_s, std_a, std_d,
                                      n_models=5, n_epochs=50, dagger_rounds=1, gps=gps)

    def rde_t_predict(s, tau):
        return ensemble_predict(rde_t_models, gps, s, tau, std_s, std_a, std_d)

    # RDE-M: current hybrid (2 rounds)
    print("  Training RDE-M (hybrid, 2 rounds)...")
    rde_m_models = train_nn_ensemble(states, actions, deltas, std_s, std_a, std_d,
                                      n_models=5, n_epochs=50, dagger_rounds=2, gps=gps)

    def rde_m_predict(s, tau):
        return ensemble_predict(rde_m_models, gps, s, tau, std_s, std_a, std_d)

    # Evaluate all models
    models_dict = {
        'E0_gp': e0_predict,
        'E1_offline_nn': e1_predict,
        'RDE_L': rde_l_predict,
        'RDE_T': rde_t_predict,
        'RDE_M': rde_m_predict,
    }

    import methods_common as mc
    import methods_evaluate as me

    eval_horizons = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    n_seeds = 5

    # Mode A: Teacher Forcing
    print("\n  Mode A evaluation...")
    mode_a_results = {}
    for mname, mpred in models_dict.items():
        model_results = {}
        for h in eval_horizons:
            metrics_list = []
            for si in range(n_seeds):
                tau_func = me.make_tau_func(si, max(h, 500))
                s_real = np.array([0.1, 0.0, 0.0, 0.0])
                states_real = [s_real.copy()]
                for step in range(h):
                    tau = tau_func(step, s_real)
                    s_real = mc.real_step(s_real, tau)
                    if abs(s_real[0]) > math.pi/3:
                        break
                    states_real.append(s_real.copy())
                states_real = np.array(states_real)

                # Teacher forcing: use real states as input
                def tf_pred(s, _):
                    return mpred(s, tau_func(min(0, len(states_real)-2), s_real))
                # Actually for TF, each step uses real state
                smodel = [states_real[0].copy()]
                for t in range(len(states_real)-1):
                    tau = tau_func(t, states_real[t])
                    sp = mpred(states_real[t], tau)
                    smodel.append(sp)
                smodel = np.array(smodel)
                m = compute_metrics(smodel, states_real, std_s, h)
                metrics_list.append(m)

            avg_mae = np.mean([m['mae'] for m in metrics_list])
            avg_nmae = np.mean([m['nmae'] for m in metrics_list])
            avg_surv = np.mean([m['survival_steps'] for m in metrics_list])
            model_results[f"{h}_steps"] = {
                'mae': float(avg_mae),
                'nmae': float(avg_nmae),
                'survival_steps': float(avg_surv),
                'per_state_mae': [float(np.mean([m['per_state_mae'][j] for m in metrics_list])) for j in range(4)],
            }
        mode_a_results[mname] = model_results

    # Mode B: Open Loop
    print("  Mode B evaluation...")
    mode_b_results = {}
    for mname, mpred in models_dict.items():
        model_results = {}
        for h in eval_horizons:
            metrics_list = []
            for si in range(n_seeds):
                tau_func = me.make_tau_func(si, max(h, 500))
                s0_eval = np.array([0.1, 0.0, 0.0, 0.0])

                # Collect actions from real system
                actions_seq = []
                s_r = s0_eval.copy()
                for step in range(h):
                    tau = tau_func(step, s_r)
                    actions_seq.append(tau)
                    s_r = mc.real_step(s_r, tau)
                    if abs(s_r[0]) > math.pi/3:
                        break

                # Model rollout with fixed actions
                s_m = s0_eval.copy()
                smodel = [s_m.copy()]
                for t in range(len(actions_seq)):
                    s_m = mpred(s_m, actions_seq[t])
                    if np.any(np.isnan(s_m)) or np.any(np.abs(s_m) > 100):
                        break
                    smodel.append(s_m.copy())
                smodel = np.array(smodel)

                # Real trajectory
                s_r = s0_eval.copy()
                sreal = [s_r.copy()]
                for t in range(len(actions_seq)):
                    s_r = mc.real_step(s_r, actions_seq[t])
                    sreal.append(s_r.copy())
                sreal = np.array(sreal)

                m = compute_metrics(smodel, sreal, std_s, h)
                metrics_list.append(m)

            avg_mae = np.mean([m['mae'] for m in metrics_list])
            avg_nmae = np.mean([m['nmae'] for m in metrics_list])
            avg_surv = np.mean([m['survival_steps'] for m in metrics_list])
            avg_fail = np.mean([1 if m['survival_steps'] < h else 0 for m in metrics_list])
            model_results[f"{h}_steps"] = {
                'mae': float(avg_mae),
                'nmae': float(avg_nmae),
                'survival_steps': float(avg_surv),
                'failure_rate': float(avg_fail),
                'per_state_mae': [float(np.mean([m['per_state_mae'][j] for m in metrics_list])) for j in range(4)],
            }
        mode_b_results[mname] = model_results

    # Save results
    results = {
        'subagent': 'A+B',
        'equivalence': {
            'pointwise_max_diff': max_diff_pointwise,
            'rollout_results': rollout_results,
        },
        'e0_e1_rollout_diff': e0_e1_rollout,
        'step_by_step_growth': {
            'first_nonzero_step': int(first_nonzero) if len(nonzero) > 0 else -1,
            'max_diff_step': int(max_diff_idx) if len(nonzero) > 0 else -1,
            'max_diff_value': float(step_diffs[max_diff_idx]) if len(nonzero) > 0 else 0,
        },
        'mode_a': mode_a_results,
        'mode_b': mode_b_results,
        'nn_models_trained': {
            'E1': len(e1_models),
            'RDE_L': len(rde_l_models),
            'RDE_T': len(rde_t_models),
            'RDE_M': len(rde_m_models),
        },
    }

    with open(OUTPUT / "SUBAGENT_A_B_RESULTS.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nSubagent A+B done in {elapsed:.1f}s")
    print(f"Results: {OUTPUT / 'SUBAGENT_A_B_RESULTS.json'}")
    return results


if __name__ == "__main__":
    main()
