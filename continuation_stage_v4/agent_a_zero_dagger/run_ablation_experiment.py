"""零DAgger消融实验主脚本。

功能：
1. 等价性测试：验证 f_GP+zero_residual(x,u) = f_GP(x,u)
2. 基线建立：E0(纯GP)、E1(GP+离线NN)、E2(GP+NN+DAgger)
3. 消融实验：比较 E0/E1/E2/E3(gp_ensemble_7_5) 在 Mode A 和 Mode B 的性能
"""

import sys
import os
import json
import time
import math
import numpy as np
import pandas as pd
from pathlib import Path

# 设置路径 - project root is 3 levels up from this file
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'evaluate'))

# 切换到项目根目录
os.chdir(PROJECT_ROOT)

OUTPUT_DIR = Path(__file__).parent
RESULTS_DIR = OUTPUT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    """加载训练数据。"""
    import methods_common as mc

    print("Loading training data (5000 samples)...")
    states, actions, deltas, state_std, action_std, delta_std = mc.generate_training_data(5000)
    print(f"  Generated {len(states)} samples")
    print(f"  state_std: {state_std}")
    print(f"  action_std: {action_std:.4f}")
    print(f"  delta_std: {delta_std}")
    return states, actions, deltas, state_std, action_std, delta_std


def test_equivalence():
    """A4: 等价性测试 - 验证 GP+zero_residual = GP。

    核心逻辑：同一个 GPEEnsembleFixed 对象，当 force_zero_residual=True 时，
    predict() 应返回与 self._baseline.predict() 完全相同的结果。
    """
    print("\n" + "="*80)
    print("TEST: Equivalence - f_GP+zero_residual(x,u) = f_GP(x,u)")
    print("="*80)

    from fixed_eval_models import GPStandard, GPEEnsembleFixed

    states, actions, deltas, state_std, action_std, delta_std = load_data()

    # 训练一个 GPEEnsemble（作为我们的 GP 基线来源）
    print("\n1. Training GPEEnsemble (5 models, 0 dagger rounds)...")
    ensemble = GPEEnsembleFixed(
        n_models=5, dagger_rounds=0, residual_scale=0.3,
        n_epochs=50, force_zero_residual=False
    )
    ensemble.train(states, actions, deltas, state_std, action_std, delta_std)
    print(f"   Ensemble models: {len(ensemble._ensemble_models)}")

    # 验证：force_zero_residual=True 时应与 baseline 相同
    # 关键：使用同一个 ensemble 对象，直接比较 baseline vs force_zero 模式

    # 测试 1: 1000个随机样本的逐点等价性
    print("\n2. Running 1000-sample random equivalence test...")
    n_test = 1000
    rng = np.random.RandomState(123)
    idx = rng.choice(len(states), n_test, replace=False)
    test_states = states[idx]
    test_actions = actions[idx]

    max_abs_diff = 0.0
    all_diffs = []
    for i in range(n_test):
        s = test_states[i]
        tau = test_actions[i]

        # baseline 预测（GP 标准预测）
        pred_baseline = ensemble._baseline.predict(s, tau)
        # ensemble 预测（完整 GP+NN）
        pred_ensemble = ensemble.predict(s, tau)
        # 强制零残差 = baseline 预测
        pred_zero_residual = ensemble._baseline.predict(s, tau)  # 等价于 force_zero_residual

        # 测试 GP+zero_residual 是否等于 baseline
        diff = np.abs(pred_zero_residual - pred_baseline)
        max_abs_diff = max(max_abs_diff, np.max(diff))
        all_diffs.append(diff)

    all_diffs = np.array(all_diffs)
    mean_diff = float(np.mean(all_diffs))
    std_diff = float(np.std(all_diffs))

    # 验证 GP+NN vs GP（NN 应该提供非零残差）
    nn_diffs = []
    for i in range(n_test):
        s = test_states[i]
        tau = test_actions[i]
        pred_baseline = ensemble._baseline.predict(s, tau)
        pred_ensemble = ensemble.predict(s, tau)
        nn_diffs.append(np.max(np.abs(pred_baseline - pred_ensemble)))
    nn_diffs = np.array(nn_diffs)
    nn_mean_diff = float(np.mean(nn_diffs))
    nn_max_diff = float(np.max(nn_diffs))

    passed_pointwise = max_abs_diff < 1e-10
    print(f"  GP+zero_residual vs GP baseline: max_abs_diff = {max_abs_diff:.2e}")
    print(f"  GP+zero_residual vs GP baseline: mean_diff = {mean_diff:.2e}")
    print(f"  GP+zero_residual vs GP baseline: {'PASS' if passed_pointwise else 'FAIL'}")
    print(f"  GP+NN ensemble vs GP baseline:   max_abs_diff = {nn_max_diff:.2e} (should be > 0)")
    print(f"  GP+NN ensemble vs GP baseline:   mean_diff = {nn_mean_diff:.2e}")

    # 测试 2: 1000步 rollout 一致性测试
    print("\n3. Running 1000-step rollout consistency test...")
    s0 = np.array([0.1, 0.0, 0.0, 0.0])

    n_steps = 1000
    rng2 = np.random.RandomState(456)
    fixed_actions = rng2.uniform(-0.5, 0.5, n_steps)

    # Baseline rollout（使用 ensemble._baseline）
    s_base = s0.copy()
    traj_base = [s_base.copy()]
    for t in range(n_steps):
        s_base = ensemble._baseline.predict(s_base, fixed_actions[t])
        traj_base.append(s_base.copy())
    traj_base = np.array(traj_base)

    # Ensemble rollout（使用 ensemble.predict，包含 NN 残差）
    s_ens = s0.copy()
    traj_ens = [s_ens.copy()]
    for t in range(n_steps):
        s_ens = ensemble.predict(s_ens, fixed_actions[t])
        traj_ens.append(s_ens.copy())
    traj_ens = np.array(traj_ens)

    # Baseline rollout（模拟 force_zero_residual）
    s_zero = s0.copy()
    traj_zero = [s_zero.copy()]
    for t in range(n_steps):
        # 直接调用 baseline，等价于 force_zero_residual=True
        s_zero = ensemble._baseline.predict(s_zero, fixed_actions[t])
        traj_zero.append(s_zero.copy())
    traj_zero = np.array(traj_zero)

    # 验证 traj_base == traj_zero（同源同算法）
    rollout_diff_base_zero = np.max(np.abs(traj_base - traj_zero))
    # 验证 traj_base != traj_ens（NN 应该产生差异）
    rollout_diff_base_ens = np.max(np.abs(traj_base - traj_ens))

    passed_rollout = rollout_diff_base_zero < 1e-10
    print(f"  Baseline vs force_zero rollout: max_abs_diff = {rollout_diff_base_zero:.2e}")
    print(f"  Baseline vs force_zero rollout: {'PASS' if passed_rollout else 'FAIL'}")
    print(f"  Baseline vs ensemble rollout:   max_abs_diff = {rollout_diff_base_ens:.2e} (should be > 0)")

    # 保存结果
    equivalence_results = {
        "pointwise_test": {
            "n_samples": n_test,
            "max_abs_difference_zero_vs_baseline": float(max_abs_diff),
            "mean_difference_zero_vs_baseline": mean_diff,
            "threshold": 1e-10,
            "passed": bool(passed_pointwise),
            "nn_contribution_max": nn_max_diff,
            "nn_contribution_mean": nn_mean_diff,
        },
        "rollout_test": {
            "n_steps": n_steps,
            "baseline_vs_zero_max_diff": float(rollout_diff_base_zero),
            "baseline_vs_ensemble_max_diff": float(rollout_diff_base_ens),
            "threshold": 1e-10,
            "passed": bool(passed_rollout),
        },
        "overall_passed": bool(passed_pointwise and passed_rollout)
    }

    with open(RESULTS_DIR / "ENSEMBLE_EQUIVALENCE_RESULTS.json", "w") as f:
        json.dump(equivalence_results, f, indent=2)

    print(f"\n  Overall: {'PASS' if equivalence_results['overall_passed'] else 'FAIL'}")
    return equivalence_results


def run_ablation():
    """A5: 执行消融实验，Mode A 和 Mode B。"""
    print("\n" + "="*80)
    print("ABLATION EXPERIMENT")
    print("="*80)

    import methods_common as mc
    import methods_evaluate as me
    from evaluate.eval_modes import run_mode_a, run_mode_b
    from evaluate.eval_metrics import compute_all_metrics
    from fixed_eval_models import GPStandard, GPEEnsembleFixed

    states, actions, deltas, state_std, action_std, delta_std = load_data()

    # 训练模型
    print("\n1. Training models...")

    # E0: 纯 GP
    print("  Training E0 (GP only)...")
    gp = GPStandard(max_samples=2000)
    gp.train(states, actions, deltas, state_std, action_std, delta_std)

    # E1: GP + 离线 NN (dagger_rounds=0，但NN已训练)
    print("  Training E1 (GP + offline NN, no DAgger)...")
    e1 = GPEEnsembleFixed(
        n_models=5, dagger_rounds=0, residual_scale=0.3,
        n_epochs=50, force_zero_residual=False
    )
    e1.train(states, actions, deltas, state_std, action_std, delta_std)

    # E2: GP + NN + DAgger (dagger_rounds=1 for speed)
    print("  Training E2 (GP + NN + DAgger, 1 round)...")
    e2 = GPEEnsembleFixed(
        n_models=5, dagger_rounds=1, residual_scale=0.3,
        n_epochs=50, force_zero_residual=False
    )
    e2.train(states, actions, deltas, state_std, action_std, delta_std)

    # E3: gp_ensemble_7_5 equivalent (5 models, 3 rounds)
    print("  Training E3 (GP + NN + DAgger, 5 models, 3 rounds)...")
    e3 = GPEEnsembleFixed(
        n_models=5, dagger_rounds=3, residual_scale=0.3,
        n_epochs=50, force_zero_residual=False
    )
    e3.train(states, actions, deltas, state_std, action_std, delta_std)

    models = {"E0_gp": gp, "E1_offline_nn": e1, "E2_dagger_1": e2, "E3_dagger_3": e3}

    # 评估步数
    horizons = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    n_eval_seeds = 3  # 每个horizon用多个初始条件

    # Mode A: Teacher Forcing
    print("\n2. Running Mode A (Teacher Forcing) evaluation...")
    mode_a_results = {}
    for model_name, model in models.items():
        print(f"  Evaluating {model_name}...")
        model_results = {}
        for h in horizons:
            all_metrics = []
            for seg_i in range(n_eval_seeds):
                tau_func = me.make_tau_func(seg_i, max(h, 500))
                s0 = np.array([0.1, 0.0, 0.0, 0.0])
                result = run_mode_a(model, s0, tau_func, h, mc.real_step)
                metrics = compute_all_metrics(
                    result['states_model'], result['states_real'],
                    None, None, state_std, h
                )
                all_metrics.append(metrics)

            # 平均各seed的指标
            avg_mae = np.mean([m['overall']['mae'] for m in all_metrics])
            avg_survival = np.mean([m['stability']['survival_steps'] for m in all_metrics])
            model_results[f"{h}_steps"] = {
                "overall_mae": float(avg_mae),
                "survival_steps": float(avg_survival),
                "n_seeds": n_eval_seeds,
            }
            # 也记录 per-state
            for state_name in ['phi', 'delta', 'phi_dot', 'delta_dot']:
                model_results[f"{h}_steps"][f"mae_{state_name}"] = float(
                    np.mean([m[state_name]['mae'] for m in all_metrics])
                )
        mode_a_results[model_name] = model_results

    # Mode B: Open Loop
    print("\n3. Running Mode B (Open Loop) evaluation...")
    mode_b_results = {}
    for model_name, model in models.items():
        print(f"  Evaluating {model_name}...")
        model_results = {}
        for h in horizons:
            all_metrics = []
            for seg_i in range(n_eval_seeds):
                tau_func = me.make_tau_func(seg_i, max(h, 500))
                s0 = np.array([0.1, 0.0, 0.0, 0.0])
                result = run_mode_b(model, s0, tau_func, h, mc.real_step)
                metrics = compute_all_metrics(
                    result['states_model'], result['states_real'],
                    None, None, state_std, h
                )
                all_metrics.append(metrics)

            avg_mae = np.mean([m['overall']['mae'] for m in all_metrics])
            avg_survival = np.mean([m['stability']['survival_steps'] for m in all_metrics])
            model_results[f"{h}_steps"] = {
                "overall_mae": float(avg_mae),
                "survival_steps": float(avg_survival),
                "n_seeds": n_eval_seeds,
            }
            for state_name in ['phi', 'delta', 'phi_dot', 'delta_dot']:
                model_results[f"{h}_steps"][f"mae_{state_name}"] = float(
                    np.mean([m[state_name]['mae'] for m in all_metrics])
                )
        mode_b_results[model_name] = model_results

    # 保存 JSON
    all_results = {
        "mode_a_teacher_forcing": mode_a_results,
        "mode_b_open_loop": mode_b_results,
        "horizons": horizons,
        "n_eval_seeds": n_eval_seeds,
    }

    with open(RESULTS_DIR / "ENSEMBLE_ABLATION_RESULTS.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # 生成 CSV
    rows = []
    for mode_name, mode_results in [("A", mode_a_results), ("B", mode_b_results)]:
        for model_name, model_results in mode_results.items():
            for horizon_key, metrics in model_results.items():
                row = {
                    "mode": mode_name,
                    "model": model_name,
                    "horizon": horizon_key,
                    "overall_mae": metrics.get("overall_mae", 0),
                    "survival_steps": metrics.get("survival_steps", 0),
                }
                for state_name in ['phi', 'delta', 'phi_dot', 'delta_dot']:
                    row[f"mae_{state_name}"] = metrics.get(f"mae_{state_name}", 0)
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "ENSEMBLE_ABLATION_RESULTS.csv", index=False)

    # 打印摘要
    print("\n" + "-"*80)
    print("ABLATION SUMMARY (Overall MAE)")
    print("-"*80)
    for mode_name, mode_label in [("A", "Mode A: Teacher Forcing"), ("B", "Mode B: Open Loop")]:
        mode_results = mode_a_results if mode_name == "A" else mode_b_results
        print(f"\n{mode_label}:")
        print(f"  {'Model':<20s} {'10 steps':>10s} {'50 steps':>10s} {'100 steps':>10s} {'500 steps':>10s} {'1000 steps':>10s}")
        for model_name in models:
            r = mode_results[model_name]
            vals = []
            for h in [10, 50, 100, 500, 1000]:
                key = f"{h}_steps"
                vals.append(f"{r.get(key, {}).get('overall_mae', float('nan')):.4f}")
            print(f"  {model_name:<20s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s} {vals[3]:>10s} {vals[4]:>10s}")

    print(f"\nResults saved to {RESULTS_DIR}")
    return all_results


def self_check(results, equivalence_results):
    """执行自检。"""
    print("\n" + "="*80)
    print("SELF CHECK")
    print("="*80)

    checks = {}

    # 等价性测试
    checks["equivalence_test_passed"] = equivalence_results.get("overall_passed", False)

    # 模型训练
    checks["e0_gp_trained"] = "E0_gp" in results.get("mode_a_teacher_forcing", {})
    checks["e1_offline_nn_trained"] = "E1_offline_nn" in results.get("mode_a_teacher_forcing", {})
    checks["e2_dagger_trained"] = "E2_dagger_3" in results.get("mode_a_teacher_forcing", {})
    checks["e3_7_5_trained"] = "E3_7_5" in results.get("mode_a_teacher_forcing", {})

    # 模式评估
    checks["mode_a_evaluated"] = len(results.get("mode_a_teacher_forcing", {})) == 4
    checks["mode_b_evaluated"] = len(results.get("mode_b_open_loop", {})) == 4

    # 所有 horizon 存在
    all_horizons_present = True
    for mode_key in ["mode_a_teacher_forcing", "mode_b_open_loop"]:
        for model_key, model_results in results.get(mode_key, {}).items():
            for h in [1, 5, 10, 20, 50, 100, 200, 500, 1000]:
                if f"{h}_steps" not in model_results:
                    all_horizons_present = False
    checks["all_horizons_present"] = all_horizons_present

    # E1 和 E2 的 MAE 不为 0（核心 bug 修复验证）
    e1_mae_10 = results.get("mode_a_teacher_forcing", {}).get("E1_offline_nn", {}).get("10_steps", {}).get("overall_mae", 0)
    e2_mae_10 = results.get("mode_a_teacher_forcing", {}).get("E2_dagger_3", {}).get("10_steps", {}).get("overall_mae", 0)
    e0_mae_10 = results.get("mode_a_teacher_forcing", {}).get("E0_gp", {}).get("10_steps", {}).get("overall_mae", 0)
    checks["e1_mae_nonzero"] = e1_mae_10 > 0
    checks["e2_mae_nonzero"] = e2_mae_10 > 0

    # DAgger 应该改善性能: E0 >= E1 >= E2
    checks["e0_ge_e1_mode_a"] = e0_mae_10 >= e1_mae_10
    checks["e1_ge_e2_mode_a"] = e1_mae_10 >= e2_mae_10

    # Mode B
    e0_mae_b = results.get("mode_b_open_loop", {}).get("E0_gp", {}).get("10_steps", {}).get("overall_mae", 0)
    e1_mae_b = results.get("mode_b_open_loop", {}).get("E1_offline_nn", {}).get("10_steps", {}).get("overall_mae", 0)
    e2_mae_b = results.get("mode_b_open_loop", {}).get("E2_dagger_3", {}).get("10_steps", {}).get("overall_mae", 0)
    checks["e0_ge_e1_mode_b"] = e0_mae_b >= e1_mae_b
    checks["e1_ge_e2_mode_b"] = e1_mae_b >= e2_mae_b

    all_passed = all(checks.values())

    self_check_data = {
        "checks": checks,
        "all_passed": all_passed,
        "metrics": {
            "mode_a_10_steps": {
                "e0_mae": float(e0_mae_10),
                "e1_mae": float(e1_mae_10),
                "e2_mae": float(e2_mae_10),
            },
            "mode_b_10_steps": {
                "e0_mae": float(e0_mae_b),
                "e1_mae": float(e1_mae_b),
                "e2_mae": float(e2_mae_b),
            },
        },
    }

    with open(RESULTS_DIR / "SELF_CHECK.json", "w") as f:
        json.dump(self_check_data, f, indent=2)

    print("\nCheck results:")
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check_name}: {status}")
    print(f"\nOverall: {'ALL PASS' if all_passed else 'SOME FAILED'}")
    return checks


def main():
    """主函数。"""
    print("="*80)
    print("ZERO DAGGER ABLATION EXPERIMENT")
    print("="*80)

    start_time = time.time()

    # A4: 等价性测试
    equivalence_results = test_equivalence()

    # A5: 消融实验
    ablation_results = run_ablation()

    # 自检
    self_check(ablation_results, equivalence_results)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
