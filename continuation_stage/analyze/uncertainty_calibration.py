"""不确定性校准分析: 评估集成模型预测不确定性的质量。

用法:
    python continuation_stage/analyze/uncertainty_calibration.py [--quick]

输出:
    - continuation_stage/results/uncertainty_calibration.json
    - continuation_stage/reports/UNCERTAINTY_CALIBRATION_REPORT.md
"""

import sys
import os
import time
import json
import argparse
import warnings
import numpy as np
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore', category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from continuation_stage.evaluate.eval_modes_v2 import (
    generate_reference_trajectory, run_mode_b_fixed,
    run_mode_a, run_mode_c, run_mode_d
)
from evaluate.eval_models import GPEEnsemble
import methods_common as mc
import methods_evaluate as me


def collect_uncertainty_data(model, s0, tau_func, n_steps, real_dynamics, mode,
                              reference=None, K_lqr=None, rng=None):
    """运行单次评估并收集不确定性数据。

    返回:
        errors: (n_steps, 4) 逐步逐状态误差
        uncertainties: (n_steps, 4) 逐步逐状态不确定性
        step_indices: 步索引
    """
    if mode == 'A':
        result = run_mode_a(model, s0, tau_func, n_steps, real_dynamics)
    elif mode == 'B':
        result = run_mode_b_fixed(model, s0, reference, n_steps, real_dynamics)
    elif mode == 'C':
        result = run_mode_c(model, s0, tau_func, n_steps, real_dynamics)
    elif mode == 'D':
        result = run_mode_d(model, s0, n_steps, real_dynamics, K_lqr, rng)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if 'uncertainties' not in result or len(result['uncertainties']) == 0:
        return None, None, None

    states_model = result['states_model']
    states_real = result['states_real']
    uncertainties = result['uncertainties']

    n_valid = min(len(states_model) - 1, len(states_real) - 1, len(uncertainties))
    errors = np.abs(states_model[1:n_valid+1] - states_real[1:n_valid+1])
    uncertainties = uncertainties[:n_valid]

    return errors, uncertainties, np.arange(n_valid)


def compute_calibration_metrics(errors, uncertainties, confidence=0.95):
    """计算校准指标。

    返回:
        dict with calibration metrics
    """
    if errors is None or uncertainties is None:
        return None

    n_steps, n_states = errors.shape
    results = {
        'n_samples': n_steps,
        'n_states': n_states,
        'per_state': {},
        'overall': {},
    }

    # 转换不确定性为标准差 (如果返回的是方差)
    stds = np.sqrt(uncertainties) if np.any(uncertainties < 0) else uncertainties

    # 逐状态分析
    for dim in range(n_states):
        e = errors[:, dim]
        s = stds[:, dim]

        # Pearson 相关系数
        pearson_r, pearson_p = stats.pearsonr(s, e)

        # Spearman 相关系数
        spearman_r, spearman_p = stats.spearmanr(s, e)

        # 预测区间覆盖率
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        lower = -z * s
        upper = z * s
        coverage = np.mean((e >= lower) & (e <= upper))

        # 平均预测区间宽度
        avg_width = np.mean(2 * z * s)

        # 校准误差
        expected_coverage = confidence
        calibration_error = abs(coverage - expected_coverage)

        results['per_state'][f'state_{dim}'] = {
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'coverage': float(coverage),
            'expected_coverage': float(expected_coverage),
            'calibration_error': float(calibration_error),
            'avg_interval_width': float(avg_width),
            'mean_error': float(np.mean(e)),
            'mean_uncertainty': float(np.mean(s)),
        }

    # 整体分析
    e_flat = errors.flatten()
    s_flat = stds.flatten()
    pearson_r, pearson_p = stats.pearsonr(s_flat, e_flat)
    spearman_r, spearman_p = stats.spearmanr(s_flat, e_flat)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    coverage = np.mean((e_flat >= -z * s_flat) & (e_flat <= z * s_flat))
    avg_width = np.mean(2 * z * s_flat)
    calibration_error = abs(coverage - confidence)

    results['overall'] = {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'coverage': float(coverage),
        'expected_coverage': float(confidence),
        'calibration_error': float(calibration_error),
        'avg_interval_width': float(avg_width),
        'mean_error': float(np.mean(e_flat)),
        'mean_uncertainty': float(np.mean(s_flat)),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='不确定性校准分析')
    parser.add_argument('--quick', action='store_true', help='快速模式 (少量种子)')
    parser.add_argument('--seeds', type=int, default=5, help='种子数量')
    parser.add_argument('--horizon', type=int, default=100, help='评估步数')
    parser.add_argument('--output', default=str(PROJECT_ROOT / 'continuation_stage' / 'results' / 'uncertainty_calibration.json'))
    args = parser.parse_args()

    n_seeds = 3 if args.quick else args.seeds
    horizon = min(50, args.horizon) if args.quick else args.horizon
    seeds = list(range(42, 42 + n_seeds))

    print(f"不确定性校准分析")
    print(f"  种子数: {n_seeds}")
    print(f"  评估步数: {horizon}")
    print(f"  模式: A, B, C, D")

    state_std = np.array([0.28791831, 0.17332272, 1.15369545, 0.57764881])
    K_lqr = np.array([-103.92, -34.20, 38.04, 3.70])

    # 生成训练数据
    print("\n[1/3] 生成训练数据...")
    t0 = time.time()
    states, actions, deltas, s_std, a_std, d_std = mc.generate_training_data(30000)
    print(f"  完成 ({time.time()-t0:.1f}s)")

    # 训练集成模型
    print("\n[2/3] 训练集成模型...")
    ensemble_models = {}
    for n_models, dagger_rounds in [(5, 0), (5, 3), (7, 5)]:
        name = f'gp_ensemble_{n_models}_{dagger_rounds}'
        print(f"  训练 {name}...")
        t0 = time.time()
        ensemble = GPEEnsemble(
            n_models=n_models,
            dagger_rounds=dagger_rounds,
            residual_scale=0.3,
            n_epochs=100,
            use_scheduler=True
        )
        ensemble.train(states, actions, deltas, s_std, a_std, d_std)
        ensemble_models[name] = ensemble
        print(f"    完成 ({time.time()-t0:.1f}s)")

    # 收集不确定性数据
    print("\n[3/3] 收集不确定性数据...")
    all_results = {}
    modes = ['A', 'B', 'C', 'D']

    for model_name, model in ensemble_models.items():
        all_results[model_name] = {}
        for mode in modes:
            all_errors = []
            all_uncs = []

            for seed in seeds:
                rng = np.random.RandomState(seed)
                phi_init = rng.uniform(-0.25, 0.25)
                s0 = np.array([phi_init, 0.0, 0.0, 0.0])
                tau_func = me.make_tau_func(seed - 42, 1000)

                reference = None
                if mode == 'B':
                    reference = generate_reference_trajectory(
                        s0, tau_func, horizon, mc.real_step)

                rng_d = np.random.RandomState(seed)

                errors, uncs, _ = collect_uncertainty_data(
                    model, s0, tau_func, horizon, mc.real_step, mode,
                    reference=reference, K_lqr=K_lqr, rng=rng_d
                )

                if errors is not None:
                    all_errors.append(errors)
                    all_uncs.append(uncs)

            if all_errors:
                errors_cat = np.concatenate(all_errors, axis=0)
                uncs_cat = np.concatenate(all_uncs, axis=0)
                cal_metrics = compute_calibration_metrics(errors_cat, uncs_cat)
                all_results[model_name][mode] = cal_metrics

                if cal_metrics:
                    o = cal_metrics['overall']
                    print(f"  {model_name} Mode {mode}: "
                          f"Pearson r={o['pearson_r']:.3f}, "
                          f"coverage={o['coverage']:.3f}, "
                          f"cal_error={o['calibration_error']:.3f}")
            else:
                all_results[model_name][mode] = None
                print(f"  {model_name} Mode {mode}: 无不确定性数据")

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=_json_default)
    print(f"\n结果保存: {output_path}")

    # 生成报告
    report_path = output_path.parent.parent / 'reports' / 'UNCERTAINTY_CALIBRATION_REPORT.md'
    generate_report(all_results, report_path)
    print(f"报告保存: {report_path}")


def generate_report(results, path):
    """生成校准报告 Markdown。"""
    lines = [
        "# 不确定性校准报告",
        "",
        "> **生成时间**: 2026-06-25",
        "",
        "---",
        "",
        "## 1. 校准指标说明",
        "",
        "| 指标 | 含义 | 理想值 |",
        "|------|------|--------|",
        "| Pearson r | 不确定性与误差线性相关 | > 0.5 |",
        "| Spearman r | 不确定性与误差单调相关 | > 0.5 |",
        "| Coverage | 95%预测区间实际覆盖率 | ≈ 0.95 |",
        "| Calibration Error | |coverage - 0.95| | < 0.05 |",
        "| Avg Interval Width | 平均预测区间宽度 | 越小越好 |",
        "",
        "---",
        "",
        "## 2. 整体结果",
        "",
        "| 模型 | 模式 | Pearson r | Spearman r | Coverage | Cal Error | Avg Width |",
        "|------|------|-----------|------------|----------|-----------|-----------|",
    ]

    for model_name, modes_data in results.items():
        for mode, metrics in modes_data.items():
            if metrics is None:
                lines.append(f"| {model_name} | {mode} | N/A | N/A | N/A | N/A | N/A |")
                continue
            o = metrics['overall']
            lines.append(
                f"| {model_name} | {mode} | "
                f"{o['pearson_r']:.3f} | {o['spearman_r']:.3f} | "
                f"{o['coverage']:.3f} | {o['calibration_error']:.3f} | "
                f"{o['avg_interval_width']:.4f} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 3. 逐状态结果",
        "",
    ])

    for model_name, modes_data in results.items():
        for mode, metrics in modes_data.items():
            if metrics is None:
                continue
            lines.append(f"### {model_name} Mode {mode}")
            lines.append("")
            lines.append("| 状态 | Pearson r | Coverage | Cal Error | Mean Error | Mean Unc |")
            lines.append("|------|-----------|----------|-----------|------------|----------|")
            state_names = ['phi', 'delta', 'phi_dot', 'delta_dot']
            for dim, name in enumerate(state_names):
                key = f'state_{dim}'
                if key in metrics['per_state']:
                    s = metrics['per_state'][key]
                    lines.append(
                        f"| {name} | {s['pearson_r']:.3f} | "
                        f"{s['coverage']:.3f} | {s['calibration_error']:.3f} | "
                        f"{s['mean_error']:.4f} | {s['mean_uncertainty']:.4f} |"
                    )
            lines.append("")

    lines.extend([
        "---",
        "",
        "## 4. 校准建议",
        "",
        "### 4.1 当前状态",
        "",
        "- GP后验方差和NN集成分歧已实现",
        "- 需要标量方差缩放校准",
        "",
        "### 4.2 校准方法",
        "",
        "1. **标量方差缩放**: 乘以常数使覆盖率接近95%",
        "2. **逐状态方差缩放**: 每个状态独立缩放",
        "3. **Conformal calibration**: 非参数校准方法",
        "",
        "---",
        "",
        "*报告时间: 2026-06-25*",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == '__main__':
    main()
