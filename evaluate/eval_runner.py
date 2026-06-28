"""主执行入口：训练模型、运行评估、保存结果。"""

import sys
import os
import json
import time
import numpy as np
import math
from pathlib import Path
from typing import Optional

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.eval_config import (
    load_config, get_seeds, get_horizons, get_model_configs,
    get_state_std, get_action_std, get_delta_std, get_dt,
    get_max_steps, get_lqr_K, get_phi_limit
)
from evaluate.eval_models import build_model, BaseModel, RealDynamics, GPEEnsemble
from evaluate.eval_modes import run_mode_a, run_mode_b, run_mode_c, run_mode_d
from evaluate.eval_metrics import compute_all_metrics, compute_uncertainty_metrics


def train_all_models(config: dict) -> dict:
    """训练所有需要训练的模型，返回模型字典。"""
    import methods_common as mc

    print("=" * 60)
    print("训练所有模型")
    print("=" * 60)

    # 生成训练数据
    print("  生成训练数据 (30000样本)...")
    states, actions, deltas, state_std, action_std, delta_std = mc.generate_training_data(30000)
    print(f"  state_std: {state_std}")
    print(f"  action_std: {action_std:.4f}")
    print(f"  delta_std: {delta_std}")

    models = {}
    model_configs = get_model_configs(config)

    for mc_cfg in model_configs:
        name = mc_cfg['name']
        model_type = mc_cfg['type']
        print(f"\n  构建模型: {name} (type={model_type})")

        if model_type == 'reference':
            model = build_model(mc_cfg, config)
            models[name] = model
            print(f"    [跳过训练] 参考模型")
            continue

        model = build_model(mc_cfg, config)

        # 需要训练的模型
        if hasattr(model, 'train'):
            t0 = time.time()
            model.train(states, actions, deltas, state_std, action_std, delta_std)
            elapsed = time.time() - t0
            print(f"    训练完成 ({elapsed:.1f}s)")

        models[name] = model

    # 存储归一化常数
    models['_norm'] = {
        'state_std': state_std,
        'action_std': action_std,
        'delta_std': delta_std,
    }

    print(f"\n  共训练 {len(models) - 1} 个模型")
    return models


def run_single_evaluation(model: BaseModel, mode: str, seed: int,
                          horizon: int, config: dict, models: dict) -> dict:
    """运行单次评估。

    Args:
        model: 模型实例
        mode: 'A', 'B', 'C', 'D'
        seed: 随机种子
        horizon: 评估步数
        config: 配置字典
        models: 模型字典 (含归一化常数)

    Returns:
        dict: 评估结果和指标
    """
    import methods_evaluate as me
    import methods_common as mc

    rng = np.random.RandomState(seed)

    # 生成初始状态
    phi_init = rng.uniform(-0.25, 0.25)
    s0 = np.array([phi_init, 0.0, 0.0, 0.0])

    # 创建 tau_func
    seg_idx = seed - 42  # seeds从42开始
    tau_func = me.make_tau_func(seg_idx, max(horizon, 1000))

    K_lqr = get_lqr_K(config)
    max_steps = get_max_steps(config)

    # 运行评估模式
    if mode == 'A':
        result = run_mode_a(model, s0, tau_func, horizon, mc.real_step)
    elif mode == 'B':
        result = run_mode_b(model, s0, tau_func, horizon, mc.real_step)
    elif mode == 'C':
        result = run_mode_c(model, s0, tau_func, horizon, mc.real_step)
    elif mode == 'D':
        result = run_mode_d(model, s0, horizon, mc.real_step, K_lqr, rng)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 计算指标
    norm = models['_norm']
    metrics = compute_all_metrics(
        result['states_model'], result['states_real'],
        result.get('actions'), result.get('actions_real'),
        norm['state_std'], max_steps
    )

    # 不确定性指标
    if 'uncertainties' in result and len(result['uncertainties']) > 0:
        abs_errors = np.abs(result['states_model'] - result['states_real'])
        unc_metrics = compute_uncertainty_metrics(result['uncertainties'], abs_errors)
        metrics['uncertainty'] = unc_metrics

    result['metrics'] = metrics
    return result


def run_full_evaluation(config: dict, models: dict,
                        progress_callback=None) -> dict:
    """运行完整评估: 30 seed × 9 horizon × N model × 4 mode。

    Returns:
        dict: 完整评估结果
    """
    seeds = get_seeds(config)
    horizons = get_horizons(config)
    model_configs = get_model_configs(config)
    modes = ['A', 'B', 'C', 'D']

    total = len(seeds) * len(horizons) * len(model_configs) * len(modes)
    completed = 0
    t0 = time.time()

    results = {
        'config': {
            'seeds': seeds,
            'horizons': horizons,
            'models': [mc['name'] for mc in model_configs],
            'modes': modes,
        },
        'data': {},
    }

    for seed in seeds:
        results['data'][str(seed)] = {}
        for horizon in horizons:
            results['data'][str(seed)][str(horizon)] = {}
            for mc_cfg in model_configs:
                model_name = mc_cfg['name']
                model = models[model_name]
                results['data'][str(seed)][str(horizon)][model_name] = {}

                for mode in modes:
                    try:
                        eval_result = run_single_evaluation(
                            model, mode, seed, horizon, config, models
                        )
                        results['data'][str(seed)][str(horizon)][model_name][mode] = {
                            'metrics': eval_result['metrics'],
                            'survival_steps': eval_result['survival_steps'],
                            'instability_detected': eval_result['instability_detected'],
                        }
                    except Exception as e:
                        results['data'][str(seed)][str(horizon)][model_name][mode] = {
                            'error': str(e),
                            'metrics': {},
                            'survival_steps': 0,
                            'instability_detected': True,
                        }

                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total)
                    elif completed % 100 == 0:
                        elapsed = time.time() - t0
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (total - completed) / rate if rate > 0 else 0
                        print(f"  进度: {completed}/{total} "
                              f"({100*completed/total:.1f}%) "
                              f"ETA: {eta:.0f}s")

    results['total_time'] = time.time() - t0
    return results


def compute_summary_statistics(results: dict) -> dict:
    """从完整结果中计算汇总统计。"""
    data = results['data']
    model_names = results['config']['models']
    modes = results['config']['modes']
    horizons = results['config']['horizons']
    seeds = results['config']['seeds']

    summary = {}

    for model_name in model_names:
        summary[model_name] = {}
        for mode in modes:
            summary[model_name][mode] = {}
            for horizon in horizons:
                h_str = str(horizon)
                # 收集所有seed的指标
                mae_values = []
                survival_values = []
                for seed in seeds:
                    s_str = str(seed)
                    entry = data.get(s_str, {}).get(h_str, {}).get(model_name, {}).get(mode, {})
                    if 'error' in entry:
                        continue
                    metrics = entry.get('metrics', {})
                    overall = metrics.get('overall', {})
                    if 'mae' in overall and not math.isnan(overall['mae']):
                        mae_values.append(overall['mae'])
                    survival_values.append(entry.get('survival_steps', 0))

                if mae_values:
                    summary[model_name][mode][h_str] = {
                        'mae_mean': float(np.mean(mae_values)),
                        'mae_std': float(np.std(mae_values)),
                        'mae_median': float(np.median(mae_values)),
                        'mae_min': float(np.min(mae_values)),
                        'mae_max': float(np.max(mae_values)),
                        'n_valid': len(mae_values),
                        'survival_mean': float(np.mean(survival_values)),
                        'survival_min': float(np.min(survival_values)),
                        'instability_rate': float(np.mean([
                            1 if s < horizon else 0 for s in survival_values
                        ])),
                    }
                else:
                    summary[model_name][mode][h_str] = {
                        'mae_mean': float('nan'),
                        'mae_std': float('nan'),
                        'n_valid': 0,
                    }

    return summary


def save_results(results: dict, summary: dict, config: dict):
    """保存评估结果到文件。"""
    output = config['output']
    results_dir = PROJECT_ROOT / output['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整结果 (不含轨迹数据以节省空间)
    lite_results = _make_lite_results(results)
    fpath = results_dir / 'reproducible_evaluation_results.json'
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(lite_results, f, indent=2, default=_json_default)
    print(f"  保存: {fpath}")

    # 保存汇总统计
    fpath = results_dir / 'summary_statistics.json'
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"  保存: {fpath}")

    # 保存 per-seed CSV
    _save_per_seed_csv(results, summary, results_dir)

    # 保存 model comparison CSV
    _save_model_comparison_csv(summary, results_dir, config)


def _make_lite_results(results: dict) -> dict:
    """移除轨迹数据，只保留指标。"""
    import copy
    lite = copy.deepcopy(results)
    # data结构中只有metrics和survival，不含轨迹
    return lite


def _save_per_seed_csv(results: dict, summary: dict, results_dir: Path):
    """保存per-seed结果CSV。"""
    import csv
    fpath = results_dir / 'per_seed_results.csv'
    with open(fpath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'horizon', 'model', 'mode',
                         'mae', 'survival_steps', 'instability_detected'])

        for seed_str, seed_data in results['data'].items():
            for h_str, h_data in seed_data.items():
                for model_name, model_data in h_data.items():
                    for mode, mode_data in model_data.items():
                        metrics = mode_data.get('metrics', {})
                        overall = metrics.get('overall', {})
                        writer.writerow([
                            seed_str, h_str, model_name, mode,
                            overall.get('mae', ''),
                            mode_data.get('survival_steps', ''),
                            mode_data.get('instability_detected', ''),
                        ])
    print(f"  保存: {fpath}")


def _save_model_comparison_csv(summary: dict, results_dir: Path, config: dict):
    """保存模型对比CSV。"""
    import csv
    fpath = results_dir / 'model_comparison.csv'
    horizons = config['evaluation']['horizons']
    modes = ['A', 'B', 'C', 'D']

    with open(fpath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'mode', 'horizon', 'mae_mean', 'mae_std',
                         'n_valid', 'survival_mean', 'instability_rate'])

        for model_name, model_data in summary.items():
            for mode in modes:
                mode_data = model_data.get(mode, {})
                for horizon in horizons:
                    h_str = str(horizon)
                    h_data = mode_data.get(h_str, {})
                    writer.writerow([
                        model_name, mode, horizon,
                        h_data.get('mae_mean', ''),
                        h_data.get('mae_std', ''),
                        h_data.get('n_valid', ''),
                        h_data.get('survival_mean', ''),
                        h_data.get('instability_rate', ''),
                    ])
    print(f"  保存: {fpath}")


def _json_default(obj):
    """JSON序列化默认处理。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ============================================================
# CLI 入口
# ============================================================
def main():
    """命令行入口。"""
    import argparse
    parser = argparse.ArgumentParser(description='统一评估框架')
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'configs' / 'reproducible_world_model_evaluation.yaml'))
    parser.add_argument('--models', nargs='+', default=None, help='只评估指定模型')
    parser.add_argument('--modes', nargs='+', default=None, help='只运行指定模式 (A/B/C/D)')
    parser.add_argument('--seeds', nargs='+', type=int, default=None, help='只使用指定种子')
    parser.add_argument('--horizons', nargs='+', type=int, default=None, help='只评估指定时域')
    parser.add_argument('--dry-run', action='store_true', help='只训练模型不评估')
    args = parser.parse_args()

    print("=" * 60)
    print("统一评估框架")
    print("=" * 60)

    config = load_config(args.config)
    print(f"配置: {args.config}")

    # 训练模型
    models = train_all_models(config)

    if args.dry_run:
        print("\n[Dry Run] 跳过评估")
        return

    # 运行评估
    print("\n" + "=" * 60)
    print("运行评估")
    print("=" * 60)

    results = run_full_evaluation(config, models)

    # 计算汇总
    summary = compute_summary_statistics(results)

    # 保存结果
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)
    save_results(results, summary, config)

    # 打印关键结果
    print("\n" + "=" * 60)
    print("关键结果 (Mode C, 500步)")
    print("=" * 60)
    for model_name in results['config']['models']:
        h500 = summary.get(model_name, {}).get('C', {}).get('500', {})
        mae = h500.get('mae_mean', float('nan'))
        std = h500.get('mae_std', float('nan'))
        n = h500.get('n_valid', 0)
        print(f"  {model_name:30s} MAE={mae:.6f} ± {std:.6f} (n={n})")

    print(f"\n总耗时: {results['total_time']:.1f}s")


if __name__ == '__main__':
    main()
