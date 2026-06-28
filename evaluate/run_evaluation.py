"""完整评估执行脚本。

用法:
    python evaluate/run_evaluation.py [--quick] [--skip-gp] [--skip-ensemble]

完整评估: 30 seeds × 9 horizons × 8 models × 4 modes = 8640 次评估
快速模式: 5 seeds × 5 horizons × 8 models × 4 modes = 800 次评估
"""

import sys
import os
import time
import json
import argparse
import warnings
import numpy as np
from pathlib import Path

# 抑制GP收敛警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*lbfgs.*')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.eval_config import load_config
from evaluate.eval_models import (
    RealDynamics, LinearizedModel, GPStandard, GPB4Sparse,
    SINDy4D, GPEEnsemble
)
from evaluate.eval_modes import run_mode_a, run_mode_b, run_mode_c, run_mode_d
from evaluate.eval_metrics import compute_all_metrics
import methods_common as mc
import methods_evaluate as me


def main():
    parser = argparse.ArgumentParser(description='统一评估执行')
    parser.add_argument('--quick', action='store_true', help='快速模式 (5 seeds, 5 horizons)')
    parser.add_argument('--skip-gp', action='store_true', help='跳过GP模型')
    parser.add_argument('--skip-ensemble', action='store_true', help='跳过集成模型')
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'configs' / 'reproducible_world_model_evaluation.yaml'))
    parser.add_argument('--output', default=str(PROJECT_ROOT / 'results' / 'reproducible_evaluation_results.json'))
    args = parser.parse_args()

    print("=" * 60)
    print("统一评估执行")
    print("=" * 60)

    config = load_config(args.config)

    if args.quick:
        seeds = config['evaluation']['seeds'][:5]
        horizons = [10, 50, 100, 500, 1000]
        print(f"[快速模式] 5 seeds, 5 horizons")
    else:
        seeds = config['evaluation']['seeds']
        horizons = config['evaluation']['horizons']
        print(f"[完整模式] {len(seeds)} seeds, {len(horizons)} horizons")

    modes = ['A', 'B', 'C', 'D']
    state_std = np.array(config['normalization']['state_std'])
    K_lqr = np.array(config['evaluation']['lqr']['K_lqr'])

    # ---- 生成训练数据 ----
    print("\n[1/3] 生成训练数据...")
    t0 = time.time()
    states, actions, deltas, s_std, a_std, d_std = mc.generate_training_data(30000)
    print(f"  完成 ({time.time()-t0:.1f}s)")

    # ---- 构建模型 ----
    print("\n[2/3] 构建模型...")
    models = {}

    # 快速模型
    models['real_dynamics'] = RealDynamics()
    print("  real_dynamics: OK")

    models['linearized_model'] = LinearizedModel()
    print("  linearized_model: OK")

    # SINDy
    sindy = SINDy4D('meijaard_sindy_v35.npz')
    sindy.train(states, actions, deltas, s_std, a_std, d_std)
    models['sindy_4d'] = sindy
    print("  sindy_4d: OK")

    # GP
    if not args.skip_gp:
        print("  训练 gp_standard...")
        t0 = time.time()
        gp = GPStandard(max_samples=5000)
        gp.train(states, actions, deltas, s_std, a_std, d_std)
        models['gp_standard'] = gp
        print(f"    完成 ({time.time()-t0:.1f}s)")

        print("  训练 gp_b4_sparse...")
        t0 = time.time()
        gp_b4 = GPB4Sparse(n_components=500)
        gp_b4.train(states, actions, deltas, s_std, a_std, d_std)
        models['gp_b4_sparse'] = gp_b4
        print(f"    完成 ({time.time()-t0:.1f}s)")

    # 集成模型
    if not args.skip_ensemble:
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
            models[name] = ensemble
            print(f"    完成 ({time.time()-t0:.1f}s)")

    model_names = list(models.keys())
    print(f"\n  共 {len(model_names)} 个模型: {model_names}")

    # ---- 运行评估 ----
    print("\n[3/3] 运行评估...")
    total_evals = len(seeds) * len(horizons) * len(model_names) * len(modes)
    print(f"  总评估次数: {total_evals}")

    results_data = {}
    completed = 0
    t0 = time.time()

    for seed in seeds:
        seed_str = str(seed)
        results_data[seed_str] = {}

        for horizon in horizons:
            h_str = str(horizon)
            results_data[seed_str][h_str] = {}

            # 生成初始状态
            rng = np.random.RandomState(seed)
            phi_init = rng.uniform(-0.25, 0.25)
            s0 = np.array([phi_init, 0.0, 0.0, 0.0])
            tau_func = me.make_tau_func(seed - 42, max(horizon, 1000))

            for model_name in model_names:
                model = models[model_name]
                results_data[seed_str][h_str][model_name] = {}

                for mode in modes:
                    try:
                        if mode == 'A':
                            r = run_mode_a(model, s0, tau_func, horizon, mc.real_step)
                        elif mode == 'B':
                            r = run_mode_b(model, s0, tau_func, horizon, mc.real_step)
                        elif mode == 'C':
                            r = run_mode_c(model, s0, tau_func, horizon, mc.real_step)
                        elif mode == 'D':
                            r = run_mode_d(model, s0, horizon, mc.real_step, K_lqr,
                                           np.random.RandomState(seed))

                        metrics = compute_all_metrics(
                            r['states_model'], r['states_real'],
                            r.get('actions'), r.get('actions_real'),
                            state_std, 1000
                        )

                        results_data[seed_str][h_str][model_name][mode] = {
                            'metrics': metrics,
                            'survival_steps': r['survival_steps'],
                            'instability_detected': r['instability_detected'],
                        }
                    except Exception as e:
                        results_data[seed_str][h_str][model_name][mode] = {
                            'error': str(e),
                            'metrics': {},
                            'survival_steps': 0,
                            'instability_detected': True,
                        }

                    completed += 1
                    if completed % 100 == 0:
                        elapsed = time.time() - t0
                        rate = completed / elapsed
                        eta = (total_evals - completed) / rate
                        print(f"  进度: {completed}/{total_evals} "
                              f"({100*completed/total_evals:.1f}%) "
                              f"ETA: {eta:.0f}s")

    total_time = time.time() - t0
    print(f"\n  评估完成: {completed} 次, 耗时 {total_time:.1f}s")

    # ---- 保存结果 ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_results = {
        'config': {
            'seeds': seeds,
            'horizons': horizons,
            'models': model_names,
            'modes': modes,
        },
        'data': results_data,
        'total_time': total_time,
        'total_evaluations': completed,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, default=_json_default)
    print(f"\n  结果保存: {output_path}")

    # ---- 打印关键结果 ----
    print("\n" + "=" * 60)
    print("关键结果 (Mode C, 500步)")
    print("=" * 60)
    for model_name in model_names:
        mae_values = []
        for seed in seeds:
            entry = results_data.get(str(seed), {}).get('500', {}).get(model_name, {}).get('C', {})
            mae = entry.get('metrics', {}).get('overall', {}).get('mae')
            if mae is not None and not np.isnan(mae):
                mae_values.append(mae)
        if mae_values:
            print(f"  {model_name:30s} MAE={np.mean(mae_values):.6f} ± {np.std(mae_values):.6f} (n={len(mae_values)})")
        else:
            print(f"  {model_name:30s} MAE=N/A")


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == '__main__':
    main()
