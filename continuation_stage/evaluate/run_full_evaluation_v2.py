"""完整评估 v2: 修复 Mode B，统一评估所有模型。

用法:
    python continuation_stage/evaluate/run_full_evaluation_v2.py [--quick]

包含:
- Mode B 正确性验证
- 7个模型 × 4种模式 × 多时域
- 逐状态指标
"""

import sys
import os
import time
import json
import argparse
import warnings
import numpy as np
import math
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*lbfgs.*')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from continuation_stage.evaluate.eval_modes_v2 import (
    generate_reference_trajectory, run_mode_b_fixed,
    run_mode_a, run_mode_c, run_mode_d
)
from evaluate.eval_models import (
    RealDynamics, LinearizedModel, GPStandard, GPB4Sparse,
    SINDy4D, GPEEnsemble
)
from evaluate.eval_metrics import compute_all_metrics
import methods_common as mc
import methods_evaluate as me


def verify_mode_b_correctness(s0, tau_func, n_steps, real_dynamics):
    """验证 Mode B 实现正确性。"""
    print("\n" + "=" * 60)
    print("Mode B 正确性验证")
    print("=" * 60)

    # 生成参考轨迹
    ref = generate_reference_trajectory(s0, tau_func, n_steps, real_dynamics)
    actions = ref['actions']
    states_ref = ref['states_ref']

    print(f"  参考轨迹: {len(actions)} 步")
    print(f"  动作范围: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"  状态范围: phi=[{states_ref[:,0].min():.4f}, {states_ref[:,0].max():.4f}]")

    # 验证: 用参考轨迹中的动作重新运行真实系统
    s = s0.copy()
    states_replay = [s0.copy()]
    for i in range(len(actions)):
        s = real_dynamics(s, actions[i])
        states_replay.append(s.copy())
    states_replay = np.array(states_replay)

    # 比较
    max_diff = np.max(np.abs(states_ref[:len(states_replay)] - states_replay))
    print(f"  动作序列重放最大差异: {max_diff:.2e}")
    assert max_diff < 1e-10, f"Mode B 验证失败: 重放差异 {max_diff}"
    print("  ✓ Mode B 正确性验证通过")

    return ref


def main():
    parser = argparse.ArgumentParser(description='完整评估 v2')
    parser.add_argument('--quick', action='store_true', help='快速模式')
    parser.add_argument('--skip-gp', action='store_true', help='跳过GP')
    parser.add_argument('--skip-ensemble', action='store_true', help='跳过集成')
    parser.add_argument('--output', default=str(PROJECT_ROOT / 'continuation_stage' / 'results' / 'full_evaluation_v2.json'))
    args = parser.parse_args()

    if args.quick:
        seeds = [42, 43, 44, 45, 46]
        horizons = [10, 50, 100, 500, 1000]
        print(f"[快速模式] 5 seeds, 5 horizons")
    else:
        seeds = list(range(42, 72))
        horizons = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
        print(f"[完整模式] {len(seeds)} seeds, {len(horizons)} horizons")

    modes = ['A', 'B', 'C', 'D']
    state_std = np.array([0.28791831, 0.17332272, 1.15369545, 0.57764881])
    K_lqr = np.array([-103.92, -34.20, 38.04, 3.70])

    # ---- 生成训练数据 ----
    print("\n[1/4] 生成训练数据...")
    t0 = time.time()
    states, actions, deltas, s_std, a_std, d_std = mc.generate_training_data(30000)
    print(f"  完成 ({time.time()-t0:.1f}s)")

    # ---- Mode B 正确性验证 ----
    rng_verify = np.random.RandomState(42)
    phi_verify = rng_verify.uniform(-0.25, 0.25)
    s0_verify = np.array([phi_verify, 0.0, 0.0, 0.0])
    tau_func_verify = me.make_tau_func(0, 1000)
    ref_verify = verify_mode_b_correctness(s0_verify, tau_func_verify, 100, mc.real_step)

    # ---- 构建模型 ----
    print("\n[2/4] 构建模型...")
    models = {}

    models['real_dynamics'] = RealDynamics()
    print("  real_dynamics: OK")

    models['linearized_model'] = LinearizedModel()
    print("  linearized_model: OK")

    sindy = SINDy4D('meijaard_sindy_v35.npz')
    sindy.train(states, actions, deltas, s_std, a_std, d_std)
    models['sindy_4d'] = sindy
    print("  sindy_4d: OK")

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
    print("\n[3/4] 运行评估...")
    total_evals = len(seeds) * len(horizons) * len(model_names) * len(modes)
    print(f"  总评估次数: {total_evals}")

    # 保存参考轨迹
    reference_trajectories = {}
    results_data = {}
    completed = 0
    t0 = time.time()

    for seed in seeds:
        seed_str = str(seed)
        results_data[seed_str] = {}

        rng = np.random.RandomState(seed)
        phi_init = rng.uniform(-0.25, 0.25)
        s0 = np.array([phi_init, 0.0, 0.0, 0.0])
        tau_func = me.make_tau_func(seed - 42, 1000)

        # 生成参考轨迹 (一次)
        ref = generate_reference_trajectory(s0, tau_func, max(horizons), mc.real_step)
        reference_trajectories[seed_str] = {
            's0': s0.tolist(),
            'phi_init': float(phi_init),
            'actions': ref['actions'].tolist(),
            'states_ref': ref['states_ref'].tolist(),
            'n_valid_steps': ref['n_valid_steps'],
        }

        for horizon in horizons:
            h_str = str(horizon)
            results_data[seed_str][h_str] = {}

            for model_name in model_names:
                model = models[model_name]
                results_data[seed_str][h_str][model_name] = {}

                for mode in modes:
                    try:
                        if mode == 'A':
                            r = run_mode_a(model, s0, tau_func, horizon, mc.real_step)
                        elif mode == 'B':
                            r = run_mode_b_fixed(model, s0, ref, horizon, mc.real_step)
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
            'mode_b_fix': 'fixed_action_sequence_from_reference',
        },
        'reference_trajectories': reference_trajectories,
        'data': results_data,
        'total_time': total_time,
        'total_evaluations': completed,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, default=_json_default)
    print(f"\n  结果保存: {output_path}")

    # ---- 打印关键结果 ----
    print("\n" + "=" * 60)
    print("关键结果对比")
    print("=" * 60)

    for mode in modes:
        print(f"\n--- Mode {mode} ---")
        print(f"{'Model':<25}", end='')
        for h in horizons:
            print(f"{h:>10}", end='')
        print()
        for model_name in model_names:
            print(f"{model_name:<25}", end='')
            for h in horizons:
                mae_values = []
                for seed in seeds:
                    entry = results_data.get(str(seed), {}).get(str(h), {}).get(model_name, {}).get(mode, {})
                    mae = entry.get('metrics', {}).get('overall', {}).get('mae')
                    if mae is not None and not np.isnan(mae):
                        mae_values.append(mae)
                if mae_values:
                    mean_mae = np.mean(mae_values)
                    if mean_mae > 1e6:
                        print(f"{'diverge':>10}", end='')
                    else:
                        print(f"{mean_mae:>10.3f}", end='')
                else:
                    print(f"{'N/A':>10}", end='')
            print()


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
