"""不确定性校准实验 (精简版)"""

import sys, time, json, warnings
import numpy as np
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from continuation_stage.evaluate.eval_modes_v2 import (
    generate_reference_trajectory, run_mode_b_fixed, run_mode_a, run_mode_c, run_mode_d
)
from evaluate.eval_models import GPEEnsemble
import methods_common as mc
import methods_evaluate as me

def main():
    print('=== 不确定性校准实验 (精简版) ===')
    print(f'参数: 5000 样本, 50 epochs, 3 seeds, 50步')
    print()

    # 1. 生成训练数据
    print('[1/4] 生成训练数据...')
    t0 = time.time()
    states, actions, deltas, s_std, a_std, d_std = mc.generate_training_data(5000)
    print(f'  完成 ({time.time()-t0:.1f}s)')

    # 2. 训练集成模型
    print('[2/4] 训练集成模型...')
    ensemble_models = {}
    for n_models, dagger_rounds in [(5, 3), (7, 5)]:
        name = f'gp_ensemble_{n_models}_{dagger_rounds}'
        print(f'  训练 {name}...')
        t0 = time.time()
        ensemble = GPEEnsemble(
            n_models=n_models, dagger_rounds=dagger_rounds,
            residual_scale=0.3, n_epochs=50, use_scheduler=True
        )
        ensemble.train(states, actions, deltas, s_std, a_std, d_std)
        ensemble_models[name] = ensemble
        print(f'    完成 ({time.time()-t0:.1f}s)')

    # 3. 收集不确定性数据
    print('[3/4] 收集不确定性数据...')
    K_lqr = np.array([-103.92, -34.20, 38.04, 3.70])
    seeds = [42, 43, 44]
    horizon = 50
    modes = ['A', 'B', 'C', 'D']

    all_results = {}
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
                    reference = generate_reference_trajectory(s0, tau_func, horizon, mc.real_step)
                rng_d = np.random.RandomState(seed)

                if mode == 'A':
                    result = run_mode_a(model, s0, tau_func, horizon, mc.real_step)
                elif mode == 'B':
                    result = run_mode_b_fixed(model, s0, reference, horizon, mc.real_step)
                elif mode == 'C':
                    result = run_mode_c(model, s0, tau_func, horizon, mc.real_step)
                elif mode == 'D':
                    result = run_mode_d(model, s0, horizon, mc.real_step, K_lqr, rng_d)

                if 'uncertainties' in result and len(result['uncertainties']) > 0:
                    states_model = result['states_model']
                    states_real = result['states_real']
                    uncertainties = result['uncertainties']
                    n_valid = min(len(states_model) - 1, len(states_real) - 1, len(uncertainties))
                    errors = np.abs(states_model[1:n_valid+1] - states_real[1:n_valid+1])
                    uncertainties = uncertainties[:n_valid]
                    all_errors.append(errors)
                    all_uncs.append(uncertainties)

            if all_errors:
                errors_cat = np.concatenate(all_errors, axis=0)
                uncs_cat = np.concatenate(all_uncs, axis=0)
                stds = np.sqrt(np.abs(uncs_cat))
                n_steps, n_states = errors_cat.shape

                per_state = {}
                for dim in range(n_states):
                    e = errors_cat[:, dim]
                    s = stds[:, dim]
                    pearson_r, _ = stats.pearsonr(s, e)
                    spearman_r, _ = stats.spearmanr(s, e)
                    z = stats.norm.ppf(0.975)
                    coverage = float(np.mean((e >= -z * s) & (e <= z * s)))
                    per_state[f'state_{dim}'] = {
                        'pearson_r': float(pearson_r),
                        'spearman_r': float(spearman_r),
                        'coverage': coverage,
                        'calibration_error': abs(coverage - 0.95),
                        'mean_error': float(np.mean(e)),
                        'mean_uncertainty': float(np.mean(s)),
                    }

                e_flat = errors_cat.flatten()
                s_flat = stds.flatten()
                pearson_r, _ = stats.pearsonr(s_flat, e_flat)
                spearman_r, _ = stats.spearmanr(s_flat, e_flat)
                coverage = float(np.mean((e_flat >= -z * s_flat) & (e_flat <= z * s_flat)))

                all_results[model_name][mode] = {
                    'n_samples': int(n_steps),
                    'per_state': per_state,
                    'overall': {
                        'pearson_r': float(pearson_r),
                        'spearman_r': float(spearman_r),
                        'coverage': coverage,
                        'calibration_error': abs(coverage - 0.95),
                        'mean_error': float(np.mean(e_flat)),
                        'mean_uncertainty': float(np.mean(s_flat)),
                    }
                }
                o = all_results[model_name][mode]['overall']
                print(f'  {model_name} Mode {mode}: Pearson r={o["pearson_r"]:.3f}, coverage={o["coverage"]:.3f}, cal_error={o["calibration_error"]:.3f}')
            else:
                all_results[model_name][mode] = None
                print(f'  {model_name} Mode {mode}: 无不确定性数据')

    # 4. 保存结果
    print('[4/4] 保存结果...')
    output_path = Path(__file__).parent.parent / 'results' / 'uncertainty_calibration.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def json_default(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)): return None
        raise TypeError(f'Not serializable: {type(obj)}')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=json_default)
    print(f'结果保存: {output_path}')
    print('完成!')

if __name__ == '__main__':
    main()
