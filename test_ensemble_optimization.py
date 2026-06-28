"""GP + 集成方法优化：在最佳结果(0.064)基础上尝试改进。

优化方向：
1. 更多集成模型：7, 9
2. 不同残差缩放：0.2, 0.25, 0.3
3. 更多DAgger轮次：5轮
4. 更多训练数据：50000样本
5. GP核函数优化
6. 学习率调度
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from methods_common import generate_training_data, real_step, K_lqr, dt
from methods_nn import NeuralODEMethod, ResidualNet, DEVICE
from methods_classic import GPMethod, ParamIDMethod
from methods_evaluate import make_tau_func


# ============================================================
# 集成残差网络（带学习率调度）
# ============================================================
class OptimizedEnsembleResidualNet:
    def __init__(self, n_models=5, lr=1e-3, use_scheduler=True):
        self.n_models = n_models
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.models = []

    def train(self, train_inputs, train_residuals, n_epochs=100):
        self.models = []
        for i in range(self.n_models):
            print(f"    训练模型 {i+1}/{self.n_models}")
            torch.manual_seed(i * 42)
            np.random.seed(i * 42)

            model = ResidualNet().to(DEVICE)
            train_x = torch.FloatTensor(train_inputs).to(DEVICE)
            train_y = torch.FloatTensor(train_residuals).to(DEVICE)
            ds = TensorDataset(train_x, train_y)
            loader = DataLoader(ds, batch_size=256, shuffle=True)
            opt = optim.Adam(model.parameters(), lr=self.lr)

            if self.use_scheduler:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
            else:
                scheduler = None

            crit = nn.MSELoss()

            model.train()
            for epoch in range(n_epochs):
                for xb, yb in loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    loss = crit(model(xb), yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                if scheduler:
                    scheduler.step()
            model.eval()
            self.models.append(model)

    def predict(self, s_norm, a_norm):
        s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
        a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)

        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(s_t, a_t).cpu().numpy()[0]
            predictions.append(pred)

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        return mean_pred, std_pred

    def predict_mean(self, s_norm, a_norm):
        mean_pred, _ = self.predict(s_norm, a_norm)
        return mean_pred


# ============================================================
# 训练优化集成方法
# ============================================================
def train_optimized_ensemble(baseline_cls, baseline_kwargs=None, n_models=5,
                            n_epochs=100, dagger_rounds=3, residual_scale=0.3,
                            n_train=30000, lr=1e-3, use_scheduler=True):
    if baseline_kwargs is None:
        baseline_kwargs = {}

    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(n_train)
    baseline = baseline_cls(**baseline_kwargs)
    baseline.train(states, actions, deltas, state_std, action_std, delta_std)

    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next_base = baseline.predict(states[i], actions[i])
        residuals[i] = (deltas[i] - (s_next_base - states[i])) / delta_std

    train_inputs_base = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
    all_train_inputs = [train_inputs_base.copy()]
    all_train_residuals = [residuals.copy()]

    for round_i in range(dagger_rounds):
        train_inputs = np.vstack(all_train_inputs)
        train_residuals = np.vstack(all_train_residuals)

        ensemble = OptimizedEnsembleResidualNet(
            n_models=n_models, lr=lr, use_scheduler=use_scheduler
        )
        ensemble.train(train_inputs, train_residuals, n_epochs=n_epochs)

        if round_i < dagger_rounds - 1:
            new_inputs = []
            new_residuals = []
            for seg_i in range(3):
                tau_func = make_tau_func(100 + seg_i, 500)
                phi_init = np.random.uniform(-0.25, 0.25)
                s0 = np.array([phi_init, 0.0, 0.0, 0.0])
                s_base = s0.copy()
                s_real = s0.copy()
                for step in range(500):
                    tau = tau_func(step, s_real)
                    s_real = real_step(s_real, tau)
                    s_norm = s_base / state_std
                    a_norm = tau / action_std
                    delta_nn = ensemble.predict_mean(s_norm, a_norm)
                    s_base = baseline.predict(s_base, tau) + delta_nn * delta_std * residual_scale
                    if abs(s_real[0]) > math.pi / 3:
                        break
                    new_inputs.append(np.concatenate([s_norm, [a_norm]]))
                    s_next_base = baseline.predict(s_base - delta_nn * delta_std * residual_scale, tau)
                    base_delta = s_next_base - (s_base - delta_nn * delta_std * residual_scale)
                    real_delta = s_real - (s_base - delta_nn * delta_std * residual_scale)
                    new_residuals.append((real_delta - base_delta) / delta_std)

            if new_inputs:
                all_train_inputs.append(np.array(new_inputs))
                all_train_residuals.append(np.array(new_residuals))
                print(f"    DAgger轮{round_i+1}: 收集{len(new_inputs)}个新样本")

    return baseline, ensemble, (state_std, action_std, delta_std)


# ============================================================
# 优化集成预测
# ============================================================
def optimized_ensemble_predict(baseline, ensemble, s, tau,
                              state_std, action_std, delta_std, residual_scale=0.3):
    s_next_base = baseline.predict(s, tau)
    s_norm = s / state_std
    a_norm = tau / action_std
    delta_nn = ensemble.predict_mean(s_norm, a_norm)
    return s_next_base + delta_nn * delta_std * residual_scale


# ============================================================
# 测试函数
# ============================================================
def test_optimized_ensemble(name, baseline_cls, baseline_kwargs=None,
                           n_models=5, residual_scale=0.3, n_train=30000,
                           lr=1e-3, use_scheduler=True, dagger_rounds=3):
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")

    baseline, ensemble, (state_std, action_std, delta_std) = train_optimized_ensemble(
        baseline_cls, baseline_kwargs,
        n_models=n_models, n_epochs=100, dagger_rounds=dagger_rounds,
        residual_scale=residual_scale, n_train=n_train, lr=lr, use_scheduler=use_scheduler
    )

    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    n_segments = 5
    all_errors = {step: [] for step in eval_steps}
    baseline_errors = {step: [] for step in eval_steps}

    for seg_i in range(n_segments):
        tau_func = make_tau_func(seg_i, 500)
        phi_init = np.random.uniform(-0.25, 0.25)
        s0 = np.array([phi_init, 0.0, 0.0, 0.0])
        s_real = s0.copy()
        s_base = s0.copy()
        s_ensemble = s0.copy()

        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_base = baseline.predict(s_base, tau)
            s_ensemble = optimized_ensemble_predict(
                baseline, ensemble, s_ensemble, tau,
                state_std, action_std, delta_std, residual_scale
            )
            if abs(s_real[0]) > math.pi / 3:
                break
            if step + 1 in eval_steps:
                all_errors[step + 1].append(abs(s_ensemble[0] - s_real[0]))
                baseline_errors[step + 1].append(abs(s_base[0] - s_real[0]))

    print(f"\n  {'步数':>6} | {'纯基线':>10} | {'优化集成':>10} | {'改进倍率':>10}")
    print(f"  {'-'*50}")
    for step in eval_steps:
        b_err = np.mean(baseline_errors[step]) if baseline_errors[step] else float('nan')
        h_err = np.mean(all_errors[step]) if all_errors[step] else float('nan')
        ratio = h_err / b_err if b_err > 0 and not math.isnan(h_err) else float('nan')
        print(f"  {step:6d} | {b_err:10.5f} | {h_err:10.5f} | {ratio:10.2f}x")

    return baseline_errors, all_errors


# ============================================================
# 参数对比测试
# ============================================================
def test_optimization_params(baseline_cls, baseline_kwargs=None):
    print(f"\n{'='*60}")
    print(f"测试: 优化参数对比")
    print(f"{'='*60}")

    configs = [
        # 基准：当前最佳配置
        {'name': '基准(5模型,0.3缩放,30k数据)', 'n_models': 5, 'residual_scale': 0.3, 'n_train': 30000, 'lr': 1e-3, 'use_scheduler': False, 'dagger_rounds': 3},
        # 测试1：更多模型
        {'name': '7模型', 'n_models': 7, 'residual_scale': 0.3, 'n_train': 30000, 'lr': 1e-3, 'use_scheduler': False, 'dagger_rounds': 3},
        {'name': '9模型', 'n_models': 9, 'residual_scale': 0.3, 'n_train': 30000, 'lr': 1e-3, 'use_scheduler': False, 'dagger_rounds': 3},
        # 测试2：不同残差缩放
        {'name': '缩放0.2', 'n_models': 5, 'residual_scale': 0.2, 'n_train': 30000, 'lr': 1e-3, 'use_scheduler': False, 'dagger_rounds': 3},
        {'name': '缩放0.25', 'n_models': 5, 'residual_scale': 0.25, 'n_train': 30000, 'lr': 1e-3, 'use_scheduler': False, 'dagger_rounds': 3},
        # 测试3：更多DAgger
        {'name': '5轮DAgger', 'n_models': 5, 'residual_scale': 0.3, 'n_train': 30000, 'lr': 1e-3, 'use_scheduler': False, 'dagger_rounds': 5},
        # 测试4：更多训练数据
        {'name': '50k数据', 'n_models': 5, 'residual_scale': 0.3, 'n_train': 50000, 'lr': 1e-3, 'use_scheduler': False, 'dagger_rounds': 3},
        # 测试5：学习率调度
        {'name': '学习率调度', 'n_models': 5, 'residual_scale': 0.3, 'n_train': 30000, 'lr': 1e-3, 'use_scheduler': True, 'dagger_rounds': 3},
        # 测试6：组合优化
        {'name': '7模型+缩放0.25', 'n_models': 7, 'residual_scale': 0.25, 'n_train': 30000, 'lr': 1e-3, 'use_scheduler': False, 'dagger_rounds': 3},
        {'name': '7模型+5轮DAgger', 'n_models': 7, 'residual_scale': 0.3, 'n_train': 30000, 'lr': 1e-3, 'use_scheduler': False, 'dagger_rounds': 5},
        {'name': '7模型+50k数据', 'n_models': 7, 'residual_scale': 0.3, 'n_train': 50000, 'lr': 1e-3, 'use_scheduler': False, 'dagger_rounds': 3},
        {'name': '7模型+学习率调度', 'n_models': 7, 'residual_scale': 0.3, 'n_train': 30000, 'lr': 1e-3, 'use_scheduler': True, 'dagger_rounds': 3},
    ]

    results = {}

    for config in configs:
        print(f"\n--- 配置: {config['name']} ---")
        baseline, ensemble, (state_std, action_std, delta_std) = train_optimized_ensemble(
            baseline_cls, baseline_kwargs,
            n_models=config['n_models'], n_epochs=100, dagger_rounds=config['dagger_rounds'],
            residual_scale=config['residual_scale'], n_train=config['n_train'],
            lr=config['lr'], use_scheduler=config['use_scheduler']
        )

        n_segments = 5
        errors_500 = []
        for seg_i in range(n_segments):
            tau_func = make_tau_func(seg_i, 500)
            phi_init = np.random.uniform(-0.25, 0.25)
            s0 = np.array([phi_init, 0.0, 0.0, 0.0])
            s_real = s0.copy()
            s_ensemble = s0.copy()

            for step in range(500):
                tau = tau_func(step, s_real)
                s_real = real_step(s_real, tau)
                s_ensemble = optimized_ensemble_predict(
                    baseline, ensemble, s_ensemble, tau,
                    state_std, action_std, delta_std, config['residual_scale']
                )
                if abs(s_real[0]) > math.pi / 3:
                    break
                if step == 499:
                    errors_500.append(abs(s_ensemble[0] - s_real[0]))

        results[config['name']] = np.mean(errors_500) if errors_500 else float('nan')
        print(f"  {config['name']}: 500步MAE = {results[config['name']]:.5f}")

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("GP + 集成方法优化测试")
    print("=" * 60)

    # 运行参数对比测试
    results = test_optimization_params(GPMethod)

    # 打印总结
    print("\n" + "=" * 60)
    print("优化结果总结")
    print("=" * 60)

    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for name, error in sorted_results:
        print(f"  {name:30s} | {error:.5f} rad")

    best_name, best_error = sorted_results[0]
    print(f"\n  最佳配置: {best_name}")
    print(f"  最佳MAE: {best_error:.5f} rad")

    print("\n完成。")
