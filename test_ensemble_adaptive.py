"""集成+自适应缩放组合：GP集成 + 自适应残差缩放。

思路：
- GP集成提供多个预测
- 自适应缩放在OOD时减少残差贡献
- 结合两者优势

测试配置：
- 基线：GP
- 集成大小：5
- 自适应缩放：max_scale=0.5, k=3.0, threshold=3.0
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
# 自适应OOD检测器
# ============================================================
class AdaptiveOODDetector:
    def __init__(self, train_inputs, threshold=3.0, max_scale=0.5, k=3.0):
        self.mean = np.mean(train_inputs, axis=0)
        self.std = np.std(train_inputs, axis=0) + 1e-8
        self.threshold = threshold
        self.max_scale = max_scale
        self.k = k

    def compute_scale(self, x):
        z = np.abs(x - self.mean) / self.std
        mahal_dist = np.max(z)
        normalized_dist = (mahal_dist - 1.0) / (self.threshold - 1.0)
        clipped_input = max(-10.0, min(10.0, self.k * normalized_dist * 4))
        sigmoid = 1.0 / (1.0 + math.exp(clipped_input))
        return sigmoid * self.max_scale


# ============================================================
# 集成残差网络
# ============================================================
class EnsembleResidualNet:
    def __init__(self, n_models=5):
        self.n_models = n_models
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
            opt = optim.Adam(model.parameters(), lr=1e-3)
            crit = nn.MSELoss()

            model.train()
            for _ in range(n_epochs):
                for xb, yb in loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    loss = crit(model(xb), yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
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
# 训练集成+自适应方法
# ============================================================
def train_ensemble_adaptive(baseline_cls, baseline_kwargs=None, n_models=5,
                            n_epochs=100, dagger_rounds=3,
                            max_scale=0.5, k=3.0, threshold=3.0):
    if baseline_kwargs is None:
        baseline_kwargs = {}

    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)
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

        ensemble = EnsembleResidualNet(n_models=n_models)
        ensemble.train(train_inputs, train_residuals, n_epochs=n_epochs)

        ood_det = AdaptiveOODDetector(train_inputs, threshold=threshold, max_scale=max_scale, k=k)

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
                    inp = np.concatenate([s_norm, [a_norm]])
                    delta_nn = ensemble.predict_mean(s_norm, a_norm)
                    adaptive_scale = ood_det.compute_scale(inp)
                    s_base = baseline.predict(s_base, tau) + delta_nn * delta_std * adaptive_scale
                    if abs(s_real[0]) > math.pi / 3:
                        break
                    new_inputs.append(inp)
                    s_next_base = baseline.predict(s_base - delta_nn * delta_std * adaptive_scale, tau)
                    base_delta = s_next_base - (s_base - delta_nn * delta_std * adaptive_scale)
                    real_delta = s_real - (s_base - delta_nn * delta_std * adaptive_scale)
                    new_residuals.append((real_delta - base_delta) / delta_std)

            if new_inputs:
                all_train_inputs.append(np.array(new_inputs))
                all_train_residuals.append(np.array(new_residuals))
                print(f"    DAgger轮{round_i+1}: 收集{len(new_inputs)}个新样本")

    return baseline, ensemble, ood_det, (state_std, action_std, delta_std)


# ============================================================
# 集成+自适应预测
# ============================================================
def ensemble_adaptive_predict(baseline, ensemble, ood_detector, s, tau,
                              state_std, action_std, delta_std):
    s_next_base = baseline.predict(s, tau)
    s_norm = s / state_std
    a_norm = tau / action_std
    inp = np.concatenate([s_norm, [a_norm]])
    adaptive_scale = ood_detector.compute_scale(inp)
    if adaptive_scale < 0.01:
        return s_next_base
    delta_nn = ensemble.predict_mean(s_norm, a_norm)
    return s_next_base + delta_nn * delta_std * adaptive_scale


# ============================================================
# 测试
# ============================================================
def test_ensemble_adaptive(name, baseline_cls, baseline_kwargs=None,
                           n_models=5, max_scale=0.5, k=3.0, threshold=3.0):
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")

    baseline, ensemble, ood_det, (state_std, action_std, delta_std) = train_ensemble_adaptive(
        baseline_cls, baseline_kwargs,
        n_models=n_models, n_epochs=100, dagger_rounds=3,
        max_scale=max_scale, k=k, threshold=threshold
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
        s_combo = s0.copy()

        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_base = baseline.predict(s_base, tau)
            s_combo = ensemble_adaptive_predict(
                baseline, ensemble, ood_det, s_combo, tau,
                state_std, action_std, delta_std
            )
            if abs(s_real[0]) > math.pi / 3:
                break
            if step + 1 in eval_steps:
                all_errors[step + 1].append(abs(s_combo[0] - s_real[0]))
                baseline_errors[step + 1].append(abs(s_base[0] - s_real[0]))

    print(f"\n  {'步数':>6} | {'纯基线':>10} | {'集成+自适应':>12} | {'改进倍率':>10}")
    print(f"  {'-'*50}")
    for step in eval_steps:
        b_err = np.mean(baseline_errors[step]) if baseline_errors[step] else float('nan')
        h_err = np.mean(all_errors[step]) if all_errors[step] else float('nan')
        ratio = h_err / b_err if b_err > 0 and not math.isnan(h_err) else float('nan')
        print(f"  {step:6d} | {b_err:10.5f} | {h_err:12.5f} | {ratio:10.2f}x")

    return baseline_errors, all_errors


# ============================================================
# 参数对比测试
# ============================================================
def test_ensemble_adaptive_params(baseline_cls, baseline_kwargs=None):
    print(f"\n{'='*60}")
    print(f"测试: 集成+自适应参数对比")
    print(f"{'='*60}")

    configs = [
        {'n_models': 5, 'max_scale': 0.3, 'k': 3.0, 'threshold': 3.0},
        {'n_models': 5, 'max_scale': 0.5, 'k': 3.0, 'threshold': 3.0},
        {'n_models': 7, 'max_scale': 0.3, 'k': 3.0, 'threshold': 3.0},
        {'n_models': 7, 'max_scale': 0.5, 'k': 3.0, 'threshold': 3.0},
    ]

    results = {}

    for config in configs:
        name = f"集成({config['n_models']})+自适应({config['max_scale']})"
        print(f"\n--- 配置: {name} ---")
        baseline, ensemble, ood_det, (state_std, action_std, delta_std) = train_ensemble_adaptive(
            baseline_cls, baseline_kwargs,
            n_models=config['n_models'], n_epochs=100, dagger_rounds=3,
            max_scale=config['max_scale'], k=config['k'], threshold=config['threshold']
        )

        n_segments = 5
        errors_500 = []
        for seg_i in range(n_segments):
            tau_func = make_tau_func(seg_i, 500)
            phi_init = np.random.uniform(-0.25, 0.25)
            s0 = np.array([phi_init, 0.0, 0.0, 0.0])
            s_real = s0.copy()
            s_combo = s0.copy()

            for step in range(500):
                tau = tau_func(step, s_real)
                s_real = real_step(s_real, tau)
                s_combo = ensemble_adaptive_predict(
                    baseline, ensemble, ood_det, s_combo, tau,
                    state_std, action_std, delta_std
                )
                if abs(s_real[0]) > math.pi / 3:
                    break
                if step == 499:
                    errors_500.append(abs(s_combo[0] - s_real[0]))

        results[name] = np.mean(errors_500) if errors_500 else float('nan')
        print(f"  {name}: 500步MAE = {results[name]:.5f}")

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("集成+自适应缩放组合测试")
    print("=" * 60)

    # 测试GP + 集成 + 自适应
    test_ensemble_adaptive(
        "GP + 集成(5) + 自适应(0.5)", GPMethod,
        n_models=5, max_scale=0.5, k=3.0, threshold=3.0
    )

    # 测试不同参数组合
    test_ensemble_adaptive_params(GPMethod)

    print("\n完成。")
