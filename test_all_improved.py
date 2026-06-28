"""所有基线 + 改进NN残差 精简对比。

每个基线只测：纯基线 vs 改进版(OOD+DAgger+0.3)
减少epoch加速运行。
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from methods_common import generate_training_data, real_step, K_lqr, dt
from methods_sindy import SINDyPoly, SINDyTrig
from methods_nn import NeuralODEMethod, NNE2E, ResidualNet, DEVICE
from methods_classic import ParamIDMethod
from methods_evaluate import make_tau_func


class OODDetector:
    def __init__(self, train_inputs, threshold=3.0):
        self.mean = np.mean(train_inputs, axis=0)
        self.std = np.std(train_inputs, axis=0) + 1e-8
        self.threshold = threshold
    def is_ood(self, x):
        return np.max(np.abs(x - self.mean) / self.std) > self.threshold


def train_nn(model, train_x, train_y, n_epochs=100):
    ds = TensorDataset(train_x, train_y)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            loss = crit(model(xb.to(DEVICE)), yb.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()


def dagger_augment(baseline, nn_model, state_std, action_std, delta_std, ood_det):
    extra_x, extra_y = [], []
    for seg_i in range(2):
        tau_func = make_tau_func(100 + seg_i, 500)
        s0 = np.array([np.random.uniform(-0.25, 0.25), 0.0, 0.0, 0.0])
        s_base, s_real = s0.copy(), s0.copy()
        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_norm = s_base / state_std
            a_norm = tau / action_std
            inp = np.concatenate([s_norm, [a_norm]])
            with torch.no_grad():
                delta_nn = nn_model(
                    torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE),
                    torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
                ).cpu().numpy()[0]
            if ood_det.is_ood(inp):
                delta_nn = np.zeros(4)
            s_prev = s_base.copy()
            s_base = baseline.predict(s_base, tau) + delta_nn * delta_std * 0.3
            if abs(s_real[0]) > math.pi / 3: break
            s_next_base = baseline.predict(s_prev, tau)
            real_res = (s_real - s_next_base) / delta_std
            extra_x.append(inp)
            extra_y.append(real_res - (s_next_base - s_prev) / delta_std)
    return extra_x, extra_y


def run_rollout(baseline, nn_model, state_std, action_std, delta_std, ood_det=None, residual_scale=0.0):
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    errors = {s: [] for s in eval_steps}
    for seg_i in range(5):
        tau_func = make_tau_func(seg_i, 500)
        s0 = np.array([np.random.uniform(-0.25, 0.25), 0.0, 0.0, 0.0])
        s_real, s_pred = s0.copy(), s0.copy()
        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            if nn_model is not None:
                s_norm = s_pred / state_std
                a_norm = tau / action_std
                inp = np.concatenate([s_norm, [a_norm]])
                if ood_det is not None and ood_det.is_ood(inp):
                    delta_nn = np.zeros(4)
                else:
                    with torch.no_grad():
                        delta_nn = nn_model(
                            torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE),
                            torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
                        ).cpu().numpy()[0]
                s_pred = baseline.predict(s_pred, tau) + delta_nn * delta_std * residual_scale
            else:
                s_pred = baseline.predict(s_pred, tau)
            if abs(s_real[0]) > math.pi / 3: break
            if step + 1 in eval_steps:
                errors[step + 1].append(abs(s_pred[0] - s_real[0]))
    return errors


def test_one(name, baseline, states, actions, deltas, state_std, action_std, delta_std):
    print(f"\n  [{name}] 计算残差...")
    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next = baseline.predict(states[i], actions[i])
        residuals[i] = (deltas[i] - (s_next - states[i])) / delta_std
    train_x_np = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
    train_x = torch.FloatTensor(train_x_np).to(DEVICE)
    train_y = torch.FloatTensor(residuals).to(DEVICE)

    # 纯基线
    print(f"  [{name}] 测试纯基线...")
    err_base = run_rollout(baseline, None, state_std, action_std, delta_std)

    # 改进版: OOD + DAgger + 0.3
    print(f"  [{name}] 训练改进版(OOD+DAgger+0.3)...")
    all_x = [train_x_np.copy()]
    all_y = [residuals.copy()]
    nn_model = None
    ood_det = None
    for round_i in range(3):
        tx = torch.FloatTensor(np.vstack(all_x)).to(DEVICE)
        ty = torch.FloatTensor(np.vstack(all_y)).to(DEVICE)
        nn_model = ResidualNet().to(DEVICE)
        train_nn(nn_model, tx, ty, n_epochs=100)
        ood_det = OODDetector(np.vstack(all_x), threshold=3.0)
        if round_i < 2:
            ex, ey = dagger_augment(baseline, nn_model, state_std, action_std, delta_std, ood_det)
            if ex:
                all_x.append(np.array(ex))
                all_y.append(np.array(ey))
    err_imp = run_rollout(baseline, nn_model, state_std, action_std, delta_std,
                          ood_det=ood_det, residual_scale=0.3)
    return err_base, err_imp


if __name__ == '__main__':
    print("=" * 60)
    print("所有基线 + 改进NN残差 对比")
    print("=" * 60)

    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]

    baselines = [
        ("SINDyPoly", SINDyPoly()),
        ("SINDyTrig", SINDyTrig()),
        ("NeuralODE", NeuralODEMethod()),
        ("NNE2E(纯NN)", NNE2E()),
        ("ParamID", ParamIDMethod()),
    ]

    all_results = {}
    for name, bl in baselines:
        print(f"\n{'='*60}")
        print(f"基线: {name}")
        t0 = time.time()
        bl.train(states, actions, deltas, state_std, action_std, delta_std)
        print(f"  训练耗时: {time.time()-t0:.1f}s")
        base_err, imp_err = test_one(name, bl, states, actions, deltas, state_std, action_std, delta_std)
        all_results[name] = (base_err, imp_err)

    # GP单独（较慢）
    print(f"\n{'='*60}")
    print("基线: GP (较慢)")
    from methods_classic import GPMethod
    gp = GPMethod()
    t0 = time.time()
    gp.train(states, actions, deltas, state_std, action_std, delta_std)
    print(f"  训练耗时: {time.time()-t0:.1f}s")
    base_gp, imp_gp = test_one("GP", gp, states, actions, deltas, state_std, action_std, delta_std)
    all_results["GP"] = (base_gp, imp_gp)

    # 输出
    print(f"\n{'='*75}")
    print("总结：500步MAE对比 (rad)")
    print(f"{'='*75}")
    print(f"  {'方法':<15} | {'纯基线':>10} | {'改进混合':>10} | {'倍率':>8} | {'结论':>8}")
    print(f"  {'-'*60}")
    for name in all_results:
        base, imp = all_results[name]
        b500 = np.mean(base[500]) if base[500] else float('nan')
        i500 = np.mean(imp[500]) if imp[500] else float('nan')
        ratio = i500 / b500 if b500 > 1e-10 else float('nan')
        verdict = "改善" if ratio < 0.8 else ("持平" if ratio < 1.2 else "恶化")
        print(f"  {name:<15} | {b500:10.4f} | {i500:10.4f} | {ratio:8.2f}x | {verdict:>8}")

    print("\n完成。")
