"""SINDy + NN残差 改进版完整测试。

SINDy基线 + 逐步添加改进：残差缩放、OOD、DAgger
对比找出SINDy+NN效果恶化的原因。
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from methods_common import generate_training_data, real_step, K_lqr, dt
from methods_sindy import SINDyPoly, SINDyTrig
from methods_nn import ResidualNet, DEVICE
from methods_evaluate import make_tau_func


class OODDetector:
    def __init__(self, train_inputs, threshold=3.0):
        self.mean = np.mean(train_inputs, axis=0)
        self.std = np.std(train_inputs, axis=0) + 1e-8
        self.threshold = threshold

    def is_ood(self, x):
        return np.max(np.abs(x - self.mean) / self.std) > self.threshold


def train_nn_basic(nn_model, train_x, train_y, n_epochs=200):
    ds = TensorDataset(train_x, train_y)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    opt = optim.Adam(nn_model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    nn_model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = crit(nn_model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    nn_model.eval()


def dagger_augment(baseline, nn_model, state_std, action_std, delta_std, ood_det, residual_scale=0.3):
    extra_x, extra_y = [], []
    for seg_i in range(3):
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
            s_base = baseline.predict(s_base, tau) + delta_nn * delta_std * residual_scale
            if abs(s_real[0]) > math.pi / 3:
                break
            s_next_base = baseline.predict(s_prev, tau)
            real_res = (s_real - s_next_base) / delta_std
            extra_x.append(inp)
            extra_y.append(real_res - (s_next_base - s_prev) / delta_std)
    return extra_x, extra_y


def run_rollout(baseline, nn_model, state_std, action_std, delta_std,
                use_ood=False, ood_det=None, residual_scale=1.0):
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    errors = {s: [] for s in eval_steps}

    for seg_i in range(5):
        tau_func = make_tau_func(seg_i, 500)
        s0 = np.array([np.random.uniform(-0.25, 0.25), 0.0, 0.0, 0.0])
        s_real, s_pred = s0.copy(), s0.copy()

        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_norm = s_pred / state_std
            a_norm = tau / action_std
            inp = np.concatenate([s_norm, [a_norm]])

            if use_ood and ood_det is not None and ood_det.is_ood(inp):
                delta_nn = np.zeros(4)
            else:
                with torch.no_grad():
                    delta_nn = nn_model(
                        torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE),
                        torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
                    ).cpu().numpy()[0]

            s_pred = baseline.predict(s_pred, tau) + delta_nn * delta_std * residual_scale

            if abs(s_real[0]) > math.pi / 3:
                break
            if step + 1 in eval_steps:
                errors[step + 1].append(abs(s_pred[0] - s_real[0]))

    return errors


def diagnose_residual(baseline, states, actions, deltas, state_std, action_std, delta_std, name):
    """诊断基线的残差特性。"""
    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next = baseline.predict(states[i], actions[i])
        residuals[i] = deltas[i] - (s_next - states[i])

    residual_std = np.std(residuals, axis=0)
    ratio = residual_std / delta_std
    print(f"\n  {name} 残差诊断:")
    print(f"    delta_std: {delta_std}")
    print(f"    残差std:   {residual_std}")
    print(f"    残差/总比: {ratio}")
    print(f"    → 残差占总变化 {np.mean(ratio)*100:.1f}%")

    # 符号一致性
    nn_model = ResidualNet().to(DEVICE)
    res_norm = residuals / delta_std
    train_x = torch.FloatTensor(np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])).to(DEVICE)
    train_y = torch.FloatTensor(res_norm).to(DEVICE)
    train_nn_basic(nn_model, train_x, train_y, n_epochs=200)

    correct = 0
    total = 0
    for i in range(5000):
        s = states[i]; tau = actions[i]
        real_res = deltas[i] - (baseline.predict(s, tau) - s)
        s_norm = s / state_std; a_norm = tau / action_std
        with torch.no_grad():
            pred_res = nn_model(
                torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE),
                torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
            ).cpu().numpy()[0] * delta_std
        correct += np.sum(np.sign(real_res) == np.sign(pred_res))
        total += len(real_res)
    print(f"    符号一致率: {correct/total*100:.1f}%")

    return ratio


def test_baseline(name, baseline, states, actions, deltas, state_std, action_std, delta_std):
    """测试单个基线的所有改进组合。"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")

    # 诊断残差
    ratio = diagnose_residual(baseline, states, actions, deltas, state_std, action_std, delta_std, name)

    # 计算归一化残差
    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next = baseline.predict(states[i], actions[i])
        residuals[i] = (deltas[i] - (s_next - states[i])) / delta_std
    train_x_np = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
    train_x = torch.FloatTensor(train_x_np).to(DEVICE)
    train_y = torch.FloatTensor(residuals).to(DEVICE)

    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]

    # A: 纯基线
    err_a = {s: [] for s in eval_steps}
    for seg_i in range(5):
        tau_func = make_tau_func(seg_i, 500)
        s0 = np.array([np.random.uniform(-0.25, 0.25), 0.0, 0.0, 0.0])
        s_real, s_pred = s0.copy(), s0.copy()
        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_pred = baseline.predict(s_pred, tau)
            if abs(s_real[0]) > math.pi / 3: break
            if step + 1 in eval_steps:
                err_a[step + 1].append(abs(s_pred[0] - s_real[0]))

    # B: +NN残差(全量)
    print(f"  训练: +NN残差(全量)...")
    nn_b = ResidualNet().to(DEVICE)
    train_nn_basic(nn_b, train_x, train_y, n_epochs=200)
    err_b = run_rollout(baseline, nn_b, state_std, action_std, delta_std, residual_scale=1.0)

    # C: +NN残差(0.3)
    print(f"  训练: +NN残差(0.3)...")
    nn_c = ResidualNet().to(DEVICE)
    train_nn_basic(nn_c, train_x, train_y, n_epochs=200)
    err_c = run_rollout(baseline, nn_c, state_std, action_std, delta_std, residual_scale=0.3)

    # D: +OOD+NN(0.3)
    print(f"  训练: +OOD+NN(0.3)...")
    nn_d = ResidualNet().to(DEVICE)
    train_nn_basic(nn_d, train_x, train_y, n_epochs=200)
    ood_d = OODDetector(train_x_np, threshold=3.0)
    err_d = run_rollout(baseline, nn_d, state_std, action_std, delta_std,
                        use_ood=True, ood_det=ood_d, residual_scale=0.3)

    # E: +OOD+DAgger+NN(0.3)
    print(f"  训练: +OOD+DAgger+NN(0.3)...")
    all_x = [train_x_np.copy()]
    all_y = [residuals.copy()]
    nn_e = None
    ood_e = None
    for round_i in range(3):
        tx = torch.FloatTensor(np.vstack(all_x)).to(DEVICE)
        ty = torch.FloatTensor(np.vstack(all_y)).to(DEVICE)
        nn_e = ResidualNet().to(DEVICE)
        train_nn_basic(nn_e, tx, ty, n_epochs=200)
        ood_e = OODDetector(np.vstack(all_x), threshold=3.0)
        if round_i < 2:
            ex, ey = dagger_augment(baseline, nn_e, state_std, action_std, delta_std, ood_e)
            if ex:
                all_x.append(np.array(ex))
                all_y.append(np.array(ey))
    err_e = run_rollout(baseline, nn_e, state_std, action_std, delta_std,
                        use_ood=True, ood_det=ood_e, residual_scale=0.3)

    # 输出
    print(f"\n  {name} 对比 (MAE, rad)")
    print(f"  {'步数':>6} | {'纯基线':>8} | {'+NN':>10} | {'+NN(0.3)':>10} | {'+OOD+NN':>10} | {'+OOD+D+NN':>12}")
    print(f"  {'-'*65}")
    for step in eval_steps:
        a = np.mean(err_a[step]) if err_a[step] else float('nan')
        b = np.mean(err_b[step]) if err_b[step] else float('nan')
        c = np.mean(err_c[step]) if err_c[step] else float('nan')
        d = np.mean(err_d[step]) if err_d[step] else float('nan')
        e = np.mean(err_e[step]) if err_e[step] else float('nan')
        print(f"  {step:6d} | {a:8.5f} | {b:10.5f} | {c:10.5f} | {d:10.5f} | {e:12.5f}")

    return err_a, err_e


if __name__ == '__main__':
    print("=" * 60)
    print("SINDy/ParamID + NN残差 改进版完整测试")
    print("=" * 60)

    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)

    all_results = {}

    # SINDyPoly
    sindy_poly = SINDyPoly()
    sindy_poly.train(states, actions, deltas, state_std, action_std, delta_std)
    base_a, imp_a = test_baseline("SINDyPoly", sindy_poly, states, actions, deltas, state_std, action_std, delta_std)
    all_results["SINDyPoly"] = (base_a, imp_a)

    # SINDyTrig
    sindy_trig = SINDyTrig()
    sindy_trig.train(states, actions, deltas, state_std, action_std, delta_std)
    base_b, imp_b = test_baseline("SINDyTrig", sindy_trig, states, actions, deltas, state_std, action_std, delta_std)
    all_results["SINDyTrig"] = (base_b, imp_b)

    # ParamID
    from methods_classic import ParamIDMethod
    paramid = ParamIDMethod()
    paramid.train(states, actions, deltas, state_std, action_std, delta_std)
    base_c, imp_c = test_baseline("ParamID", paramid, states, actions, deltas, state_std, action_std, delta_std)
    all_results["ParamID"] = (base_c, imp_c)

    # 总结
    print(f"\n{'='*70}")
    print("总结：500步MAE对比 (rad)")
    print(f"{'='*70}")
    print(f"  {'方法':<15} | {'纯基线':>10} | {'+OOD+D+NN(0.3)':>15} | {'倍率':>8}")
    print(f"  {'-'*50}")
    for name in all_results:
        base, imp = all_results[name]
        b500 = np.mean(base[500]) if base[500] else float('nan')
        i500 = np.mean(imp[500]) if imp[500] else float('nan')
        ratio = i500 / b500 if b500 > 1e-10 else float('nan')
        print(f"  {name:<15} | {b500:10.5f} | {i500:15.5f} | {ratio:8.2f}x")

    print("\n完成。")
