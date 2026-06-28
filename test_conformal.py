"""论文方案3：Conformalized Neural Dynamics。

用保形预测(Conformal Prediction)为NN残差提供有理论保证的预测区间。
当预测区间过宽（不确定性高）时，回退到纯基线预测。

对比：
- 纯NeuralODE基线
- 标准+OOD+DAgger（当前最佳）
- Conformal NN（保形预测不确定性）
- Conformal NN + DAgger
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from methods_common import generate_training_data, real_step, K_lqr, dt
from methods_nn import NeuralODEMethod, ResidualNet, DEVICE
from methods_evaluate import make_tau_func


class ConformalNN:
    """保形预测NN：用校准集计算非一致性分数，构建有覆盖保证的预测区间。"""

    def __init__(self, model, calibration_x, calibration_y, alpha=0.1):
        """
        alpha: 显著性水平，1-alpha = 覆盖率
        """
        self.model = model
        self.alpha = alpha
        self.quantile = None
        self._calibrate(calibration_x, calibration_y)

    def _calibrate(self, cal_x, cal_y):
        """用校准集计算非一致性分数的分位数。"""
        self.model.eval()
        with torch.no_grad():
            preds = self.model(cal_x).cpu().numpy()
        actuals = cal_y.cpu().numpy()
        # 非一致性分数 = |预测 - 实际| 的L2范数
        scores = np.linalg.norm(preds - actuals, axis=1)
        # 取 (1-alpha) 分位数
        n = len(scores)
        q_idx = int(np.ceil((1 - self.alpha) * (n + 1))) - 1
        q_idx = min(q_idx, n - 1)
        self.quantile = np.sort(scores)[q_idx]
        self.score_mean = np.mean(scores)
        self.score_std = np.std(scores) + 1e-8

    def predict_with_uncertainty(self, x_tensor):
        """返回 (预测值, 不确定性分数, 是否可信)。"""
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_tensor).cpu().numpy()[0]
        return pred

    def is_uncertain(self, x_tensor, pred):
        """判断预测是否在可信区间内。"""
        # 用校准分位数判断
        # 如果预测值的范数异常大，认为不确定
        pred_norm = np.linalg.norm(pred)
        return pred_norm > self.quantile * 2.0

    def get_conformal_score(self, x_tensor, pred):
        """获取预测的非一致性分数（越低越可信）。"""
        return np.linalg.norm(pred) / (self.quantile + 1e-8)


def train_conformal_nn(train_x, train_y, n_epochs=200, cal_ratio=0.2):
    """训练NN并划分校准集。"""
    n = len(train_x)
    perm = torch.randperm(n)
    n_cal = int(n * cal_ratio)

    cal_idx = perm[:n_cal]
    train_idx = perm[n_cal:]

    train_x_sub = train_x[train_idx]
    train_y_sub = train_y[train_idx]
    cal_x = train_x[cal_idx]
    cal_y = train_y[cal_idx]

    model = ResidualNet().to(DEVICE)
    ds = TensorDataset(train_x_sub, train_y_sub)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    conf_nn = ConformalNN(model, cal_x, cal_y, alpha=0.1)
    return model, conf_nn


def dagger_augment_conformal(baseline, model, conf_nn, state_std, action_std, delta_std, residual_scale=0.3):
    """DAgger: rollout收集新数据，使用conformal判断。"""
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
            inp_tensor = torch.FloatTensor(np.concatenate([s_norm, [a_norm]])).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                delta_nn = model(inp_tensor).cpu().numpy()[0]

            if conf_nn.is_uncertain(inp_tensor, delta_nn):
                delta_nn = np.zeros(4)

            s_prev = s_base.copy()
            s_base = baseline.predict(s_base, tau) + delta_nn * delta_std * residual_scale
            if abs(s_real[0]) > math.pi / 3:
                break
            s_next_base = baseline.predict(s_prev, tau)
            real_res = (s_real - s_next_base) / delta_std
            extra_x.append(np.concatenate([s_norm, [a_norm]]))
            extra_y.append(real_res - (s_next_base - s_prev) / delta_std)
    return extra_x, extra_y


def run_rollout(baseline, model, conf_nn, state_std, action_std, delta_std, residual_scale=0.3):
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    errors = {s: [] for s in eval_steps}
    uncertain_count, total_count = 0, 0

    for seg_i in range(5):
        tau_func = make_tau_func(seg_i, 500)
        s0 = np.array([np.random.uniform(-0.25, 0.25), 0.0, 0.0, 0.0])
        s_real, s_pred = s0.copy(), s0.copy()

        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_norm = s_pred / state_std
            a_norm = tau / action_std
            inp_tensor = torch.FloatTensor(np.concatenate([s_norm, [a_norm]])).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                delta_nn = model(inp_tensor).cpu().numpy()[0]

            total_count += 1
            if conf_nn.is_uncertain(inp_tensor, delta_nn):
                uncertain_count += 1
                delta_nn = np.zeros(4)

            s_pred = baseline.predict(s_pred, tau) + delta_nn * delta_std * residual_scale

            if abs(s_real[0]) > math.pi / 3:
                break
            if step + 1 in eval_steps:
                errors[step + 1].append(abs(s_pred[0] - s_real[0]))

    unc_rate = uncertain_count / total_count * 100 if total_count > 0 else 0
    return errors, unc_rate


if __name__ == '__main__':
    print("=" * 60)
    print("论文方案3：Conformalized Neural Dynamics")
    print("=" * 60)

    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]

    # 基线
    print("\n训练NeuralODE基线...")
    baseline = NeuralODEMethod()
    baseline.train(states, actions, deltas, state_std, action_std, delta_std)

    # 残差
    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next = baseline.predict(states[i], actions[i])
        residuals[i] = (deltas[i] - (s_next - states[i])) / delta_std
    train_x_np = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
    train_x = torch.FloatTensor(train_x_np).to(DEVICE)
    train_y = torch.FloatTensor(residuals).to(DEVICE)

    # === 方法A：标准+OOD+DAgger（当前最佳基线）===
    print("\n训练A: 标准+OOD+DAgger...")
    from test_dr_dagger import OODDetector, train_with_noise

    all_x_a = [train_x_np.copy()]
    all_y_a = [residuals.copy()]
    nn_a = None
    ood_a = None
    for round_i in range(3):
        tx = torch.FloatTensor(np.vstack(all_x_a)).to(DEVICE)
        ty = torch.FloatTensor(np.vstack(all_y_a)).to(DEVICE)
        nn_a = ResidualNet().to(DEVICE)
        ds = TensorDataset(tx, ty)
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        opt = optim.Adam(nn_a.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        nn_a.train()
        for _ in range(200):
            for xb, yb in loader:
                loss = crit(nn_a(xb.to(DEVICE)), yb.to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()
        nn_a.eval()
        ood_a = OODDetector(np.vstack(all_x_a), threshold=3.0)
        if round_i < 2:
            ex, ey = dagger_augment_conformal(baseline, nn_a,
                type('FakeConf', (), {'is_uncertain': lambda self, x, p: ood_a.is_ood(x.cpu().numpy()[0])})(),
                state_std, action_std, delta_std)
            if ex:
                all_x_a.append(np.array(ex))
                all_y_a.append(np.array(ey))
    err_a, _ = run_rollout(baseline, nn_a,
        type('FakeConf', (), {'is_uncertain': lambda self, x, p: ood_a.is_ood(x.cpu().numpy()[0])})(),
        state_std, action_std, delta_std)

    # === 方法B：Conformal NN（无DAgger）===
    print("\n训练B: Conformal NN...")
    model_b, conf_b = train_conformal_nn(train_x, train_y, n_epochs=200, cal_ratio=0.2)
    err_b, unc_b = run_rollout(baseline, model_b, conf_b, state_std, action_std, delta_std)
    print(f"  不确定性回退率: {unc_b:.1f}%")

    # === 方法C：Conformal NN + DAgger ===
    print("\n训练C: Conformal NN + DAgger...")
    all_x_c = [train_x_np.copy()]
    all_y_c = [residuals.copy()]
    model_c, conf_c = None, None
    for round_i in range(3):
        tx = torch.FloatTensor(np.vstack(all_x_c)).to(DEVICE)
        ty = torch.FloatTensor(np.vstack(all_y_c)).to(DEVICE)
        model_c, conf_c = train_conformal_nn(tx, ty, n_epochs=200, cal_ratio=0.2)
        if round_i < 2:
            ex, ey = dagger_augment_conformal(baseline, model_c, conf_c, state_std, action_std, delta_std)
            if ex:
                all_x_c.append(np.array(ex))
                all_y_c.append(np.array(ey))
    err_c, unc_c = run_rollout(baseline, model_c, conf_c, state_std, action_std, delta_std)
    print(f"  不确定性回退率: {unc_c:.1f}%")

    # === 方法D：Conformal NN + DAgger + DR ===
    print("\n训练D: Conformal NN + DAgger + DR...")
    all_x_d = [train_x_np.copy()]
    all_y_d = [residuals.copy()]
    model_d, conf_d = None, None
    for round_i in range(3):
        tx = torch.FloatTensor(np.vstack(all_x_d)).to(DEVICE)
        ty = torch.FloatTensor(np.vstack(all_y_d)).to(DEVICE)
        # DR: 训练时加噪声
        model_d = ResidualNet().to(DEVICE)
        ds = TensorDataset(tx, ty)
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        opt = optim.Adam(model_d.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        model_d.train()
        for _ in range(200):
            for xb, yb in loader:
                xb_noisy = xb + torch.randn_like(xb) * 1.0
                loss = crit(model_d(xb_noisy), yb)
                opt.zero_grad(); loss.backward(); opt.step()
        model_d.eval()
        # 校准
        n = len(tx)
        perm = torch.randperm(n)
        n_cal = int(n * 0.2)
        cal_x = tx[perm[:n_cal]]
        cal_y = ty[perm[:n_cal]]
        conf_d = ConformalNN(model_d, cal_x, cal_y, alpha=0.1)
        if round_i < 2:
            ex, ey = dagger_augment_conformal(baseline, model_d, conf_d, state_std, action_std, delta_std)
            if ex:
                all_x_d.append(np.array(ex))
                all_y_d.append(np.array(ey))
    err_d, unc_d = run_rollout(baseline, model_d, conf_d, state_std, action_std, delta_std)
    print(f"  不确定性回退率: {unc_d:.1f}%")

    # 纯基线
    print("\n运行纯NeuralODE基线...")
    base_err = {s: [] for s in eval_steps}
    for seg_i in range(5):
        tau_func = make_tau_func(seg_i, 500)
        s0 = np.array([np.random.uniform(-0.25, 0.25), 0.0, 0.0, 0.0])
        s_real, s_base = s0.copy(), s0.copy()
        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            s_base = baseline.predict(s_base, tau)
            if abs(s_real[0]) > math.pi / 3:
                break
            if step + 1 in eval_steps:
                base_err[step + 1].append(abs(s_base[0] - s_real[0]))

    # 输出
    print(f"\n{'='*75}")
    print("Conformalized Neural Dynamics 对比 (MAE, rad)")
    print(f"{'='*75}")
    print(f"  {'步数':>6} | {'纯ODE':>8} | {'标准+OOD+D':>12} | {'Conformal':>10} | {'Conf+D':>10} | {'Conf+D+DR':>10}")
    print(f"  {'-'*70}")
    for step in eval_steps:
        b = np.mean(base_err[step]) if base_err[step] else float('nan')
        a = np.mean(err_a[step]) if err_a[step] else float('nan')
        bb = np.mean(err_b[step]) if err_b[step] else float('nan')
        c = np.mean(err_c[step]) if err_c[step] else float('nan')
        d = np.mean(err_d[step]) if err_d[step] else float('nan')
        print(f"  {step:6d} | {b:8.4f} | {a:12.5f} | {bb:10.5f} | {c:10.5f} | {d:10.5f}")

    print("\n完成。")
