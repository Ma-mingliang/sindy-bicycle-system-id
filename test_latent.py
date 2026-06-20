"""论文方案4：Latent-space dynamics。

用自编码器将4维状态映射到2维潜空间，在潜空间中学习动力学。
优势：降维可去噪，潜空间动力学更简单，可能更鲁棒。

对比：
- 纯NeuralODE基线
- 标准+OOD+DAgger（当前最佳）
- Latent-space dynamics（潜空间预测）
- Latent + OOD + DAgger
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


class Encoder(nn.Module):
    def __init__(self, state_dim=4, latent_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=2, state_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, state_dim)
        )

    def forward(self, z):
        return self.net(z)


class LatentDynamics(nn.Module):
    def __init__(self, latent_dim=2, action_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim)
        )

    def forward(self, z, a):
        za = torch.cat([z, a], dim=-1)
        return self.net(za)


class LatentModel:
    """潜空间动力学模型。"""

    def __init__(self, latent_dim=2, hidden=64):
        self.encoder = Encoder(latent_dim=latent_dim, hidden=hidden).to(DEVICE)
        self.decoder = Decoder(latent_dim=latent_dim, hidden=hidden).to(DEVICE)
        self.dynamics = LatentDynamics(latent_dim=latent_dim, hidden=hidden).to(DEVICE)
        self.latent_dim = latent_dim

    def train(self, states, actions, deltas, state_std, action_std, delta_std, n_epochs=300):
        """训练编码器、解码器和潜空间动力学。"""
        # 准备数据
        s_norm = states / state_std
        a_norm = actions.reshape(-1, 1) / action_std
        s_next = states + deltas
        s_next_norm = s_next / state_std

        s_t = torch.FloatTensor(s_norm).to(DEVICE)
        a_t = torch.FloatTensor(a_norm).to(DEVICE)
        s_next_t = torch.FloatTensor(s_next_norm).to(DEVICE)

        # 联合训练
        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.dynamics.parameters())
        opt = optim.Adam(params, lr=1e-3)

        ds = TensorDataset(s_t, a_t, s_next_t)
        loader = DataLoader(ds, batch_size=256, shuffle=True)

        for epoch in range(n_epochs):
            total_loss = 0
            for sb, ab, snb in loader:
                # 编码
                z = self.encoder(sb)
                z_next_true = self.encoder(snb)

                # 潜空间预测
                z_next_pred = z + self.dynamics(z, ab)

                # 解码
                s_recon = self.decoder(z)
                s_next_recon = self.decoder(z_next_pred)

                # 损失：重构 + 潜空间一致性 + 动力学
                loss_recon = nn.MSELoss()(s_recon, sb) + nn.MSELoss()(s_next_recon, snb)
                loss_latent = nn.MSELoss()(z_next_pred, z_next_true)
                loss = loss_recon + loss_latent

                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()

        self.encoder.eval()
        self.decoder.eval()
        self.dynamics.eval()

    def predict(self, s, tau, state_std, action_std, delta_std):
        """单步预测。"""
        s_norm = s / state_std
        a_norm = tau / action_std

        with torch.no_grad():
            s_t = torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)
            a_t = torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
            z = self.encoder(s_t)
            z_next = z + self.dynamics(z, a_t)
            s_next_norm = self.decoder(z_next).cpu().numpy()[0]

        return s_next_norm * state_std


class LatentOODDetector:
    """潜空间OOD检测：基于编码器输出的分布。"""

    def __init__(self, encoder, train_states, state_std, threshold=3.0):
        self.encoder = encoder
        self.threshold = threshold
        with torch.no_grad():
            z = encoder(torch.FloatTensor(train_states / state_std).to(DEVICE)).cpu().numpy()
        self.z_mean = np.mean(z, axis=0)
        self.z_std = np.std(z, axis=0) + 1e-8

    def is_ood(self, s, state_std):
        s_norm = s / state_std
        with torch.no_grad():
            z = self.encoder(torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
        return np.max(np.abs(z - self.z_mean) / self.z_std) > self.threshold


def run_rollout(baseline, latent_model, state_std, action_std, delta_std, ood_det=None, residual_scale=0.3):
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    errors = {s: [] for s in eval_steps}

    for seg_i in range(5):
        tau_func = make_tau_func(seg_i, 500)
        s0 = np.array([np.random.uniform(-0.25, 0.25), 0.0, 0.0, 0.0])
        s_real, s_pred = s0.copy(), s0.copy()

        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)

            if ood_det is not None and ood_det.is_ood(s_pred, state_std):
                s_pred = baseline.predict(s_pred, tau)
            else:
                s_pred = latent_model.predict(s_pred, tau, state_std, action_std, delta_std)

            if abs(s_real[0]) > math.pi / 3:
                break
            if step + 1 in eval_steps:
                errors[step + 1].append(abs(s_pred[0] - s_real[0]))

    return errors


def run_rollout_hybrid(baseline, latent_model, nn_model, state_std, action_std, delta_std, ood_det=None, residual_scale=0.3):
    """混合：潜空间模型 + NN残差。"""
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]
    errors = {s: [] for s in eval_steps}

    for seg_i in range(5):
        tau_func = make_tau_func(seg_i, 500)
        s0 = np.array([np.random.uniform(-0.25, 0.25), 0.0, 0.0, 0.0])
        s_real, s_pred = s0.copy(), s0.copy()

        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)

            # 潜空间预测
            s_latent = latent_model.predict(s_pred, tau, state_std, action_std, delta_std)

            # NN残差
            s_norm = s_pred / state_std
            a_norm = tau / action_std
            inp = np.concatenate([s_norm, [a_norm]])

            if ood_det is not None and ood_det.is_ood(inp):
                s_pred = s_latent
            else:
                with torch.no_grad():
                    delta_nn = nn_model(
                        torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE),
                        torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
                    ).cpu().numpy()[0]
                s_pred = s_latent + delta_nn * delta_std * residual_scale

            if abs(s_real[0]) > math.pi / 3:
                break
            if step + 1 in eval_steps:
                errors[step + 1].append(abs(s_pred[0] - s_real[0]))

    return errors


if __name__ == '__main__':
    print("=" * 60)
    print("论文方案4：Latent-space dynamics")
    print("=" * 60)

    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]

    # 基线
    print("\n训练NeuralODE基线...")
    baseline = NeuralODEMethod()
    baseline.train(states, actions, deltas, state_std, action_std, delta_std)

    # === 方法A：Latent-space dynamics ===
    print("\n训练A: Latent-space dynamics...")
    t0 = time.time()
    latent_a = LatentModel(latent_dim=2, hidden=64)
    latent_a.train(states, actions, deltas, state_std, action_std, delta_std, n_epochs=300)
    print(f"  耗时: {time.time()-t0:.1f}s")
    err_a = run_rollout(baseline, latent_a, state_std, action_std, delta_std)

    # === 方法B：Latent + OOD ===
    print("\n训练B: Latent + OOD...")
    ood_b = LatentOODDetector(latent_a.encoder, states, state_std, threshold=3.0)
    err_b = run_rollout(baseline, latent_a, state_std, action_std, delta_std, ood_det=ood_b)

    # === 方法C：Latent + NN残差 + OOD ===
    print("\n训练C: Latent + NN残差 + OOD...")
    # 计算残差（相对于潜空间模型）
    residuals_c = np.empty_like(deltas)
    for i in range(len(states)):
        s_next_latent = latent_a.predict(states[i], actions[i], state_std, action_std, delta_std)
        s_next_real = states[i] + deltas[i]
        residuals_c[i] = (s_next_real - s_next_latent) / delta_std

    train_x_np = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
    train_x = torch.FloatTensor(train_x_np).to(DEVICE)
    train_y = torch.FloatTensor(residuals_c).to(DEVICE)

    nn_c = ResidualNet().to(DEVICE)
    ds = TensorDataset(train_x, train_y)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    opt = optim.Adam(nn_c.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    nn_c.train()
    for _ in range(200):
        for xb, yb in loader:
            loss = crit(nn_c(xb.to(DEVICE)), yb.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    nn_c.eval()

    ood_c = OODDetector(train_x_np, threshold=3.0) if False else None
    # 使用简单的OOD检测
    from test_dr_dagger import OODDetector
    ood_c = OODDetector(train_x_np, threshold=3.0)
    err_c = run_rollout_hybrid(baseline, latent_a, nn_c, state_std, action_std, delta_std, ood_det=ood_c)

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
    print(f"\n{'='*70}")
    print("Latent-space dynamics 对比 (MAE, rad)")
    print(f"{'='*70}")
    print(f"  {'步数':>6} | {'纯ODE':>8} | {'Latent':>10} | {'Lat+OOD':>10} | {'Lat+NN+OOD':>12}")
    print(f"  {'-'*55}")
    for step in eval_steps:
        b = np.mean(base_err[step]) if base_err[step] else float('nan')
        a = np.mean(err_a[step]) if err_a[step] else float('nan')
        bb = np.mean(err_b[step]) if err_b[step] else float('nan')
        c = np.mean(err_c[step]) if err_c[step] else float('nan')
        print(f"  {step:6d} | {b:8.4f} | {a:10.5f} | {bb:10.5f} | {c:12.5f}")

    print("\n完成。")
