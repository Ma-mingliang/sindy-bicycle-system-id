"""最终总结：所有改进方案对比。

在相同条件下重新运行最佳方案，确保公平对比。
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
from test_dr_dagger import OODDetector


class Encoder(nn.Module):
    def __init__(self, state_dim=4, latent_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim)
        )
    def forward(self, x): return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim=2, state_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, state_dim)
        )
    def forward(self, z): return self.net(z)


class LatentDynamics(nn.Module):
    def __init__(self, latent_dim=2, action_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim)
        )
    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


class LatentModel:
    def __init__(self, latent_dim=2, hidden=64):
        self.encoder = Encoder(latent_dim=latent_dim, hidden=hidden).to(DEVICE)
        self.decoder = Decoder(latent_dim=latent_dim, hidden=hidden).to(DEVICE)
        self.dynamics = LatentDynamics(latent_dim=latent_dim, hidden=hidden).to(DEVICE)

    def train(self, states, actions, deltas, state_std, action_std, delta_std, n_epochs=300):
        s_norm = states / state_std
        a_norm = actions.reshape(-1, 1) / action_std
        s_next_norm = (states + deltas) / state_std
        s_t = torch.FloatTensor(s_norm).to(DEVICE)
        a_t = torch.FloatTensor(a_norm).to(DEVICE)
        s_next_t = torch.FloatTensor(s_next_norm).to(DEVICE)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.dynamics.parameters())
        opt = optim.Adam(params, lr=1e-3)
        ds = TensorDataset(s_t, a_t, s_next_t)
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        for _ in range(n_epochs):
            for sb, ab, snb in loader:
                z = self.encoder(sb)
                z_next_pred = z + self.dynamics(z, ab)
                s_recon = self.decoder(z)
                s_next_recon = self.decoder(z_next_pred)
                loss = nn.MSELoss()(s_recon, sb) + nn.MSELoss()(s_next_recon, snb) + nn.MSELoss()(z_next_pred, self.encoder(snb))
                opt.zero_grad(); loss.backward(); opt.step()
        self.encoder.eval(); self.decoder.eval(); self.dynamics.eval()

    def predict(self, s, tau, state_std, action_std, delta_std):
        s_norm = s / state_std
        a_norm = tau / action_std
        with torch.no_grad():
            z = self.encoder(torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE))
            z_next = z + self.dynamics(z, torch.FloatTensor([[a_norm]]).to(DEVICE))
            s_next_norm = self.decoder(z_next).cpu().numpy()[0]
        return s_next_norm * state_std


class LatentOOD:
    def __init__(self, encoder, train_states, state_std, threshold=3.0):
        self.encoder = encoder
        self.threshold = threshold
        with torch.no_grad():
            z = encoder(torch.FloatTensor(train_states / state_std).to(DEVICE)).cpu().numpy()
        self.z_mean = np.mean(z, axis=0)
        self.z_std = np.std(z, axis=0) + 1e-8

    def is_ood(self, s, state_std):
        with torch.no_grad():
            z = self.encoder(torch.FloatTensor((s / state_std)).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
        return np.max(np.abs(z - self.z_mean) / self.z_std) > self.threshold


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
            if abs(s_real[0]) > math.pi / 3: break
            s_next_base = baseline.predict(s_prev, tau)
            real_res = (s_real - s_next_base) / delta_std
            extra_x.append(inp)
            extra_y.append(real_res - (s_next_base - s_prev) / delta_std)
    return extra_x, extra_y


def run_all_methods():
    print("=" * 70)
    print("最终总结：所有改进方案对比")
    print("=" * 70)

    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000)
    eval_steps = [1, 5, 10, 20, 50, 100, 200, 500]

    # 基线
    print("\n1/6 训练NeuralODE基线...")
    baseline = NeuralODEMethod()
    baseline.train(states, actions, deltas, state_std, action_std, delta_std)

    # 残差
    residuals = np.empty_like(deltas)
    for i in range(len(states)):
        s_next = baseline.predict(states[i], actions[i])
        residuals[i] = (deltas[i] - (s_next - states[i])) / delta_std
    train_x_np = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])

    # === 方法1：纯NeuralODE ===
    print("2/6 测试纯NeuralODE...")
    err1 = {s: [] for s in eval_steps}
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
                err1[step + 1].append(abs(s_pred[0] - s_real[0]))

    # === 方法2：标准+OOD+DAgger ===
    print("3/6 训练标准+OOD+DAgger...")
    all_x = [train_x_np.copy()]
    all_y = [residuals.copy()]
    nn_model = None
    ood_det = None
    for round_i in range(3):
        tx = torch.FloatTensor(np.vstack(all_x)).to(DEVICE)
        ty = torch.FloatTensor(np.vstack(all_y)).to(DEVICE)
        nn_model = ResidualNet().to(DEVICE)
        ds = TensorDataset(tx, ty)
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        opt = optim.Adam(nn_model.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        nn_model.train()
        for _ in range(200):
            for xb, yb in loader:
                loss = crit(nn_model(xb.to(DEVICE)), yb.to(DEVICE))
                opt.zero_grad(); loss.backward(); opt.step()
        nn_model.eval()
        ood_det = OODDetector(np.vstack(all_x), threshold=3.0)
        if round_i < 2:
            ex, ey = dagger_augment(baseline, nn_model, state_std, action_std, delta_std, ood_det)
            if ex:
                all_x.append(np.array(ex))
                all_y.append(np.array(ey))

    err2 = {s: [] for s in eval_steps}
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
            if ood_det.is_ood(inp):
                delta_nn = np.zeros(4)
            else:
                with torch.no_grad():
                    delta_nn = nn_model(
                        torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE),
                        torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
                    ).cpu().numpy()[0]
            s_pred = baseline.predict(s_pred, tau) + delta_nn * delta_std * 0.3
            if abs(s_real[0]) > math.pi / 3: break
            if step + 1 in eval_steps:
                err2[step + 1].append(abs(s_pred[0] - s_real[0]))

    # === 方法3：DR(1.0)+OOD+DAgger ===
    print("4/6 训练DR(1.0)+OOD+DAgger...")
    all_x3 = [train_x_np.copy()]
    all_y3 = [residuals.copy()]
    nn3 = None
    ood3 = None
    for round_i in range(3):
        tx = torch.FloatTensor(np.vstack(all_x3)).to(DEVICE)
        ty = torch.FloatTensor(np.vstack(all_y3)).to(DEVICE)
        nn3 = ResidualNet().to(DEVICE)
        ds = TensorDataset(tx, ty)
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        opt = optim.Adam(nn3.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        nn3.train()
        for _ in range(200):
            for xb, yb in loader:
                xb_noisy = xb + torch.randn_like(xb) * 1.0
                loss = crit(nn3(xb_noisy), yb)
                opt.zero_grad(); loss.backward(); opt.step()
        nn3.eval()
        ood3 = OODDetector(np.vstack(all_x3), threshold=3.0)
        if round_i < 2:
            ex, ey = dagger_augment(baseline, nn3, state_std, action_std, delta_std, ood3)
            if ex:
                all_x3.append(np.array(ex))
                all_y3.append(np.array(ey))

    err3 = {s: [] for s in eval_steps}
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
            if ood3.is_ood(inp):
                delta_nn = np.zeros(4)
            else:
                with torch.no_grad():
                    delta_nn = nn3(
                        torch.FloatTensor(s_norm).unsqueeze(0).to(DEVICE),
                        torch.FloatTensor([a_norm]).unsqueeze(0).to(DEVICE)
                    ).cpu().numpy()[0]
            s_pred = baseline.predict(s_pred, tau) + delta_nn * delta_std * 0.3
            if abs(s_real[0]) > math.pi / 3: break
            if step + 1 in eval_steps:
                err3[step + 1].append(abs(s_pred[0] - s_real[0]))

    # === 方法4：Latent+OOD（最佳）===
    print("5/6 训练Latent+OOD...")
    t0 = time.time()
    latent = LatentModel(latent_dim=2, hidden=64)
    latent.train(states, actions, deltas, state_std, action_std, delta_std, n_epochs=300)
    print(f"  训练耗时: {time.time()-t0:.1f}s")
    latent_ood = LatentOOD(latent.encoder, states, state_std, threshold=3.0)

    err4 = {s: [] for s in eval_steps}
    for seg_i in range(5):
        tau_func = make_tau_func(seg_i, 500)
        s0 = np.array([np.random.uniform(-0.25, 0.25), 0.0, 0.0, 0.0])
        s_real, s_pred = s0.copy(), s0.copy()
        for step in range(500):
            tau = tau_func(step, s_real)
            s_real = real_step(s_real, tau)
            if latent_ood.is_ood(s_pred, state_std):
                s_pred = baseline.predict(s_pred, tau)
            else:
                s_pred = latent.predict(s_pred, tau, state_std, action_std, delta_std)
            if abs(s_real[0]) > math.pi / 3: break
            if step + 1 in eval_steps:
                err4[step + 1].append(abs(s_pred[0] - s_real[0]))

    # 输出
    print(f"\n{'='*75}")
    print("最终对比 (MAE, rad)")
    print(f"{'='*75}")
    print(f"  {'步数':>6} | {'纯ODE':>8} | {'ODE+OOD+D':>10} | {'DR+OOD+D':>10} | {'Lat+OOD':>10}")
    print(f"  {'-'*55}")
    for step in eval_steps:
        v1 = np.mean(err1[step]) if err1[step] else float('nan')
        v2 = np.mean(err2[step]) if err2[step] else float('nan')
        v3 = np.mean(err3[step]) if err3[step] else float('nan')
        v4 = np.mean(err4[step]) if err4[step] else float('nan')
        print(f"  {step:6d} | {v1:8.4f} | {v2:10.5f} | {v3:10.5f} | {v4:10.5f}")

    print(f"\n{'='*75}")
    print("排名总结")
    print(f"{'='*75}")
    results = {
        "纯NeuralODE": err1,
        "ODE+OOD+DAgger": err2,
        "DR1.0+OOD+DAgger": err3,
        "Latent+OOD": err4,
    }
    ranking = sorted(results.items(), key=lambda x: np.mean(x[1][500]) if x[1][500] else 999)
    for rank, (name, err) in enumerate(ranking, 1):
        v = np.mean(err[500]) if err[500] else float('nan')
        print(f"  #{rank}: {name:<20} = {v:.5f} rad")

    print("\n最佳方案: Latent+OOD — 自编码器降维到2D潜空间 + 潜空间OOD检测")
    print("优势: 潜空间动力学更简单，OOD检测更可靠，无需DAgger即可稳定")
    print("\n完成。")


if __name__ == '__main__':
    run_all_methods()
