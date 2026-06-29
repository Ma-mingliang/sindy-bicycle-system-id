"""诊断方案 5 (PerState) 的根源性问题。

分析每个状态的误差来源、GP vs NODE 的误差对比、
以及误差随时间的演化。
"""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')
sys.stdout.reconfigure(line_buffering=True)

import warnings
import numpy as np
import torch
import torch.nn as nn
from canonical_7d.data_loader import load_7d_data, get_test_segments
from canonical_7d.config import DataConfig
from canonical_node.config_v9 import STATE_NAMES_7D, STATE_DIM, ACTION_DIM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
DT = 1.0 / 30.0


# ============================================================
# 加载 V9 Neural ODE
# ============================================================
class ODEFunc(nn.Module):
    def __init__(self, hidden=64, depth=3):
        super().__init__()
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


def load_v9_node(state_std, action_std, delta_std):
    model = ODEFunc(hidden=64, depth=3).to(DEVICE)
    ckpt = torch.load(
        'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/checkpoints/BEST_NEURAL_ODE_V9.pt',
        map_location=DEVICE, weights_only=False,
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


def node_predict(model, state_std, action_std, delta_std, s, tau):
    s_norm = torch.FloatTensor(s / state_std).unsqueeze(0).to(DEVICE)
    a_norm = torch.FloatTensor([tau / action_std]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        dsdt_norm = model(s_norm, a_norm).cpu().numpy()[0]
    return s + dsdt_norm * delta_std * DT


def node_predict_delta(model, state_std, action_std, delta_std, s, tau):
    s_norm = torch.FloatTensor(s / state_std).unsqueeze(0).to(DEVICE)
    a_norm = torch.FloatTensor([tau / action_std]).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        dsdt_norm = model(s_norm, a_norm).cpu().numpy()[0]
    return dsdt_norm * delta_std * DT


# ============================================================
# GP
# ============================================================
def train_gp(X_train, y_train, max_samples=2000):
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    n = len(X_train)
    if n > max_samples:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(n, max_samples, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-8
    X_scaled = (X_train - x_mean) / x_std

    kernel = Matern(nu=2.5, length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-3)
    gp.fit(X_scaled, y_train)
    return gp, x_mean, x_std


def gp_predict(gp, x_mean, x_std, s_norm, a_norm):
    x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
    x_scaled = (x - x_mean) / x_std
    return gp.predict(x_scaled)[0]


# ============================================================
# 主诊断
# ============================================================
def main():
    print("=" * 80)
    print("根源性问题诊断：PerState 混合模型")
    print("=" * 80)

    # 加载数据
    cfg = DataConfig(
        data_path='D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz',
        seed=SEED,
    )
    data = load_7d_data(cfg)
    train_s = data['train_states']
    train_a = data['train_actions']
    train_d = data['train_deltas']
    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    # 加载 V9 NODE
    node = load_v9_node(state_std, action_std, delta_std)

    # 获取测试段
    segments = get_test_segments(data, n_segments=5, segment_length=1100, seed=SEED)

    # ============================================================
    # 诊断 1: 哪些状态的误差贡献最大？
    # ============================================================
    print("\n" + "=" * 80)
    print("诊断 1: 各状态在不同 horizon 的 NMAE 贡献")
    print("=" * 80)

    # 用纯 NODE 做 rollout，记录每步每状态的误差
    seg = segments[0]
    s0 = seg['states'][0].copy()
    actions_seg = seg['actions']
    real_states = seg['states']

    # NODE rollout
    s_cur = s0.copy()
    node_traj = [s0.copy()]
    for step in range(min(1000, len(actions_seg))):
        s_next = node_predict(node, state_std, action_std, delta_std, s_cur, actions_seg[step])
        node_traj.append(s_next.copy())
        s_cur = s_next
    node_traj = np.array(node_traj)

    # GP rollout (for e_y, e_psi, theta_dot, delta_dot)
    print("\nTraining GP for e_y, e_psi, theta_dot, delta_dot...")
    X_train = np.column_stack([train_s / state_std, train_a.reshape(-1, 1) / action_std])
    gp_dims = [0, 1, 4, 6]  # e_y, e_psi, theta_dot, delta_dot
    gps = {}
    for dim in gp_dims:
        Y_train = train_d[:, dim] / delta_std[dim]
        gp, x_mean, x_std = train_gp(X_train, Y_train)
        gps[dim] = (gp, x_mean, x_std)
        print(f"  GP for {STATE_NAMES_7D[dim]} trained")

    # PerState rollout
    s_cur = s0.copy()
    perstate_traj = [s0.copy()]
    for step in range(min(1000, len(actions_seg))):
        delta_node = node_predict_delta(node, state_std, action_std, delta_std, s_cur, actions_seg[step])
        s_norm = s_cur / state_std
        a_norm = actions_seg[step] / action_std

        delta = np.zeros(STATE_DIM)
        for dim in range(STATE_DIM):
            if dim in gps:
                gp, xm, xs = gps[dim]
                delta[dim] = gp_predict(gp, xm, xs, s_norm, a_norm) * delta_std[dim]
            else:
                delta[dim] = delta_node[dim]

        s_next = s_cur + delta
        perstate_traj.append(s_next.copy())
        s_cur = s_next
    perstate_traj = np.array(perstate_traj)

    real = real_states[:len(node_traj)]

    # 按 horizon 分析
    print(f"\n{'H':>5} | ", end="")
    for name in STATE_NAMES_7D:
        print(f"{name:>10} | ", end="")
    print("NODE总NMAE | PS总NMAE")
    print("-" * (5 + 13 * 7 + 24))

    for h in [1, 5, 10, 20, 50, 100, 200, 500, 1000]:
        if h >= len(node_traj):
            continue
        node_err = np.abs(node_traj[h] - real[h]) / state_std
        ps_err = np.abs(perstate_traj[h] - real[h]) / state_std

        print(f"{h:>5} | ", end="")
        for j in range(STATE_DIM):
            # 用 NODE 还是 PerState 的误差？显示两者中较差的
            print(f"{max(node_err[j], ps_err[j]):10.4f} | ", end="")
        print(f"{np.mean(node_err):10.4f} | {np.mean(ps_err):10.4f}")

    # ============================================================
    # 诊断 2: e_y 和 e_psi 的误差随时间演化
    # ============================================================
    print("\n" + "=" * 80)
    print("诊断 2: e_y 和 e_psi 的误差随时间演化 (多个 segment)")
    print("=" * 80)

    for seg_i, seg in enumerate(segments[:3]):
        print(f"\n--- Segment {seg_i} ---")
        s0 = seg['states'][0].copy()
        actions_seg = seg['actions']
        real_states = seg['states']

        # NODE rollout
        s_cur = s0.copy()
        node_ey = [s0[0]]
        node_epsi = [s0[1]]
        for step in range(min(500, len(actions_seg))):
            s_next = node_predict(node, state_std, action_std, delta_std, s_cur, actions_seg[step])
            node_ey.append(s_next[0])
            node_epsi.append(s_next[1])
            s_cur = s_next

        # PerState rollout
        s_cur = s0.copy()
        ps_ey = [s0[0]]
        ps_epsi = [s0[1]]
        for step in range(min(500, len(actions_seg))):
            delta_node = node_predict_delta(node, state_std, action_std, delta_std, s_cur, actions_seg[step])
            s_norm = s_cur / state_std
            a_norm = actions_seg[step] / action_std

            delta = np.zeros(STATE_DIM)
            for dim in range(STATE_DIM):
                if dim in gps:
                    gp, xm, xs = gps[dim]
                    delta[dim] = gp_predict(gp, xm, xs, s_norm, a_norm) * delta_std[dim]
                else:
                    delta[dim] = delta_node[dim]

            s_next = s_cur + delta
            ps_ey.append(s_next[0])
            ps_epsi.append(s_next[1])
            s_cur = s_next

        node_ey = np.array(node_ey)
        node_epsi = np.array(node_epsi)
        ps_ey = np.array(ps_ey)
        ps_epsi = np.array(ps_epsi)
        real_ey = real_states[:len(node_ey), 0]
        real_epsi = real_states[:len(node_epsi), 1]

        print(f"  {'Step':>6} | {'NODE e_y':>10} {'PS e_y':>10} {'Real e_y':>10} | {'NODE e_psi':>10} {'PS e_psi':>10} {'Real e_psi':>10}")
        for step in [0, 9, 19, 49, 99, 199, 499]:
            if step < len(node_ey):
                print(f"  {step+1:>6} | {node_ey[step]:10.4f} {ps_ey[step]:10.4f} {real_ey[step]:10.4f} "
                      f"| {node_epsi[step]:10.4f} {ps_epsi[step]:10.4f} {real_epsi[step]:10.4f}")

    # ============================================================
    # 诊断 3: GP 预测的 e_y/e_psi delta 是否真的比 NODE 好？
    # ============================================================
    print("\n" + "=" * 80)
    print("诊断 3: GP vs NODE 对 e_y/e_psi 的单步预测对比")
    print("=" * 80)

    test_s = data['test_states']
    test_a = data['test_actions']
    test_d = data['test_deltas']

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(test_s), 500, replace=False)

    for dim, name in [(0, 'e_y'), (1, 'e_psi')]:
        node_errors = []
        gp_errors = []
        gp_res = gps[dim]

        for i in sample_idx:
            # NODE delta
            delta_node = node_predict_delta(node, state_std, action_std, delta_std, test_s[i], test_a[i])
            real_delta = test_d[i]

            # GP delta
            s_norm = test_s[i] / state_std
            a_norm = test_a[i] / action_std
            gp_delta = gp_predict(gp_res[0], gp_res[1], gp_res[2], s_norm, a_norm) * delta_std[dim]

            node_errors.append(abs(delta_node[dim] - real_delta[dim]))
            gp_errors.append(abs(gp_delta - real_delta[dim]))

        node_errors = np.array(node_errors)
        gp_errors = np.array(gp_errors)

        print(f"\n  {name}:")
        print(f"    NODE MAE: {np.mean(node_errors):.6f}")
        print(f"    GP   MAE: {np.mean(gp_errors):.6f}")
        print(f"    GP/NODE:  {np.mean(gp_errors)/np.mean(node_errors):.2f}x")
        print(f"    GP better: {np.mean(gp_errors < node_errors):.1%} of samples")

    # ============================================================
    # 诊断 4: e_y 的动力学结构
    # ============================================================
    print("\n" + "=" * 80)
    print("诊断 4: e_y 的动力学结构 — e_y 与 e_psi、v 的关系")
    print("=" * 80)

    # e_y_dot 应该 ≈ v * sin(e_psi)
    # 如果 e_psi ≈ 0，则 e_y 变化很小
    # 如果 e_psi 大，则 e_y 快速变化

    # 计算 e_y_dot 与 v * sin(e_psi) 的相关性
    ey_dot = train_d[:, 0] / DT  # e_y 的导数
    v = train_s[:, 2]
    epsi = train_s[:, 1]
    expected_ey_dot = v * np.sin(epsi)

    correlation = np.corrcoef(ey_dot, expected_ey_dot)[0, 1]
    print(f"\n  corr(e_y_dot, v*sin(e_psi)) = {correlation:.4f}")

    # 分段分析
    print(f"\n  |e_psi| < 0.1: e_y_dot mean = {np.mean(np.abs(ey_dot[np.abs(epsi) < 0.1])):.6f}")
    print(f"  |e_psi| > 0.5: e_y_dot mean = {np.mean(np.abs(ey_dot[np.abs(epsi) > 0.5])):.6f}")
    print(f"  |e_psi| > 1.0: e_y_dot mean = {np.mean(np.abs(ey_dot[np.abs(epsi) > 1.0])):.6f}")

    # e_psi 与 e_y 的关系
    print(f"\n  真实数据中 e_psi 的分布:")
    print(f"    mean: {np.mean(epsi):.4f}")
    print(f"    std:  {np.std(epsi):.4f}")
    print(f"    |e_psi| < 0.1: {np.mean(np.abs(epsi) < 0.1):.1%}")
    print(f"    |e_psi| > 0.5: {np.mean(np.abs(epsi) > 0.5):.1%}")
    print(f"    |e_psi| > 1.0: {np.mean(np.abs(epsi) > 1.0):.1%}")

    # ============================================================
    # 诊断 5: NODE 预测的 e_psi 为什么不准？
    # ============================================================
    print("\n" + "=" * 80)
    print("诊断 5: NODE 对 e_psi 的预测误差来源")
    print("=" * 80)

    # 分析 NODE 的 e_psi 预测误差与 e_psi 大小的关系
    node_epsi_errors = []
    node_epsi_preds = []
    real_epsis = []

    for i in sample_idx:
        delta_node = node_predict_delta(node, state_std, action_std, delta_std, test_s[i], test_a[i])
        real_delta = test_d[i]
        node_epsi_errors.append(delta_node[1] - real_delta[1])
        node_epsi_preds.append(delta_node[1])
        real_epsis.append(test_s[i, 1])  # e_psi

    node_epsi_errors = np.array(node_epsi_errors)
    node_epsi_preds = np.array(node_epsi_preds)
    real_epsis = np.array(real_epsis)

    print(f"\n  NODE e_psi delta 预测统计:")
    print(f"    mean pred: {np.mean(node_epsi_preds):.6f}")
    print(f"    mean real: {np.mean(test_d[sample_idx, 1]):.6f}")
    print(f"    error mean: {np.mean(node_epsi_errors):.6f}")
    print(f"    error std:  {np.std(node_epsi_errors):.6f}")

    # 分段分析
    for lo, hi, label in [(-0.1, 0.1, 'small'), (0.5, 1.5, 'medium'), (1.5, 3.0, 'large')]:
        mask = (np.abs(real_epsis) >= lo) & (np.abs(real_epsis) < hi)
        if mask.sum() > 0:
            print(f"\n  e_psi in [{lo}, {hi}) ({label}):")
            print(f"    count: {mask.sum()}")
            print(f"    NODE error mean: {np.mean(np.abs(node_epsi_errors[mask])):.6f}")
            print(f"    NODE error std:  {np.std(node_epsi_errors[mask]):.6f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
