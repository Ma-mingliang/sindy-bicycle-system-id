"""诊断 Neural ODE 预测行为：符号一致率、delta 大小、与 GP 对比。"""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
from canonical_7d.data_loader import load_7d_data, get_test_segments
from canonical_7d.config import DataConfig
from canonical_node.config_v9 import STATE_NAMES_7D, STATE_DIM, ACTION_DIM

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# 加载训练好的 Neural ODE（复现 V9 最佳配置）
# ============================================================
class ODEFunc(nn.Module):
    def __init__(self, hidden=64, depth=3, activation='tanh'):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)
        # 最后一层小初始化
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class NeuralODE7D:
    def __init__(self, state_std, action_std, delta_std, dt=1/30):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._dt = dt
        self._model = None

    def load(self, checkpoint_path):
        self._model = ODEFunc(hidden=64, depth=3, activation='tanh').to(DEVICE)
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def predict(self, s, tau):
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0).to(DEVICE)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).cpu().numpy()[0]
        dsdt = dsdt_norm * self._delta_std * self._dt
        return s + dsdt

    def predict_delta(self, s, tau):
        """返回物理空间的 delta（非归一化）。"""
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0).to(DEVICE)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).cpu().numpy()[0]
        return dsdt_norm * self._delta_std * self._dt

    def predict_delta_norm(self, s, tau):
        """返回归一化 delta。"""
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0).to(DEVICE)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).cpu().numpy()[0]
        return dsdt_norm


def main():
    print("Loading data and Neural ODE...")
    cfg = DataConfig(
        data_path='D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz',
        seed=SEED,
    )
    data = load_7d_data(cfg)

    state_std = data['state_std']
    action_std = data['action_std']
    delta_std = data['delta_std']

    # 加载 Neural ODE checkpoint
    node = NeuralODE7D(state_std, action_std, delta_std)
    ckpt_path = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/checkpoints/BEST_NEURAL_ODE_V9.pt'
    try:
        node.load(ckpt_path)
        print(f"Loaded checkpoint from {ckpt_path}")
    except Exception as e:
        print(f"Cannot load checkpoint: {e}")
        print("Training a fresh Neural ODE for comparison...")
        node = train_fresh_node(data, state_std, action_std, delta_std)

    # ============================================================
    # 分析 1: Neural ODE 单步 delta vs 真实 delta
    # ============================================================
    print("\n" + "=" * 80)
    print("分析 1: Neural ODE 单步 delta vs 真实 delta")
    print("=" * 80)

    test_s = data['test_states']
    test_a = data['test_actions']
    test_d = data['test_deltas']

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(test_s), 500, replace=False)

    node_deltas = []
    real_deltas = []

    for i in sample_idx:
        delta_node = node.predict_delta(test_s[i], test_a[i])
        node_deltas.append(delta_node)
        real_deltas.append(test_d[i])

    node_deltas = np.array(node_deltas)
    real_deltas = np.array(real_deltas)

    print(f"\n{'状态':>12} | {'真实|delta|均值':>14} | {'NODE|delta|均值':>14} | {'比值':>8} | {'符号一致率':>10}")
    print("-" * 70)
    for j, name in enumerate(STATE_NAMES_7D):
        real_mean = np.mean(np.abs(real_deltas[:, j]))
        node_mean = np.mean(np.abs(node_deltas[:, j]))
        ratio = node_mean / real_mean if real_mean > 1e-10 else float('nan')
        sign_match = np.mean(np.sign(node_deltas[:, j]) == np.sign(real_deltas[:, j]))
        print(f"  {name:>12} | {real_mean:14.6f} | {node_mean:14.6f} | {ratio:8.4f} | {sign_match:9.1%}")

    # 总体符号一致率
    overall_sign = np.mean(np.sign(node_deltas) == np.sign(real_deltas))
    print(f"\n  总体符号一致率: {overall_sign:.1%}")

    # ============================================================
    # 分析 2: 对比 GP 和 Neural ODE 的符号一致率
    # ============================================================
    print("\n" + "=" * 80)
    print("分析 2: GP vs Neural ODE 符号一致率对比")
    print("=" * 80)

    # 重新训练 GP
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    train_s = data['train_states']
    train_a = data['train_actions']
    train_d = data['train_deltas']

    max_samples = 2000
    idx = rng.choice(len(train_s), max_samples, replace=False)
    X = np.column_stack([train_s[idx] / state_std, train_a[idx].reshape(-1, 1) / action_std])
    Y = train_d[idx] / delta_std
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    X_scaled = (X - x_mean) / x_std

    gps = []
    kernel = Matern(nu=2.5, length_scale=1.0)
    for col in range(STATE_DIM):
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-3)
        gp.fit(X_scaled, Y[:, col])
        gps.append(gp)

    gp_deltas = []
    for i in sample_idx:
        s_norm = test_s[i] / state_std
        a_norm = test_a[i] / action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        x_scaled = (x - x_mean) / x_std
        delta_norm = np.array([gp.predict(x_scaled)[0] for gp in gps])
        gp_deltas.append(delta_norm * delta_std)
    gp_deltas = np.array(gp_deltas)

    print(f"\n{'状态':>12} | {'GP符号一致率':>12} | {'NODE符号一致率':>14} | {'真实|delta|':>12}")
    print("-" * 60)
    for j, name in enumerate(STATE_NAMES_7D):
        gp_sign = np.mean(np.sign(gp_deltas[:, j]) == np.sign(real_deltas[:, j]))
        node_sign = np.mean(np.sign(node_deltas[:, j]) == np.sign(real_deltas[:, j]))
        real_mean = np.mean(np.abs(real_deltas[:, j]))
        print(f"  {name:>12} | {gp_sign:11.1%} | {node_sign:13.1%} | {real_mean:12.6f}")

    # ============================================================
    # 分析 3: Neural ODE 的 delta 是否也接近零？
    # ============================================================
    print("\n" + "=" * 80)
    print("分析 3: Neural ODE delta 是否也接近零？")
    print("=" * 80)

    print(f"\n{'状态':>12} | {'NODE|delta|/|state|':>18} | {'GP|delta|/|state|':>16}")
    print("-" * 55)
    for j, name in enumerate(STATE_NAMES_7D):
        node_ratio = np.mean(np.abs(node_deltas[:, j])) / (np.mean(np.abs(test_s[sample_idx, j])) + 1e-8)
        gp_ratio = np.mean(np.abs(gp_deltas[:, j])) / (np.mean(np.abs(test_s[sample_idx, j])) + 1e-8)
        print(f"  {name:>12} | {node_ratio:18.6f} | {gp_ratio:16.6f}")

    # ============================================================
    # 分析 4: 多步 rollout 中 Neural ODE 的 delta 变化
    # ============================================================
    print("\n" + "=" * 80)
    print("分析 4: Neural ODE 多步 rollout 中 delta 的变化")
    print("=" * 80)

    segments = get_test_segments(data, n_segments=3, segment_length=1100, seed=SEED)
    seg = segments[0]
    s0 = seg['states'][0].copy()
    actions_seg = seg['actions']
    real_states = seg['states']

    s_cur = s0.copy()
    node_trajectory = [s0.copy()]
    delta_magnitudes = []

    for step in range(min(500, len(actions_seg))):
        delta_node = node.predict_delta(s_cur, actions_seg[step])
        delta_magnitudes.append(np.linalg.norm(delta_node))
        s_next = s_cur + delta_node
        node_trajectory.append(s_next.copy())
        s_cur = s_next

    delta_magnitudes = np.array(delta_magnitudes)
    node_trajectory = np.array(node_trajectory)

    print(f"\n  Neural ODE 按窗口统计 |delta|:")
    for start, end in [(0, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500)]:
        end = min(end, len(delta_magnitudes))
        if start < end:
            window = delta_magnitudes[start:end]
            print(f"    Steps {start:>3}-{end:>3}: mean={np.mean(window):.6f}, std={np.std(window):.6f}")

    print(f"\n  Neural ODE rollout 轨迹:")
    print(f"  {'Step':>6} | {'NODE e_y':>10} | {'真实 e_y':>10} | {'NODE e_psi':>10} | {'真实 e_psi':>10}")
    print("  " + "-" * 55)
    for step in [0, 4, 9, 19, 49, 99, 199, 499]:
        if step < len(node_trajectory) and step + 1 < len(real_states):
            print(f"  {step+1:>6} | {node_trajectory[step+1, 0]:10.4f} | {real_states[step+1, 0]:10.4f} "
                  f"| {node_trajectory[step+1, 1]:10.4f} | {real_states[step+1, 1]:10.4f}")

    print("\nDone.")


def train_fresh_node(data, state_std, action_std, delta_std):
    """如果没有 checkpoint，快速训练一个 Neural ODE。"""
    train_s = data['train_states']
    train_a = data['train_actions']
    train_d = data['train_deltas']
    dt = 1/30

    # 归一化
    s_norm = train_s / state_std
    a_norm = train_a / action_std
    derivs = train_d / (delta_std * dt)

    train_x = torch.FloatTensor(np.column_stack([s_norm, a_norm.reshape(-1, 1)])).to(DEVICE)
    train_y = torch.FloatTensor(derivs).to(DEVICE)

    model = ODEFunc(hidden=64, depth=3, activation='tanh').to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(train_x, train_y)
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    model.train()
    for epoch in range(50):
        total_loss = 0
        for xb, yb in loader:
            pred = model(xb[:, :STATE_DIM], xb[:, STATE_DIM:STATE_DIM+1])
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={total_loss/len(loader):.6f}")
    model.eval()

    node = NeuralODE7D(state_std, action_std, delta_std, dt)
    node._model = model
    return node


if __name__ == '__main__':
    main()
