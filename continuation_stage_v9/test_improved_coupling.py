"""改进 Coupling Network — 多版本对比测试。

针对第一性原理发现的问题，创建 5 个改进版本：
V1: 更大耦合维度 (128)
V2: 解码器可以看到所有状态 (cross-state attention)
V3: 残差连接 (skip connection)
V4: V1+V2+V3 组合
V5: V4 + 多步训练

用法:
    python test_improved_coupling.py
"""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')
sys.stdout.reconfigure(line_buffering=True)

import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from canonical_7d.data_loader import load_7d_data, get_test_segments
from canonical_7d.config import DataConfig
from canonical_node.evaluation_v9 import multi_step_evaluate
from canonical_node.config_v9 import (
    STATE_NAMES_7D, STATE_DIM, ACTION_DIM,
    ROLLOUT_HORIZONS, N_EVAL_SEGMENTS, SEGMENT_LENGTH,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
DT = 1.0 / 30.0


# ============================================================
# V0: 原始 Coupling Network (基线)
# ============================================================
class CouplingEncoderV0(nn.Module):
    def __init__(self, input_dim, coupling_dim=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, coupling_dim),
        )

    def forward(self, x):
        return self.net(x)


class StateDecoderV0(nn.Module):
    def __init__(self, coupling_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coupling_dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

    def forward(self, z, own_state):
        x = torch.cat([z, own_state.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


class CouplingNetworkV0(nn.Module):
    """原始版本: 32D 瓶颈 + 独立解码器。"""
    def __init__(self):
        super().__init__()
        self.encoder = CouplingEncoderV0(STATE_DIM + ACTION_DIM, coupling_dim=32, hidden=64)
        self.decoders = nn.ModuleList([StateDecoderV0(32, 32) for _ in range(STATE_DIM)])

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        z = self.encoder(x)
        deltas = [decoder(z, s[:, i]) for i, decoder in enumerate(self.decoders)]
        return torch.stack(deltas, dim=-1)


# ============================================================
# V1: 更大耦合维度 (128)
# ============================================================
class CouplingNetworkV1(nn.Module):
    """V1: 耦合维度从 32 增加到 128，减少信息瓶颈。"""
    def __init__(self):
        super().__init__()
        self.encoder = CouplingEncoderV0(STATE_DIM + ACTION_DIM, coupling_dim=128, hidden=128)
        self.decoders = nn.ModuleList([StateDecoderV0(128, 64) for _ in range(STATE_DIM)])

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        z = self.encoder(x)
        deltas = [decoder(z, s[:, i]) for i, decoder in enumerate(self.decoders)]
        return torch.stack(deltas, dim=-1)


# ============================================================
# V2: 解码器可以看到所有状态 (cross-state)
# ============================================================
class StateDecoderV2(nn.Module):
    """V2 解码器: 输入包含耦合向量 + 所有状态 (不只是自身)。"""
    def __init__(self, coupling_dim, state_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coupling_dim + state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)

    def forward(self, z, all_states):
        x = torch.cat([z, all_states], dim=-1)
        return self.net(x).squeeze(-1)


class CouplingNetworkV2(nn.Module):
    """V2: 解码器可以看到所有状态，不只是自身。"""
    def __init__(self):
        super().__init__()
        self.encoder = CouplingEncoderV0(STATE_DIM + ACTION_DIM, coupling_dim=32, hidden=64)
        self.decoders = nn.ModuleList([StateDecoderV2(32, STATE_DIM, 64) for _ in range(STATE_DIM)])

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        z = self.encoder(x)
        deltas = [decoder(z, s) for decoder in self.decoders]
        return torch.stack(deltas, dim=-1)


# ============================================================
# V3: 残差连接 (skip connection)
# ============================================================
class CouplingNetworkV3(nn.Module):
    """V3: 残差连接 — 编码器输出直接加到解码器输入。"""
    def __init__(self):
        super().__init__()
        self.encoder = CouplingEncoderV0(STATE_DIM + ACTION_DIM, coupling_dim=32, hidden=64)
        self.decoders = nn.ModuleList([StateDecoderV0(32, 32) for _ in range(STATE_DIM)])
        # 残差连接: 直接从输入映射到输出
        self.skip = nn.Linear(STATE_DIM + ACTION_DIM, STATE_DIM, bias=False)
        nn.init.eye_(self.skip.weight[:STATE_DIM, :STATE_DIM])

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        z = self.encoder(x)
        deltas = [decoder(z, s[:, i]) for i, decoder in enumerate(self.decoders)]
        coupling_out = torch.stack(deltas, dim=-1)
        skip_out = self.skip(x)
        return coupling_out + skip_out


# ============================================================
# V4: V1+V2+V3 组合 (大维度 + cross-state + 残差)
# ============================================================
class CouplingNetworkV4(nn.Module):
    """V4: 组合所有改进 — 大耦合维度 + cross-state + 残差连接。"""
    def __init__(self):
        super().__init__()
        self.encoder = CouplingEncoderV0(STATE_DIM + ACTION_DIM, coupling_dim=128, hidden=128)
        self.decoders = nn.ModuleList([StateDecoderV2(128, STATE_DIM, 64) for _ in range(STATE_DIM)])
        self.skip = nn.Linear(STATE_DIM + ACTION_DIM, STATE_DIM, bias=False)
        nn.init.eye_(self.skip.weight[:STATE_DIM, :STATE_DIM])

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        z = self.encoder(x)
        deltas = [decoder(z, s) for decoder in self.decoders]
        coupling_out = torch.stack(deltas, dim=-1)
        skip_out = self.skip(x)
        return coupling_out + skip_out


# ============================================================
# V5: V4 + 多步训练
# ============================================================
class CouplingNetworkV5(CouplingNetworkV4):
    """V5: 与 V4 相同架构，但使用多步 rollout 训练。"""
    pass


# ============================================================
# 训练函数
# ============================================================
def train_model(model, train_s, train_a, train_d, state_std, action_std, delta_std,
                n_epochs=200, lr=1e-3, batch_size=256, multi_step=False, rollout_len=5):
    """训练模型。"""
    s_norm = train_s / state_std
    a_norm = train_a / action_std
    derivs = train_d / (delta_std * DT)

    train_x = torch.FloatTensor(np.column_stack([s_norm, a_norm.reshape(-1, 1)])).to(DEVICE)
    train_y = torch.FloatTensor(derivs).to(DEVICE)

    ds = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for xb, yb in loader:
            s_b = xb[:, :STATE_DIM]
            a_b = xb[:, STATE_DIM:]

            pred = model(s_b, a_b)
            loss = crit(pred, yb)

            # 多步损失
            if multi_step and epoch >= 50:
                s_cur = s_b.detach()
                multi_loss = torch.tensor(0.0, device=DEVICE)
                for step in range(rollout_len - 1):
                    pred_deriv = model(s_cur, a_b)
                    s_next = s_cur + pred_deriv * DT
                    target_deriv = model(s_next.detach(), a_b)
                    multi_loss = multi_loss + crit(pred_deriv, target_deriv)
                    s_cur = s_next
                loss = loss + 0.3 * multi_loss / (rollout_len - 1)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        scheduler.step()
        if epoch % 50 == 0:
            print(f"    Epoch {epoch}: loss={total_loss/len(loader):.4f}")

    model.eval()
    return model


# ============================================================
# 模型包装器 (用于 V9 评估框架)
# ============================================================
class CouplingModelWrapper:
    def __init__(self, model, state_std, action_std, delta_std):
        self._model = model
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

    def predict(self, s, tau):
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0).to(DEVICE)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).cpu().numpy()[0]
        return s + dsdt_norm * self._delta_std * DT


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 80)
    print("改进 Coupling Network — 多版本对比测试")
    print("=" * 80)
    print(f"Device: {DEVICE}\n")

    # 加载数据
    print("Loading data...")
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
    segments = get_test_segments(data, n_segments=N_EVAL_SEGMENTS,
                                 segment_length=SEGMENT_LENGTH, seed=SEED)
    print(f"  Train: {len(train_s)}, Segments: {len(segments)}\n")

    # ============================================================
    # 定义所有版本
    # ============================================================
    versions = [
        ('V0: Original (32D bottleneck)', CouplingNetworkV0(), False),
        ('V1: Large coupling (128D)', CouplingNetworkV1(), False),
        ('V2: Cross-state decoders', CouplingNetworkV2(), False),
        ('V3: Residual connection', CouplingNetworkV3(), False),
        ('V4: V1+V2+V3 combined', CouplingNetworkV4(), False),
        ('V5: V4 + multi-step', CouplingNetworkV5(), True),
    ]

    # ============================================================
    # 训练和评估所有版本
    # ============================================================
    all_results = {}

    for name, model, multi_step in versions:
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")

        t0 = time.time()
        trained_model = train_model(
            model, train_s, train_a, train_d,
            state_std, action_std, delta_std,
            n_epochs=200, lr=1e-3, batch_size=256,
            multi_step=multi_step, rollout_len=5,
        )
        print(f"  Training done in {time.time()-t0:.1f}s")

        # 包装为 V9 评估接口
        wrapper = CouplingModelWrapper(trained_model, state_std, action_std, delta_std)

        print(f"Evaluating {name}...")
        results = multi_step_evaluate(wrapper, segments, state_std, ROLLOUT_HORIZONS)
        all_results[name] = results

    # 加载 V9 参考
    v9_ref = {
        1: 0.0051, 5: 0.0170, 10: 0.0628, 20: 0.1817,
        50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529, 1000: 0.6443,
    }

    # ============================================================
    # 输出对比表
    # ============================================================
    print("\n" + "=" * 120)
    print("综合对比: 所有 Coupling Network 版本")
    print("=" * 120)

    header = f"{'H':>5} | {'V9(ref)':>10}"
    for name, _, _ in versions:
        short_name = name.split(':')[0]
        header += f" | {short_name:>10}"
    print(header)
    print("-" * (5 + 13 * (1 + len(versions))))

    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        row += f" | {v9_ref.get(h, float('nan')):10.4f}"
        for name, _, _ in versions:
            row += f" | {all_results[name][h]['nmae_mean']:10.4f}"
        print(row)

    # 存活率
    print("\n存活率:")
    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        row += f" | {'100%':>10}"
        for name, _, _ in versions:
            row += f" | {all_results[name][h]['survival_rate']:9.0%}%"
        print(row)

    # 改善百分比
    print("\n改善百分比 (vs V9 NODE):")
    for h in ROLLOUT_HORIZONS:
        v9_val = v9_ref.get(h, None)
        if v9_val is None:
            continue
        row = f"{h:>5}"
        row += f" | {'---':>10}"
        for name, _, _ in versions:
            val = all_results[name][h]['nmae_mean']
            imp = (val - v9_val) / v9_val * 100
            row += f" | {imp:>+9.1f}%"
        print(row)

    # 保存
    log_dir = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/logs')
    log_dir.mkdir(exist_ok=True)
    out_path = log_dir / 'improved_coupling_results.json'

    save_results = {}
    for h in ROLLOUT_HORIZONS:
        save_results[str(h)] = {'v9_ref': v9_ref.get(h, None)}
        for name, _, _ in versions:
            save_results[str(h)][name] = all_results[name][h]['nmae_mean']

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'results': save_results}, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
