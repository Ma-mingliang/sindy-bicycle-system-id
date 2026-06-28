"""方案 2: 收缩性约束 Neural ODE。

强制 Jacobian 特征值为负，保证动力学是收缩的，
从而改善长期稳定性。

V11 结果: H=500 从 0.5529 降到 0.4748 (改善 14%)

用法:
    python test_contractive_node.py
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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
# 收缩性 Neural ODE
# ============================================================
class ContractiveODEFunc(nn.Module):
    """带 Jacobian 正则化的 Neural ODE。"""

    def __init__(self, hidden=64, depth=3, activation='tanh'):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class ContractiveNeuralODE:
    def __init__(self, state_std, action_std, delta_std):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._model = None

    def train(self, train_s, train_a, train_d, n_epochs=200, lr=1e-3,
              lambda_contractive=0.05, warmup_epochs=20):
        """训练带收缩性约束的 Neural ODE。"""
        s_norm = torch.FloatTensor(train_s / self._state_std).to(DEVICE)
        a_norm = torch.FloatTensor(train_a / self._action_std).unsqueeze(1).to(DEVICE)
        derivs = torch.FloatTensor(train_d / (self._delta_std * DT)).to(DEVICE)

        train_x = torch.cat([s_norm, a_norm], dim=1)
        train_y = derivs

        self._model = ContractiveODEFunc(hidden=64, depth=3, activation='tanh').to(DEVICE)
        opt = optim.Adam(self._model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        crit = nn.MSELoss()

        ds = TensorDataset(train_x, train_y)
        loader = DataLoader(ds, batch_size=256, shuffle=True)

        self._model.train()
        for epoch in range(n_epochs):
            total_data = 0.0
            total_contract = 0.0
            n_batches = 0

            for xb, yb in loader:
                s_b = xb[:, :STATE_DIM].requires_grad_(True)
                a_b = xb[:, STATE_DIM:]

                pred = self._model(s_b, a_b)
                loss_data = crit(pred, yb)

                # 收缩性正则化: 惩罚 Jacobian 最大特征值
                loss_contract = torch.tensor(0.0, device=DEVICE)
                if epoch >= warmup_epochs and lambda_contractive > 0:
                    # 计算 Jacobian (对状态的梯度)
                    jac = torch.zeros(s_b.shape[0], STATE_DIM, STATE_DIM, device=DEVICE)
                    for j in range(STATE_DIM):
                        grad = torch.autograd.grad(
                            pred[:, j].sum(), s_b, create_graph=True
                        )[0]
                        jac[:, j, :] = grad

                    # 对称部分的特征值
                    jac_sym = (jac + jac.transpose(-1, -2)) / 2
                    eigvals = torch.linalg.eigvalsh(jac_sym)
                    max_eig = eigvals.max(dim=-1).values
                    # 惩罚正特征值（扩张区域）
                    loss_contract = torch.relu(max_eig).mean()

                loss = loss_data + lambda_contractive * loss_contract

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()

                total_data += loss_data.item()
                total_contract += loss_contract.item()
                n_batches += 1

            scheduler.step()

            if epoch % 20 == 0:
                print(f"    Epoch {epoch}: data={total_data/n_batches:.4f}, "
                      f"contract={total_contract/n_batches:.4f}, "
                      f"lr={scheduler.get_last_lr()[0]:.6f}")

        self._model.eval()

    def predict(self, s, tau):
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0).to(DEVICE)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).cpu().numpy()[0]
        return s + dsdt_norm * self._delta_std * DT


# ============================================================
# 基线 Neural ODE (无收缩)
# ============================================================
class BaselineODEFunc(nn.Module):
    def __init__(self, hidden=64, depth=3, activation='tanh'):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.SiLU
        layers = [nn.Linear(STATE_DIM + ACTION_DIM, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class BaselineNeuralODE:
    def __init__(self, state_std, action_std, delta_std):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._model = None

    def train(self, train_s, train_a, train_d, n_epochs=200, lr=1e-3):
        s_norm = torch.FloatTensor(train_s / self._state_std).to(DEVICE)
        a_norm = torch.FloatTensor(train_a / self._action_std).unsqueeze(1).to(DEVICE)
        derivs = torch.FloatTensor(train_d / (self._delta_std * DT)).to(DEVICE)

        train_x = torch.cat([s_norm, a_norm], dim=1)
        train_y = derivs

        self._model = BaselineODEFunc(hidden=64, depth=3, activation='tanh').to(DEVICE)
        opt = optim.Adam(self._model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        crit = nn.MSELoss()

        ds = TensorDataset(train_x, train_y)
        loader = DataLoader(ds, batch_size=256, shuffle=True)

        self._model.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for xb, yb in loader:
                pred = self._model(xb[:, :STATE_DIM], xb[:, STATE_DIM:])
                loss = crit(pred, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
            scheduler.step()
            if epoch % 20 == 0:
                print(f"    Epoch {epoch}: loss={total_loss/len(loader):.4f}")
        self._model.eval()

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
    print("方案 2: 收缩性约束 Neural ODE")
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
    # 训练基线 Neural ODE (无收缩)
    # ============================================================
    print("[1] Training baseline Neural ODE (no contractivity)...")
    t0 = time.time()
    baseline_node = BaselineNeuralODE(state_std, action_std, delta_std)
    baseline_node.train(train_s, train_a, train_d, n_epochs=200, lr=1e-3)
    print(f"  Done in {time.time()-t0:.1f}s\n")

    print("[2] Evaluating baseline NODE...")
    results_baseline = multi_step_evaluate(baseline_node, segments, state_std, ROLLOUT_HORIZONS)

    # ============================================================
    # 训练收缩性 Neural ODE (不同 lambda)
    # ============================================================
    lambdas = [0.01, 0.05, 0.1]
    contractive_results = {}

    for lam in lambdas:
        print(f"\n[3] Training contractive NODE (lambda={lam})...")
        t0 = time.time()
        c_node = ContractiveNeuralODE(state_std, action_std, delta_std)
        c_node.train(train_s, train_a, train_d, n_epochs=200, lr=1e-3,
                     lambda_contractive=lam, warmup_epochs=20)
        print(f"  Done in {time.time()-t0:.1f}s")

        print(f"[4] Evaluating contractive NODE (lambda={lam})...")
        results_c = multi_step_evaluate(c_node, segments, state_std, ROLLOUT_HORIZONS)
        contractive_results[lam] = results_c

    # ============================================================
    # 加载 V9 checkpoint 作为参考
    # ============================================================
    print("\n[5] Loading V9 checkpoint for reference...")
    from test_improved_gp_v2 import NeuralODE7D
    v9_node = NeuralODE7D(state_std, action_std, delta_std)
    ckpt_path = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/checkpoints/BEST_NEURAL_ODE_V9.pt')
    if ckpt_path.exists():
        v9_node.load_checkpoint(ckpt_path)
        results_v9 = multi_step_evaluate(v9_node, segments, state_std, ROLLOUT_HORIZONS)
    else:
        results_v9 = results_baseline

    # ============================================================
    # 输出对比表
    # ============================================================
    print("\n" + "=" * 100)
    print("综合对比: 收缩性约束 Neural ODE")
    print("=" * 100)

    v9_ref = {
        1: 0.0051, 5: 0.0170, 10: 0.0628, 20: 0.1817,
        50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529, 1000: 0.6443,
    }

    header = f"{'H':>5} | {'V9(ref)':>10} | {'Baseline':>10}"
    for lam in lambdas:
        header += f" | {'C-lam'+str(lam):>10}"
    print(header)
    print("-" * (5 + 13 * (3 + len(lambdas))))

    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        row += f" | {v9_ref.get(h, float('nan')):10.4f}"
        row += f" | {results_baseline[h]['nmae_mean']:10.4f}"
        for lam in lambdas:
            row += f" | {contractive_results[lam][h]['nmae_mean']:10.4f}"
        print(row)

    # 存活率
    print("\n存活率:")
    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        row += f" | {'100%':>10}"
        row += f" | {results_baseline[h]['survival_rate']:9.0%}%"
        for lam in lambdas:
            row += f" | {contractive_results[lam][h]['survival_rate']:9.0%}%"
        print(row)

    # 改善百分比
    print("\n改善百分比 (vs V9 baseline):")
    for h in ROLLOUT_HORIZONS:
        v9_val = v9_ref.get(h, None)
        if v9_val is None:
            continue
        row = f"{h:>5}"
        row += f" | {'---':>10}"
        base_imp = (results_baseline[h]['nmae_mean'] - v9_val) / v9_val * 100
        row += f" | {base_imp:>+9.1f}%"
        for lam in lambdas:
            c_val = contractive_results[lam][h]['nmae_mean']
            imp = (c_val - v9_val) / v9_val * 100
            row += f" | {imp:>+9.1f}%"
        print(row)

    # 保存结果
    log_dir = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/logs')
    log_dir.mkdir(exist_ok=True)
    out_path = log_dir / 'contractive_node_results.json'

    all_results = {}
    for h in ROLLOUT_HORIZONS:
        all_results[str(h)] = {
            'v9_ref': v9_ref.get(h, None),
            'baseline': results_baseline[h]['nmae_mean'],
            'baseline_survival': results_baseline[h]['survival_rate'],
        }
        for lam in lambdas:
            all_results[str(h)][f'contractive_{lam}'] = contractive_results[lam][h]['nmae_mean']
            all_results[str(h)][f'contractive_{lam}_survival'] = contractive_results[lam][h]['survival_rate']

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'results': all_results, 'lambdas': lambdas}, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
