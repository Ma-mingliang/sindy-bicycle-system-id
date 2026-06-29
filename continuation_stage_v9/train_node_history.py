"""方向 3: 历史增强 Neural ODE。

给 NODE 添加历史信息作为额外输入：
- 方案 A: 添加前一步的动作 (s, a, a_prev) → ds/dt
- 方案 B: 添加前一步的状态变化 (s, a, ds_prev) → ds/dt
- 方案 C: 添加前 N 步的动作 (s, a, a[-1], a[-2]) → ds/dt

这可以帮助 NODE 捕捉控制器的参考目标变化。

用法:
    python train_node_history.py
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
# 历史增强 Neural ODE
# ============================================================
class HistoryODEFunc(nn.Module):
    """带历史输入的 Neural ODE。

    输入: [s_norm(7), a_norm(1), history_features]
    输出: dsdt_norm(7)

    history_features 取决于历史类型:
    - 'prev_action': [a_prev_norm(1)] → 输入 9D
    - 'prev_delta': [ds_prev_norm(7)] → 输入 15D
    - 'prev_2_actions': [a_prev_norm(1), a_prev2_norm(1)] → 输入 10D
    """

    def __init__(self, history_dim, hidden=64, depth=3):
        super().__init__()
        input_dim = STATE_DIM + ACTION_DIM + history_dim
        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*layers)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, a, history):
        return self.net(torch.cat([s, a, history], dim=-1))


class HistoryNeuralODE:
    """带历史信息的 Neural ODE。"""

    def __init__(self, state_std, action_std, delta_std, history_type='prev_action'):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._history_type = history_type
        self._history_dim = self._get_history_dim()
        self._model = None

    def _get_history_dim(self):
        if self._history_type == 'prev_action':
            return 1
        elif self._history_type == 'prev_delta':
            return STATE_DIM
        elif self._history_type == 'prev_2_actions':
            return 2
        else:
            raise ValueError(f"Unknown history type: {self._history_type}")

    def _prepare_history_data(self, states, actions, deltas):
        """准备历史特征。"""
        n = len(states)
        if self._history_type == 'prev_action':
            # 前一步动作 (用零填充第一步)
            history = np.zeros((n, 1))
            history[1:, 0] = actions[:-1]
        elif self._history_type == 'prev_delta':
            # 前一步状态变化 (用零填充第一步)
            history = np.zeros((n, STATE_DIM))
            history[1:, :] = deltas[:-1, :]
        elif self._history_type == 'prev_2_actions':
            # 前两步动作
            history = np.zeros((n, 2))
            history[1:, 0] = actions[:-1]
            history[2:, 1] = actions[:-2]
        return history

    def train(self, train_s, train_a, train_d, n_epochs=200, lr=1e-3):
        """训练历史增强 Neural ODE。"""
        # 准备历史特征
        history = self._prepare_history_data(train_s, train_a, train_d)

        # 归一化
        s_norm = train_s / self._state_std
        a_norm = train_a / self._action_std
        derivs = train_d / (self._delta_std * DT)

        # 历史特征归一化
        if self._history_type == 'prev_action':
            history_norm = history / self._action_std
        elif self._history_type == 'prev_delta':
            history_norm = history / (self._delta_std * DT)
        elif self._history_type == 'prev_2_actions':
            history_norm = history / self._action_std

        train_x = torch.FloatTensor(np.column_stack([s_norm, a_norm.reshape(-1, 1)])).to(DEVICE)
        train_h = torch.FloatTensor(history_norm).to(DEVICE)
        train_y = torch.FloatTensor(derivs).to(DEVICE)

        ds = TensorDataset(train_x, train_h, train_y)
        loader = DataLoader(ds, batch_size=256, shuffle=True)

        self._model = HistoryODEFunc(self._history_dim, hidden=64, depth=3).to(DEVICE)
        opt = optim.Adam(self._model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        crit = nn.MSELoss()

        self._model.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for xb, hb, yb in loader:
                pred = self._model(xb[:, :STATE_DIM], xb[:, STATE_DIM:STATE_DIM+1], hb)
                loss = crit(pred, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
            scheduler.step()
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: loss={total_loss/len(loader):.4f}")
        self._model.eval()

    def predict(self, s, tau, history=None):
        """预测下一状态。history 是历史特征（归一化后）。"""
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0).to(DEVICE)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0).to(DEVICE)

        if history is None:
            history = torch.zeros(1, self._history_dim).to(DEVICE)
        else:
            history = torch.FloatTensor(history).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm, history).cpu().numpy()[0]
        return s + dsdt_norm * self._delta_std * DT


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 80)
    print("方向 3: 历史增强 Neural ODE")
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
    # 训练不同历史类型
    # ============================================================
    history_types = ['prev_action', 'prev_delta', 'prev_2_actions']
    all_results = {}

    for ht in history_types:
        print(f"\n{'='*60}")
        print(f"Training: {ht}")
        print(f"{'='*60}")

        t0 = time.time()
        node = HistoryNeuralODE(state_std, action_std, delta_std, history_type=ht)
        node.train(train_s, train_a, train_d, n_epochs=200, lr=1e-3)
        print(f"  Training done in {time.time()-t0:.1f}s")

        # 评估 (需要自定义评估，因为模型需要历史输入)
        print(f"Evaluating {ht}...")
        results = evaluate_with_history(node, segments, state_std, ht, data)
        all_results[ht] = results

    # 加载 V9 参考
    v9_ref = {
        1: 0.0051, 5: 0.0170, 10: 0.0628, 20: 0.1817,
        50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529, 1000: 0.6443,
    }

    # ============================================================
    # 输出对比表
    # ============================================================
    print("\n" + "=" * 100)
    print("综合对比: 历史增强 Neural ODE")
    print("=" * 100)

    header = f"{'H':>5} | {'V9(ref)':>10}"
    for ht in history_types:
        header += f" | {ht:>15}"
    print(header)
    print("-" * (5 + 13 * (1 + len(history_types))))

    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        row += f" | {v9_ref.get(h, float('nan')):10.4f}"
        for ht in history_types:
            row += f" | {all_results[ht][h]['nmae_mean']:15.4f}"
        print(row)

    # 存活率
    print("\n存活率:")
    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        row += f" | {'100%':>10}"
        for ht in history_types:
            row += f" | {all_results[ht][h]['survival_rate']:14.0%}%"
        print(row)

    # 保存结果
    log_dir = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/logs')
    log_dir.mkdir(exist_ok=True)
    out_path = log_dir / 'history_node_results.json'

    save_results = {}
    for h in ROLLOUT_HORIZONS:
        save_results[str(h)] = {'v9_ref': v9_ref.get(h, None)}
        for ht in history_types:
            save_results[str(h)][ht] = all_results[ht][h]['nmae_mean']

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'results': save_results}, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_path}")
    print("Done.")


def evaluate_with_history(model, segments, state_std, history_type, data):
    """带历史输入的评估。"""
    from canonical_node.evaluation_v9 import compute_nmae, check_survival, compute_survival

    state_std_np = state_std
    action_std = model._action_std
    delta_std = model._delta_std

    results = {}
    for h in ROLLOUT_HORIZONS:
        nmae_list = []
        survival_list = []

        for seg in segments:
            s0 = seg['states'][0].copy()
            actions_seg = seg['actions']
            real_states = seg['states']
            n = min(h, len(actions_seg))
            predicted = [s0.copy()]
            s_cur = s0.copy()
            survived = True

            # 初始化历史
            prev_action = 0.0
            prev_delta = np.zeros(STATE_DIM)
            prev_action2 = 0.0

            for step in range(n):
                try:
                    # 构建历史特征
                    if history_type == 'prev_action':
                        history = np.array([prev_action / action_std])
                    elif history_type == 'prev_delta':
                        history = prev_delta / (delta_std * DT)
                    elif history_type == 'prev_2_actions':
                        history = np.array([prev_action / action_std, prev_action2 / action_std])
                    else:
                        history = None

                    s_next = model.predict(s_cur, actions_seg[step], history)

                    if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                        survived = False
                        break
                    if not check_survival(s_next):
                        survived = False
                        break

                    # 更新历史
                    prev_action2 = prev_action
                    prev_action = actions_seg[step]
                    prev_delta = s_next - s_cur

                    predicted.append(s_next.copy())
                    s_cur = s_next
                except Exception:
                    survived = False
                    break

            predicted = np.array(predicted)
            real = real_states[:len(predicted)]
            n_valid = min(len(predicted), len(real)) - 1
            if n_valid > 0:
                nmae = compute_nmae(predicted[1:n_valid+1], real[1:n_valid+1], state_std_np)
                nmae_list.append(nmae['overall'])
            else:
                nmae_list.append(float('nan'))
            survival_list.append(survived and n_valid >= n - 1)

        valid_nmae = [x for x in nmae_list if not np.isnan(x)]
        results[h] = {
            'nmae_mean': float(np.nanmean(valid_nmae)) if valid_nmae else float('nan'),
            'nmae_std': float(np.nanstd(valid_nmae)) if valid_nmae else float('nan'),
            'survival_rate': compute_survival(survival_list),
        }

    return results


if __name__ == '__main__':
    main()
