"""方向 2: 用 V9 课程学习训练 NODE。

V9 的训练方法：
- 课程学习: rollout 1→5→10→20 步
- 多步损失: lambda_multi=0.3
- 一致性损失: lambda_consistency=0.1
- Jacobian 正则化: lambda_jacobian=0.01
- 200 epochs, cosine annealing, gradient clipping

用法:
    python train_node_curriculum.py
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
# Neural ODE 架构
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
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


# ============================================================
# 课程学习训练器
# ============================================================
class CurriculumTrainer:
    """V9 风格的课程学习训练。"""

    def __init__(self, state_std, action_std, delta_std):
        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std
        self._model = None

    def train(self, train_s, train_a, train_d, n_epochs=200, lr=1e-3,
              lambda_multi=0.3, lambda_consistency=0.1, lambda_jacobian=0.01,
              curriculum_schedule='1,5,10,20', batch_size=256):
        """训练带课程学习的 Neural ODE。"""

        # 准备数据
        s_norm = torch.FloatTensor(train_s / self._state_std).to(DEVICE)
        a_norm = torch.FloatTensor(train_a / self._action_std).unsqueeze(1).to(DEVICE)
        derivs = torch.FloatTensor(train_d / (self._delta_std * DT)).to(DEVICE)

        train_x = torch.cat([s_norm, a_norm], dim=1)
        train_y = derivs

        ds = TensorDataset(train_x, train_y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        # 解析课程计划
        curriculum = [int(x) for x in curriculum_schedule.split(',')]
        print(f"  Curriculum: {curriculum}")

        # 初始化模型
        self._model = ODEFunc(hidden=64, depth=3, activation='tanh').to(DEVICE)
        opt = optim.Adam(self._model.parameters(), lr=lr, weight_decay=0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
        crit = nn.MSELoss()

        self._model.train()
        for epoch in range(n_epochs):
            # 确定当前 rollout 长度
            progress = epoch / n_epochs
            rollout_len = self._get_curriculum_rollout(progress, curriculum)

            total_single = 0.0
            total_multi = 0.0
            total_consist = 0.0
            total_jac = 0.0
            n_batches = 0

            for xb, yb in loader:
                s_b = xb[:, :STATE_DIM].requires_grad_(True)
                a_b = xb[:, STATE_DIM:]

                # 单步损失
                pred = self._model(s_b, a_b)
                loss_single = crit(pred, yb)

                # 多步损失 (课程学习)
                loss_multi = torch.tensor(0.0, device=DEVICE)
                if lambda_multi > 0 and rollout_len > 1:
                    loss_multi = self._compute_multi_step_loss(s_b, a_b, yb, rollout_len)

                # 一致性损失
                loss_consist = torch.tensor(0.0, device=DEVICE)
                if lambda_consistency > 0:
                    loss_consist = self._compute_consistency_loss(s_b, a_b)

                # Jacobian 正则化
                loss_jac = torch.tensor(0.0, device=DEVICE)
                if lambda_jacobian > 0 and epoch >= 20:
                    loss_jac = self._compute_jacobian_loss(s_b, a_b)

                loss = (loss_single
                        + lambda_multi * loss_multi
                        + lambda_consistency * loss_consist
                        + lambda_jacobian * loss_jac)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()

                total_single += loss_single.item()
                total_multi += loss_multi.item()
                total_consist += loss_consist.item()
                total_jac += loss_jac.item()
                n_batches += 1

            scheduler.step()

            if epoch % 20 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch:>3}: single={total_single/n_batches:.4f} "
                      f"multi={total_multi/n_batches:.4f} "
                      f"consist={total_consist/n_batches:.4f} "
                      f"jac={total_jac/n_batches:.4f} "
                      f"rollout={rollout_len} "
                      f"lr={scheduler.get_last_lr()[0]:.6f}")

        self._model.eval()

    def _get_curriculum_rollout(self, progress, curriculum):
        """根据训练进度返回当前 rollout 长度。"""
        n = len(curriculum)
        idx = min(int(progress * n), n - 1)
        return curriculum[idx]

    def _compute_multi_step_loss(self, s_b, a_b, target_deriv, rollout_len):
        """多步 rollout 损失。"""
        crit = nn.MSELoss()
        s_cur = s_b.detach()  # 不回传梯度到初始状态
        total_loss = torch.tensor(0.0, device=DEVICE)

        for step in range(rollout_len - 1):
            # 用当前模型做一步预测
            pred_deriv = self._model(s_cur, a_b)
            # Euler 积分
            s_next = s_cur + pred_deriv * DT
            # 计算下一步的导数目标
            target_next = self._model(s_next.detach(), a_b)
            # 多步损失：预测的导数应该一致
            total_loss = total_loss + crit(pred_deriv, target_deriv)
            s_cur = s_next

        return total_loss / (rollout_len - 1)

    def _compute_consistency_loss(self, s_b, a_b):
        """一致性损失：theta 预测应与 theta_dot 积分一致。"""
        pred = self._model(s_b, a_b)
        # theta_dot 的导数应该是 theta 的导数
        theta_idx = 3      # theta 在 7D 中的索引
        theta_dot_idx = 4  # theta_dot 在 7D 中的索引

        # theta 的预测变化
        dtheta = pred[:, theta_idx]
        # theta_dot * dt 应该约等于 dtheta
        theta_dot = s_b[:, theta_dot_idx]
        expected_dtheta = theta_dot * DT

        return nn.MSELoss()(dtheta, expected_dtheta)

    def _compute_jacobian_loss(self, s_b, a_b):
        """Jacobian 正则化：平滑动力学函数。"""
        pred = self._model(s_b, a_b)
        # 计算 Jacobian
        jac = torch.zeros(s_b.shape[0], STATE_DIM, STATE_DIM, device=DEVICE)
        for j in range(STATE_DIM):
            grad = torch.autograd.grad(
                pred[:, j].sum(), s_b, create_graph=True
            )[0]
            jac[:, j, :] = grad

        # 惩罚 Jacobian 的范数
        jac_norm = torch.norm(jac, dim=(1, 2))
        return jac_norm.mean()

    def predict(self, s, tau):
        s_norm = torch.FloatTensor(s / self._state_std).unsqueeze(0).to(DEVICE)
        a_norm = torch.FloatTensor([tau / self._action_std]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dsdt_norm = self._model(s_norm, a_norm).cpu().numpy()[0]
        return s + dsdt_norm * self._delta_std * DT

    def save(self, path):
        torch.save({
            'model_state': self._model.state_dict(),
            'config': {
                'hidden': 64, 'depth': 3, 'activation': 'tanh',
                'dt': DT, 'state_dim': STATE_DIM,
            },
            'state_std': self._state_std,
            'action_std': self._action_std,
            'delta_std': self._delta_std,
        }, path)
        print(f"  Model saved to {path}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 80)
    print("方向 2: 课程学习训练 Neural ODE")
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
    # 训练配置
    # ============================================================
    configs = [
        {
            'name': 'V9-style (rollout 1,5,10,20)',
            'n_epochs': 200,
            'lr': 1e-3,
            'lambda_multi': 0.3,
            'lambda_consistency': 0.1,
            'lambda_jacobian': 0.01,
            'curriculum_schedule': '1,5,10,20',
        },
        {
            'name': 'Extended (rollout 1,5,10,20,50)',
            'n_epochs': 300,
            'lr': 1e-3,
            'lambda_multi': 0.3,
            'lambda_consistency': 0.1,
            'lambda_jacobian': 0.01,
            'curriculum_schedule': '1,5,10,20,50',
        },
        {
            'name': 'No curriculum (single-step only)',
            'n_epochs': 200,
            'lr': 1e-3,
            'lambda_multi': 0.0,
            'lambda_consistency': 0.0,
            'lambda_jacobian': 0.0,
            'curriculum_schedule': '1',
        },
    ]

    # ============================================================
    # 训练和评估
    # ============================================================
    all_results = {}

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Training: {cfg['name']}")
        print(f"{'='*60}")

        t0 = time.time()
        trainer = CurriculumTrainer(state_std, action_std, delta_std)
        trainer.train(
            train_s, train_a, train_d,
            n_epochs=cfg['n_epochs'],
            lr=cfg['lr'],
            lambda_multi=cfg['lambda_multi'],
            lambda_consistency=cfg['lambda_consistency'],
            lambda_jacobian=cfg['lambda_jacobian'],
            curriculum_schedule=cfg['curriculum_schedule'],
        )
        print(f"  Training done in {time.time()-t0:.1f}s")

        print(f"Evaluating {cfg['name']}...")
        results = multi_step_evaluate(trainer, segments, state_std, ROLLOUT_HORIZONS)
        all_results[cfg['name']] = results

        # 保存模型
        log_dir = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/checkpoints')
        log_dir.mkdir(exist_ok=True)
        safe_name = cfg['name'].replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
        trainer.save(log_dir / f'node_curriculum_{safe_name}.pt')

    # 加载 V9 参考
    v9_ref = {
        1: 0.0051, 5: 0.0170, 10: 0.0628, 20: 0.1817,
        50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529, 1000: 0.6443,
    }

    # ============================================================
    # 输出对比表
    # ============================================================
    print("\n" + "=" * 100)
    print("综合对比: 课程学习 vs 单步训练")
    print("=" * 100)

    header = f"{'H':>5} | {'V9(ref)':>10}"
    for cfg in configs:
        header += f" | {cfg['name'][:15]:>15}"
    print(header)
    print("-" * (5 + 13 * (1 + len(configs))))

    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        row += f" | {v9_ref.get(h, float('nan')):10.4f}"
        for cfg in configs:
            name = cfg['name']
            row += f" | {all_results[name][h]['nmae_mean']:15.4f}"
        print(row)

    # 存活率
    print("\n存活率:")
    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        row += f" | {'100%':>10}"
        for cfg in configs:
            name = cfg['name']
            row += f" | {all_results[name][h]['survival_rate']:14.0%}%"
        print(row)

    # 保存结果
    log_dir = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/logs')
    log_dir.mkdir(exist_ok=True)
    out_path = log_dir / 'curriculum_training_results.json'

    save_results = {}
    for h in ROLLOUT_HORIZONS:
        save_results[str(h)] = {'v9_ref': v9_ref.get(h, None)}
        for cfg in configs:
            name = cfg['name']
            save_results[str(h)][name] = all_results[name][h]['nmae_mean']

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'results': save_results}, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
