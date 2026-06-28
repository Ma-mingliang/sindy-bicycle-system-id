"""方案 4: Sparse GP + 更多数据。

用更多训练样本（5000-10000）训练 GP，
看是否能改善 GP 的短期精度。

用法:
    python test_sparse_gp.py
"""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')
sys.stdout.reconfigure(line_buffering=True)

import json
import time
import warnings
import numpy as np
from pathlib import Path

from canonical_7d.data_loader import load_7d_data, get_test_segments
from canonical_7d.config import DataConfig
from canonical_node.evaluation_v9 import multi_step_evaluate
from canonical_node.config_v9 import (
    STATE_NAMES_7D, STATE_DIM, ROLLOUT_HORIZONS, N_EVAL_SEGMENTS, SEGMENT_LENGTH,
)

SEED = 42


class ScalableGP:
    """可扩展的 GP：支持不同样本量。"""

    def __init__(self, max_samples=2000):
        self._max_samples = max_samples
        self._gps = None
        self._x_mean = None
        self._x_std = None
        self._state_std = None
        self._action_std = None
        self._delta_std = None

    def train(self, states, actions, deltas, state_std, action_std, delta_std):
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        self._state_std = state_std
        self._action_std = action_std
        self._delta_std = delta_std

        n = len(states)
        if n > self._max_samples:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(n, self._max_samples, replace=False)
        else:
            idx = np.arange(n)

        X = np.column_stack([
            states[idx] / state_std,
            actions[idx].reshape(-1, 1) / action_std,
        ])
        Y = deltas[idx] / delta_std

        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self._x_mean) / self._x_std

        self._gps = []
        kernel = Matern(nu=2.5, length_scale=1.0)
        for col in range(STATE_DIM):
            t0 = time.time()
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-3)
            gp.fit(X_scaled, Y[:, col])
            self._gps.append(gp)
            print(f"    GP dim {col} ({STATE_NAMES_7D[col]}) done ({time.time()-t0:.1f}s)")

    def predict(self, s, tau):
        s_norm = s / self._state_std
        a_norm = tau / self._action_std
        x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
        x_scaled = (x - self._x_mean) / self._x_std
        delta_norm = np.array([gp.predict(x_scaled)[0] for gp in self._gps])
        return s + delta_norm * self._delta_std


def main():
    print("=" * 80)
    print("方案 4: Sparse GP + 更多数据")
    print("=" * 80)

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

    # 测试不同样本量
    sample_sizes = [1000, 2000, 5000, 10000]
    all_results = {}

    for n_samples in sample_sizes:
        print(f"\n{'='*60}")
        print(f"Training GP with {n_samples} samples...")
        print(f"{'='*60}")
        t0 = time.time()

        gp = ScalableGP(max_samples=n_samples)
        gp.train(train_s, train_a, train_d, state_std, action_std, delta_std)
        print(f"  Training done in {time.time()-t0:.1f}s")

        print(f"Evaluating GP (n={n_samples})...")
        results = multi_step_evaluate(gp, segments, state_std, ROLLOUT_HORIZONS)
        all_results[n_samples] = results

    # V9 参考
    v9_ref = {
        1: 0.0051, 5: 0.0170, 10: 0.0628, 20: 0.1817,
        50: 0.4557, 100: 0.5064, 200: 0.4737, 500: 0.5529, 1000: 0.6443,
    }

    # 输出对比表
    print("\n" + "=" * 100)
    print("综合对比: 不同样本量 GP")
    print("=" * 100)

    header = f"{'H':>5} | {'V9(ref)':>10}"
    for n in sample_sizes:
        header += f" | {'GP-'+str(n):>10}"
    print(header)
    print("-" * (5 + 13 * (1 + len(sample_sizes))))

    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        row += f" | {v9_ref.get(h, float('nan')):10.4f}"
        for n in sample_sizes:
            row += f" | {all_results[n][h]['nmae_mean']:10.4f}"
        print(row)

    # 存活率
    print("\n存活率:")
    for h in ROLLOUT_HORIZONS:
        row = f"{h:>5}"
        row += f" | {'100%':>10}"
        for n in sample_sizes:
            row += f" | {all_results[n][h]['survival_rate']:9.0%}%"
        print(row)

    # 保存
    log_dir = Path('D:/系统辨识作业/sindy_bicycle/continuation_stage_v9/logs')
    log_dir.mkdir(exist_ok=True)
    out_path = log_dir / 'sparse_gp_results.json'

    save_results = {}
    for h in ROLLOUT_HORIZONS:
        save_results[str(h)] = {'v9_ref': v9_ref.get(h, None)}
        for n in sample_sizes:
            save_results[str(h)][f'gp_{n}'] = all_results[n][h]['nmae_mean']
            save_results[str(h)][f'gp_{n}_survival'] = all_results[n][h]['survival_rate']

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'results': save_results, 'sample_sizes': sample_sizes}, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
