"""诊断 GP 预测行为：分析 GP 预测的 delta 大小、方向、与真实 delta 的关系。"""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import json
from pathlib import Path
from canonical_7d.data_loader import load_7d_data, get_test_segments
from canonical_7d.config import DataConfig
from canonical_node.config_v9 import STATE_NAMES_7D, STATE_DIM

SEED = 42

# ============================================================
# 加载训练好的 GP（重新训练，因为没有持久化）
# ============================================================
def load_and_train_gp():
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

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

    # 训练 GP
    max_samples = 2000
    rng = np.random.RandomState(SEED)
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

    return gps, data, x_mean, x_std, state_std, action_std, delta_std


def gp_predict(gps, x_mean, x_std, state_std, action_std, delta_std, s, tau):
    s_norm = s / state_std
    a_norm = tau / action_std
    x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
    x_scaled = (x - x_mean) / x_std
    delta_norm = np.array([gp.predict(x_scaled)[0] for gp in gps])
    return s + delta_norm * delta_std, delta_norm * delta_std


def gp_predict_with_uncertainty(gps, x_mean, x_std, state_std, action_std, delta_std, s, tau):
    s_norm = s / state_std
    a_norm = tau / action_std
    x = np.concatenate([s_norm, [a_norm]]).reshape(1, -1)
    x_scaled = (x - x_mean) / x_std
    means = []
    stds = []
    for gp in gps:
        mean, std = gp.predict(x_scaled, return_std=True)
        means.append(mean[0])
        stds.append(std[0])
    delta_norm = np.array(means)
    return (s + delta_norm * delta_std,
            delta_norm * delta_std,
            np.array(stds) * delta_std)


def main():
    print("Loading GP and data...")
    gps, data, x_mean, x_std, state_std, action_std, delta_std = load_and_train_gp()
    print("Done.\n")

    # 获取测试段
    segments = get_test_segments(data, n_segments=5, segment_length=1100, seed=SEED)

    # ============================================================
    # 分析 1: GP 预测的 delta 大小 vs 真实 delta
    # ============================================================
    print("=" * 80)
    print("分析 1: GP 预测 delta vs 真实 delta (单步)")
    print("=" * 80)

    test_s = data['test_states']
    test_a = data['test_actions']
    test_d = data['test_deltas']

    # 采样 500 个测试点
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(test_s), 500, replace=False)

    gp_deltas = []
    real_deltas = []
    gp_uncertainties = []

    for i in sample_idx:
        _, delta_gp, unc = gp_predict_with_uncertainty(
            gps, x_mean, x_std, state_std, action_std, delta_std,
            test_s[i], test_a[i],
        )
        gp_deltas.append(delta_gp)
        real_deltas.append(test_d[i])
        gp_uncertainties.append(unc)

    gp_deltas = np.array(gp_deltas)
    real_deltas = np.array(real_deltas)
    gp_uncertainties = np.array(gp_uncertainties)

    print(f"\n{'状态':>12} | {'真实|delta|均值':>14} | {'GP|delta|均值':>14} | {'比值':>8} | {'符号一致率':>10} | {'GP unc均值':>12}")
    print("-" * 80)
    for j, name in enumerate(STATE_NAMES_7D):
        real_mean = np.mean(np.abs(real_deltas[:, j]))
        gp_mean = np.mean(np.abs(gp_deltas[:, j]))
        ratio = gp_mean / real_mean if real_mean > 1e-10 else float('nan')
        sign_match = np.mean(np.sign(gp_deltas[:, j]) == np.sign(real_deltas[:, j]))
        unc_mean = np.mean(np.abs(gp_uncertainties[:, j]))
        print(f"  {name:>12} | {real_mean:14.6f} | {gp_mean:14.6f} | {ratio:8.4f} | {sign_match:9.1%} | {unc_mean:12.6f}")

    # ============================================================
    # 分析 2: 多步 rollout 中 GP 预测的 delta 如何变化
    # ============================================================
    print("\n" + "=" * 80)
    print("分析 2: 多步 rollout 中 GP 预测 delta 的变化")
    print("=" * 80)

    seg = segments[0]
    s0 = seg['states'][0].copy()
    actions_seg = seg['actions']
    real_states = seg['states']

    # 做 500 步 rollout
    s_cur = s0.copy()
    gp_trajectory = [s0.copy()]
    delta_norms_over_time = []
    uncertainty_over_time = []
    state_deviation = []  # 与真实轨迹的偏差

    for step in range(min(500, len(actions_seg))):
        tau = actions_seg[step]
        s_next, delta_gp, unc = gp_predict_with_uncertainty(
            gps, x_mean, x_std, state_std, action_std, delta_std,
            s_cur, tau,
        )
        delta_norms_over_time.append(np.linalg.norm(delta_gp))
        uncertainty_over_time.append(np.mean(np.abs(unc)))

        if step + 1 < len(real_states):
            state_deviation.append(np.linalg.norm(s_next - real_states[step + 1]))

        gp_trajectory.append(s_next.copy())
        s_cur = s_next

    gp_trajectory = np.array(gp_trajectory)
    delta_norms_over_time = np.array(delta_norms_over_time)
    uncertainty_over_time = np.array(uncertainty_over_time)
    state_deviation = np.array(state_deviation)

    # 按 horizon 输出
    print(f"\n{'Step':>6} | {'|delta|均值':>12} | {'GP unc均值':>12} | {'与真实偏差':>12} | {'GP e_y':>10} | {'真实 e_y':>10}")
    print("-" * 75)
    for step in [0, 4, 9, 19, 49, 99, 199, 499]:
        if step < len(delta_norms_over_time):
            d_norm = delta_norms_over_time[step]
            unc = uncertainty_over_time[step]
            dev = state_deviation[step] if step < len(state_deviation) else float('nan')
            gp_ey = gp_trajectory[step + 1, 0] if step + 1 < len(gp_trajectory) else float('nan')
            real_ey = real_states[step + 1, 0] if step + 1 < len(real_states) else float('nan')
            print(f"  {step+1:>6} | {d_norm:12.6f} | {unc:12.6f} | {dev:12.4f} | {gp_ey:10.4f} | {real_ey:10.4f}")

    # ============================================================
    # 分析 3: GP 是否在预测"近恒等映射"？
    # ============================================================
    print("\n" + "=" * 80)
    print("分析 3: GP 预测的 delta 相对于状态本身的大小")
    print("=" * 80)

    # 对于每个状态维度，计算 GP delta / |state| 的比值
    print(f"\n{'Step':>6} | ", end="")
    for name in STATE_NAMES_7D:
        print(f"{name:>10} | ", end="")
    print()
    print("-" * (6 + 13 * 7))

    for step in [0, 4, 9, 19, 49, 99, 199, 499]:
        if step < len(delta_norms_over_time):
            s_cur = gp_trajectory[step + 1] if step + 1 < len(gp_trajectory) else None
            tau = actions_seg[step] if step < len(actions_seg) else None
            if s_cur is not None and tau is not None:
                _, delta_gp, _ = gp_predict_with_uncertainty(
                    gps, x_mean, x_std, state_std, action_std, delta_std,
                    s_cur, tau,
                )
                # delta / |state| 比值
                ratios = np.abs(delta_gp) / (np.abs(s_cur) + 1e-8)
                print(f"  {step+1:>6} | ", end="")
                for r in ratios:
                    print(f"{r:10.4f} | ", end="")
                print()

    # ============================================================
    # 分析 4: GP 在不同 horizon 的 NMAE 贡献来源
    # ============================================================
    print("\n" + "=" * 80)
    print("分析 4: 每个状态在不同 horizon 的 NMAE 贡献")
    print("=" * 80)

    horizons = [1, 5, 10, 20, 50, 100, 200, 500]
    for seg_i, seg in enumerate(segments[:3]):  # 只看前 3 个段
        print(f"\n--- Segment {seg_i} ---")
        s0 = seg['states'][0].copy()
        actions_seg = seg['actions']
        real_states = seg['states']

        for h in horizons:
            s_cur = s0.copy()
            n = min(h, len(actions_seg))
            errors = []
            for step in range(n):
                s_next, _, _ = gp_predict_with_uncertainty(
                    gps, x_mean, x_std, state_std, action_std, delta_std,
                    s_cur, actions_seg[step],
                )
                if step + 1 < len(real_states):
                    err = np.abs(s_next - real_states[step + 1]) / state_std
                    errors.append(err)
                s_cur = s_next
                if np.any(np.isnan(s_next)) or np.any(np.isinf(s_next)):
                    break

            if errors:
                mean_err = np.mean(errors, axis=0)
                print(f"  H={h:>4}: NMAE={np.mean(mean_err):.4f} | ", end="")
                for j, name in enumerate(STATE_NAMES_7D):
                    print(f"{name}={mean_err[j]:.3f} ", end="")
                print()

    # ============================================================
    # 分析 5: GP 的"隐式正则化"效应
    # ============================================================
    print("\n" + "=" * 80)
    print("分析 5: GP 隐式正则化 — 预测 delta 是否随 rollout 趋于零？")
    print("=" * 80)

    seg = segments[0]
    s0 = seg['states'][0].copy()
    actions_seg = seg['actions']

    s_cur = s0.copy()
    delta_magnitudes = []
    for step in range(min(500, len(actions_seg))):
        _, delta_gp, _ = gp_predict_with_uncertainty(
            gps, x_mean, x_std, state_std, action_std, delta_std,
            s_cur, actions_seg[step],
        )
        delta_magnitudes.append(np.linalg.norm(delta_gp))
        s_next = s_cur + delta_gp
        s_cur = s_next

    delta_magnitudes = np.array(delta_magnitudes)
    print(f"\n  按窗口统计 |delta|:")
    for start, end in [(0, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500)]:
        end = min(end, len(delta_magnitudes))
        if start < end:
            window = delta_magnitudes[start:end]
            print(f"    Steps {start:>3}-{end:>3}: mean={np.mean(window):.6f}, std={np.std(window):.6f}, min={np.min(window):.6f}, max={np.max(window):.6f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
