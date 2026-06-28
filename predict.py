"""独立预测脚本：GP+Ensemble模型的训练、保存、加载和预测。

用法:
    # 1. 训练并保存模型
    python predict.py --train --save models/gp_ensemble.npz

    # 2. 加载模型并预测（开环：给定动作序列）
    python predict.py --load models/gp_ensemble.npz --mode open_loop --steps 200

    # 3. 加载模型并预测（闭环：LQR控制）
    python predict.py --load models/gp_ensemble.npz --mode closed_loop --steps 500

    # 4. 单步预测
    python predict.py --load models/gp_ensemble.npz --mode single --state 0.1,0.05,0.1,0.05 --tau 1.0

预测模式说明:
    - single:     单步预测 predict(s, tau) -> s_next
    - open_loop:  开环预测，给定初始状态+随机动作序列，输出轨迹
    - closed_loop: 闭环预测，LQR控制，tau来自模型自身状态
    - hybrid:     混合模式(当前评估方式)，tau来自真实状态，状态来自模型
"""

import sys, os
import argparse
import numpy as np
import math
import json
import time


# ============================================================
# 模型保存/加载
# ============================================================
def save_model(path, gp, nn_models, state_std, action_std, delta_std,
               residual_scale=0.3, config=None):
    """保存GP+Ensemble模型到npz文件。

    保存内容:
        - 4个GP的超参数和训练数据（用于重建）
        - NN残差模型的state_dict
        - 归一化常数
        - 配置信息
    """
    import torch

    data = {
        # 归一化常数
        'state_std': state_std,
        'action_std': np.array([action_std]),
        'delta_std': delta_std,
        'residual_scale': np.array([residual_scale]),
        # GP训练数据（用于重建GP模型）
        'gp_X_train': gp._X_train,
        'gp_Y_train': gp._Y_train,
        'gp_max_samples': np.array([gp.max_samples]),
    }

    # 保存NN模型state_dict
    for i, model in enumerate(nn_models):
        for key, val in model.state_dict().items():
            data[f'nn_{i}_{key}'] = val.numpy()

    # 保存配置
    if config:
        data['config'] = json.dumps(config)

    np.savez_compressed(path, **data)
    print(f"模型已保存到: {path}")
    print(f"  GP训练数据: {gp._X_train.shape}")
    print(f"  NN模型数: {len(nn_models)}")
    print(f"  residual_scale: {residual_scale}")


def load_model(path):
    """从npz文件加载GP+Ensemble模型。

    返回: (gp, nn_models, state_std, action_std, delta_std, residual_scale)
    """
    import torch
    from methods_classic import GPMethod
    from methods_nn import ResidualNet

    data = np.load(path, allow_pickle=True)

    state_std = data['state_std']
    action_std = float(data['action_std'][0])
    delta_std = data['delta_std']
    residual_scale = float(data['residual_scale'][0])

    # 重建GP模型
    gp = GPMethod(max_samples=int(data['gp_max_samples'][0]))
    gp._state_std = state_std
    gp._action_std = action_std
    gp._delta_std = delta_std
    gp._X_train = data['gp_X_train']
    gp._Y_train = data['gp_Y_train']

    # 从训练数据重新fit GP（sklearn GP不能直接序列化，需要refit）
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gp._gps = []
    for col in range(gp._Y_train.shape[1]):
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-6)
        gpr.fit(gp._X_train, gp._Y_train[:, col])
        gp._gps.append(gpr)
    print(f"  GP已重建: {len(gp._gps)}个输出")

    # 重建NN模型
    nn_models = []
    nn_keys = [k for k in data.files if k.startswith('nn_0_')]
    # 检测有多少个NN模型
    model_indices = set()
    for k in data.files:
        if k.startswith('nn_') and '_' in k[3:]:
            idx = k.split('_')[1]
            if idx.isdigit():
                model_indices.add(int(idx))
    n_models = len(model_indices)

    for i in range(n_models):
        model = ResidualNet()
        state_dict = {}
        prefix = f'nn_{i}_'
        for key in data.files:
            if key.startswith(prefix):
                param_name = key[len(prefix):]
                state_dict[param_name] = torch.FloatTensor(data[key])
        model.load_state_dict(state_dict)
        model.eval()
        nn_models.append(model)
    print(f"  NN模型已加载: {n_models}个")

    return gp, nn_models, state_std, action_std, delta_std, residual_scale


# ============================================================
# 预测函数
# ============================================================
def ensemble_predict_single(gp, nn_models, s, tau,
                            state_std, action_std, delta_std, residual_scale):
    """单步集成预测。

    公式: s_next = GP(s, tau) + mean(NN_1..n)(s_norm, a_norm) * delta_std * 0.3

    与 test_ensemble.py:147 的 ensemble_predict() 完全一致。
    """
    import torch

    # GP基线预测
    s_next_gp = gp.predict(s, tau)

    # NN残差预测
    s_norm = s / state_std
    a_norm = tau / action_std
    x = torch.FloatTensor(np.concatenate([s_norm, [a_norm]]))

    nn_residuals = []
    for model in nn_models:
        model.eval()
        with torch.no_grad():
            nn_res = model(x).numpy()
        nn_residuals.append(nn_res)

    mean_residual = np.mean(nn_residuals, axis=0)
    nn_std = np.std(nn_residuals, axis=0)  # 不确定性

    s_next = s_next_gp + mean_residual * delta_std * residual_scale
    return s_next, nn_std


def predict_open_loop(gp, nn_models, s0, actions,
                      state_std, action_std, delta_std, residual_scale):
    """开环预测：给定初始状态和动作序列，预测轨迹。

    这是"正常预测方案"：纯模型递推，无真实状态反馈。

    参数:
        s0: 初始状态 (4,)
        actions: 动作序列 (n_steps,)

    返回:
        states: 预测轨迹 (n_steps+1, 4)
        uncertainties: 不确定性 (n_steps, 4)
    """
    n_steps = len(actions)
    states = np.zeros((n_steps + 1, 4))
    uncertainties = np.zeros((n_steps, 4))
    states[0] = s0.copy()

    for i in range(n_steps):
        s_next, unc = ensemble_predict_single(
            gp, nn_models, states[i], actions[i],
            state_std, action_std, delta_std, residual_scale
        )
        states[i + 1] = s_next
        uncertainties[i] = unc

    return states, uncertainties


def predict_closed_loop(gp, nn_models, s0, n_steps,
                        state_std, action_std, delta_std, residual_scale,
                        seed=42):
    """闭环预测：LQR控制，tau来自模型自身状态。

    这是 Mode D (Full closed-loop)：模型自己控制自己。

    返回:
        states: 预测轨迹 (n_steps+1, 4)
        actions: 实际动作序列 (n_steps,)
        uncertainties: 不确定性 (n_steps, 4)
    """
    from methods_common import K_lqr
    from methods_evaluate import make_tau_func

    tau_func = make_tau_func(0, n_steps)

    states = np.zeros((n_steps + 1, 4))
    actions = np.zeros(n_steps)
    uncertainties = np.zeros((n_steps, 4))
    states[0] = s0.copy()

    rng = np.random.RandomState(seed)
    disturbances = rng.uniform(-0.5, 0.5, n_steps)
    target = np.clip(rng.normal(0, 0.15), -math.pi / 12, math.pi / 12)

    for i in range(n_steps):
        # tau来自模型状态（闭环）
        s = states[i]
        tau = float(-K_lqr @ np.array([s[0] - target, s[2], s[1], s[3]]))
        tau += disturbances[i]
        actions[i] = tau

        s_next, unc = ensemble_predict_single(
            gp, nn_models, s, tau,
            state_std, action_std, delta_std, residual_scale
        )
        states[i + 1] = s_next
        uncertainties[i] = unc

    return states, actions, uncertainties


def predict_hybrid(gp, nn_models, s0, n_steps,
                   state_std, action_std, delta_std, residual_scale,
                   seed=42):
    """混合预测：tau来自真实状态，状态来自模型。

    这是当前评估方式 Mode C (Hybrid)。

    返回:
        states_model: 模型预测轨迹 (n_steps+1, 4)
        states_real: 真实轨迹 (n_steps+1, 4)
        actions: 动作序列 (n_steps,)
        errors: 每步误差 (n_steps, 4)
    """
    from methods_common import real_step
    from methods_evaluate import make_tau_func

    tau_func = make_tau_func(0, n_steps)

    states_model = np.zeros((n_steps + 1, 4))
    states_real = np.zeros((n_steps + 1, 4))
    actions = np.zeros(n_steps)
    errors = np.zeros((n_steps, 4))

    states_model[0] = s0.copy()
    states_real[0] = s0.copy()

    for i in range(n_steps):
        # tau来自真实状态（混合模式）
        tau = tau_func(i, states_real[i])
        actions[i] = tau

        # 真实动力学
        states_real[i + 1] = real_step(states_real[i], tau)

        # 模型预测
        s_next, _ = ensemble_predict_single(
            gp, nn_models, states_model[i], tau,
            state_std, action_std, delta_std, residual_scale
        )
        states_model[i + 1] = s_next

        # 误差
        errors[i] = np.abs(states_model[i + 1] - states_real[i + 1])

    return states_model, states_real, actions, errors


# ============================================================
# 训练函数
# ============================================================
def train_and_save(save_path, n_models=5, dagger_rounds=3, residual_scale=0.3):
    """训练GP+Ensemble模型并保存。

    与 test_ensemble.py:train_ensemble() 流程一致。
    """
    from methods_common import generate_training_data, real_step
    from methods_classic import GPMethod
    from methods_nn import ResidualNet
    from methods_evaluate import make_tau_func
    import torch

    print("=" * 60)
    print("训练 GP+Ensemble 模型")
    print("=" * 60)

    # 1. 生成训练数据
    print("\n1. 生成训练数据...")
    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(30000, seed=42)
    print(f"   states: {states.shape}, state_std: {state_std}")

    # 2. 训练GP
    print("\n2. 训练GP基线...")
    t0 = time.time()
    gp = GPMethod()
    gp.train(states, actions, deltas, state_std, action_std, delta_std)
    print(f"   GP训练完成: {time.time()-t0:.1f}s")

    # 保存GP训练数据（用于重建）
    gp._X_train = np.column_stack([states / state_std, actions.reshape(-1, 1) / action_std])
    gp._Y_train = deltas / delta_std

    # 3. 计算GP残差
    print("\n3. 计算GP残差...")
    gp_deltas = []
    for i in range(len(states)):
        s_next_gp = gp.predict(states[i], actions[i])
        gp_deltas.append(s_next_gp - states[i])
    gp_deltas = np.array(gp_deltas)
    residuals = (deltas - gp_deltas) / delta_std

    # 4. 训练NN集成
    print(f"\n4. 训练NN集成 ({n_models}个模型)...")
    nn_models = []
    for i in range(n_models):
        torch.manual_seed(i * 42)
        np.random.seed(i * 42)
        model = ResidualNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        X = torch.FloatTensor(np.column_stack([states / state_std, actions / action_std]))
        Y = torch.FloatTensor(residuals)

        model.train()
        for epoch in range(100):
            idx = np.random.permutation(len(X))
            for batch_start in range(0, len(X), 256):
                batch_idx = idx[batch_start:batch_start + 256]
                optimizer.zero_grad()
                pred = model(X[batch_idx])
                loss = torch.nn.functional.mse_loss(pred, Y[batch_idx])
                loss.backward()
                optimizer.step()

        nn_models.append(model)
        print(f"   模型 {i} 训练完成")

    # 5. DAgger
    for round_i in range(dagger_rounds):
        print(f"\n5. DAgger轮次 {round_i+1}/{dagger_rounds}...")
        dagger_states, dagger_actions, dagger_deltas = [], [], []

        for seg_i in range(3):
            tau_func = make_tau_func(100 + round_i * 3 + seg_i, 500)
            phi_init = np.random.uniform(-0.25, 0.25)
            s = np.array([phi_init, 0.0, 0.0, 0.0])

            for step in range(500):
                tau = tau_func(step, s)
                s_real = real_step(s, tau)
                dagger_states.append(s.copy())
                dagger_actions.append(tau)
                dagger_deltas.append(s_real - s)
                s = s_real
                if abs(s[0]) > math.pi / 3:
                    break

        dagger_states = np.array(dagger_states)
        dagger_actions = np.array(dagger_actions)
        dagger_deltas = np.array(dagger_deltas)

        # GP残差
        gp_deltas_d = []
        for i in range(len(dagger_states)):
            s_next_gp = gp.predict(dagger_states[i], dagger_actions[i])
            gp_deltas_d.append(s_next_gp - dagger_states[i])
        residuals_d = (dagger_deltas - np.array(gp_deltas_d)) / delta_std

        # 合并数据
        X_aug = torch.FloatTensor(np.column_stack([
            np.vstack([states, dagger_states]) / state_std,
            np.concatenate([actions, dagger_actions]) / action_std
        ]))
        Y_aug = torch.FloatTensor(np.vstack([residuals, residuals_d]))

        # 重训练
        for i, model in enumerate(nn_models):
            torch.manual_seed(i * 42 + round_i * 1000)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model.train()
            for epoch in range(50):
                idx = np.random.permutation(len(X_aug))
                for batch_start in range(0, len(X_aug), 256):
                    batch_idx = idx[batch_start:batch_start + 256]
                    optimizer.zero_grad()
                    pred = model(X_aug[batch_idx])
                    loss = torch.nn.functional.mse_loss(pred, Y_aug[batch_idx])
                    loss.backward()
                    optimizer.step()

    # 6. 保存
    print(f"\n6. 保存模型...")
    config = {
        'n_models': n_models,
        'dagger_rounds': dagger_rounds,
        'residual_scale': residual_scale,
        'n_training_samples': 30000,
        'training_seed': 42,
    }
    save_model(save_path, gp, nn_models, state_std, action_std, delta_std,
               residual_scale, config)

    return gp, nn_models, state_std, action_std, delta_std, residual_scale


# ============================================================
# 可视化
# ============================================================
def plot_trajectory(states, states_real=None, actions=None, title="Prediction"):
    """绘制预测轨迹。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib未安装，跳过绘图")
        return

    n = len(states) - 1
    t = np.arange(n + 1) / 30.0  # 30Hz

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # phi (横滚角)
    axes[0].plot(t, np.degrees(states[:, 0]), 'b-', label='Model', linewidth=1.5)
    if states_real is not None:
        axes[0].plot(t, np.degrees(states_real[:, 0]), 'r--', label='Real', linewidth=1.5)
    axes[0].set_ylabel('phi (deg)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)

    # delta (转向角)
    axes[1].plot(t, np.degrees(states[:, 1]), 'b-', label='Model', linewidth=1.5)
    if states_real is not None:
        axes[1].plot(t, np.degrees(states_real[:, 1]), 'r--', label='Real', linewidth=1.5)
    axes[1].set_ylabel('delta (deg)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # actions
    if actions is not None:
        axes[2].plot(t[:-1], actions, 'g-', linewidth=1)
        axes[2].set_ylabel('tau (N*m)')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].set_ylabel('tau (N*m)')

    axes[2].set_xlabel('Time (s)')
    plt.tight_layout()

    out_path = title.replace(' ', '_') + '.png'
    plt.savefig(out_path, dpi=150)
    print(f"图已保存: {out_path}")
    plt.close()


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='GP+Ensemble 预测脚本')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--save', type=str, help='保存路径')
    parser.add_argument('--load', type=str, help='加载路径')
    parser.add_argument('--mode', type=str, default='open_loop',
                        choices=['single', 'open_loop', 'closed_loop', 'hybrid'],
                        help='预测模式')
    parser.add_argument('--steps', type=int, default=200, help='预测步数')
    parser.add_argument('--state', type=str, default='0.1,0.05,0.1,0.05',
                        help='初始状态 (逗号分隔)')
    parser.add_argument('--tau', type=float, default=1.0, help='单步预测的力矩')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--n_models', type=int, default=5, help='NN模型数量')
    parser.add_argument('--dagger_rounds', type=int, default=3, help='DAgger轮数')
    parser.add_argument('--residual_scale', type=float, default=0.3, help='残差缩放')
    parser.add_argument('--plot', action='store_true', help='绘制轨迹')

    args = parser.parse_args()

    # 训练或加载
    if args.train:
        save_path = args.save or 'models/gp_ensemble.npz'
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        gp, nn_models, state_std, action_std, delta_std, residual_scale = \
            train_and_save(save_path, args.n_models, args.dagger_rounds, args.residual_scale)
    elif args.load:
        gp, nn_models, state_std, action_std, delta_std, residual_scale = load_model(args.load)
    else:
        print("错误: 需要 --train 或 --load")
        return

    # 预测
    print(f"\n{'='*60}")
    print(f"预测模式: {args.mode}")
    print(f"{'='*60}")

    if args.mode == 'single':
        s = np.array([float(x) for x in args.state.split(',')])
        tau = args.tau
        s_next, unc = ensemble_predict_single(
            gp, nn_models, s, tau,
            state_std, action_std, delta_std, residual_scale
        )
        print(f"\n输入状态: {s}")
        print(f"输入力矩: {tau}")
        print(f"预测下一状态: {s_next}")
        print(f"不确定性(std): {unc}")
        print(f"状态变化: {s_next - s}")

    elif args.mode == 'open_loop':
        s0 = np.array([float(x) for x in args.state.split(',')])
        rng = np.random.RandomState(args.seed)
        actions = rng.uniform(-5, 5, args.steps)

        print(f"\n初始状态: {s0}")
        print(f"动作序列: {args.steps}步, 范围[-5, 5]")

        t0 = time.time()
        states, uncertainties = predict_open_loop(
            gp, nn_models, s0, actions,
            state_std, action_std, delta_std, residual_scale
        )
        print(f"预测耗时: {time.time()-t0:.2f}s")

        # 统计
        phi_max = np.max(np.abs(states[:, 0]))
        print(f"\n结果:")
        print(f"  phi最大值: {np.degrees(phi_max):.2f}度")
        print(f"  终态: {states[-1]}")
        print(f"  平均不确定性: {np.mean(uncertainties, axis=0)}")

        if phi_max > math.pi / 3:
            print(f"  警告: phi超过60度，模型可能已发散")

        if args.plot:
            plot_trajectory(states, actions=actions, title="Open Loop Prediction")

    elif args.mode == 'closed_loop':
        s0 = np.array([float(x) for x in args.state.split(',')])

        print(f"\n初始状态: {s0}")
        print(f"控制: LQR, {args.steps}步")

        t0 = time.time()
        states, actions, uncertainties = predict_closed_loop(
            gp, nn_models, s0, args.steps,
            state_std, action_std, delta_std, residual_scale,
            seed=args.seed
        )
        print(f"预测耗时: {time.time()-t0:.2f}s")

        phi_max = np.max(np.abs(states[:, 0]))
        print(f"\n结果:")
        print(f"  phi最大值: {np.degrees(phi_max):.2f}度")
        print(f"  终态: {states[-1]}")

        if args.plot:
            plot_trajectory(states, actions=actions, title="Closed Loop Prediction")

    elif args.mode == 'hybrid':
        s0 = np.array([float(x) for x in args.state.split(',')])

        print(f"\n初始状态: {s0}")
        print(f"模式: 混合(tau来自真实，状态来自模型)")

        from methods_common import real_step
        from methods_evaluate import make_tau_func

        states_model, states_real, actions, errors = predict_hybrid(
            gp, nn_models, s0, args.steps,
            state_std, action_std, delta_std, residual_scale,
            seed=args.seed
        )

        # 统计
        phi_errors = errors[:, 0]
        print(f"\n结果:")
        print(f"  phi MAE: {np.mean(phi_errors):.6f} rad ({np.degrees(np.mean(phi_errors)):.4f} deg)")
        print(f"  phi Max Error: {np.max(phi_errors):.6f} rad")
        print(f"  终态误差: {np.abs(states_model[-1] - states_real[-1])}")

        if args.plot:
            plot_trajectory(states_model, states_real=states_real, actions=actions,
                            title="Hybrid Prediction (Model vs Real)")


if __name__ == '__main__':
    main()
