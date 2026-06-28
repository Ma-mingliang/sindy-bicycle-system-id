"""评估框架：轨迹回放、RMSE计算、结果输出。"""

import numpy as np
import math
from methods_common import real_step, K_lqr


def make_tau_func(seg_idx: int, n_steps: int = 500):
    """生成LQR+扰动力矩函数。"""
    rng = np.random.RandomState(42 + seg_idx)
    disturbances = rng.uniform(-0.5, 0.5, n_steps)
    target = np.clip(rng.normal(0, 0.15), -math.pi / 12, math.pi / 12)

    def tau_func(step_i: int, s: np.ndarray) -> float:
        x_lqr = np.array([s[0] - target, s[2], s[1], s[3]])
        return float(-K_lqr @ x_lqr) + disturbances[step_i]

    return tau_func


def run_trajectory(s0: np.ndarray, n_steps: int, tau_func, methods: list) -> dict:
    """运行一段轨迹，返回真实模型和各方法的轨迹。

    Returns:
        dict: {'real': (n+1, 4), 'method_name': (n+1, 4), ...}
    """
    states = {m.name: s0.copy() for m in methods}
    states['real'] = s0.copy()
    trajs = {k: [v.copy()] for k, v in states.items()}

    for i in range(n_steps):
        tau = tau_func(i, states['real'])
        states['real'] = real_step(states['real'], tau)

        if abs(states['real'][0]) > math.pi / 3:
            for key in trajs:
                trajs[key].extend([np.full(4, np.nan)] * (n_steps - i))
            break

        for m in methods:
            states[m.name] = m.predict(states[m.name], tau)

        for key in states:
            trajs[key].append(states[key].copy())

    return {k: np.array(v) for k, v in trajs.items()}


def compute_rmse_at_steps(trajs_list: list, eval_steps: list, method_names: list) -> dict:
    """在指定步数计算各方法vs真实的RMSE。

    Returns:
        dict: {step: {method_name: rmse_value}}
    """
    results = {}
    for step in eval_steps:
        results[step] = {}
        for name in method_names:
            sq = []
            for trajs in trajs_list:
                if step >= len(trajs['real']):
                    continue
                s_real = trajs['real'][step]
                s_model = trajs[name][step]
                if np.any(np.isnan(s_real)) or np.any(np.isnan(s_model)):
                    continue
                sq.append((s_model[0] - s_real[0]) ** 2)
            results[step][name] = math.sqrt(np.mean(sq)) if sq else float('nan')
    return results


def print_results_table(rmse_results: dict, eval_steps: list, method_names: list):
    """格式化输出RMSE结果表。"""
    # 表头
    header = f"  {'步数':>6}"
    for name in method_names:
        short = name.split('_', 1)[1] if '_' in name else name
        header += f" | {short:>10}"
    print(header)
    print("  " + "-" * (8 + 13 * len(method_names)))

    # 数据行
    for step in eval_steps:
        row = f"  {step:6d}"
        for name in method_names:
            v = rmse_results.get(step, {}).get(name, float('nan'))
            row += f" | {_fmt_rmse(v):>10}"
        print(row)

    # 排名（按最后一个评估步数）
    last_step = eval_steps[-1]
    print(f"\n  排名（按{last_step}步RMSE）:")
    sorted_methods = sorted(
        [(name, rmse_results.get(last_step, {}).get(name, float('inf'))) for name in method_names],
        key=lambda x: x[1]
    )
    for rank, (name, rmse) in enumerate(sorted_methods, 1):
        marker = " ★" if rank == 1 else ""
        print(f"    {rank:2d}. {name:<25s} {_fmt_rmse(rmse):>10}{marker}")


def _fmt_rmse(v: float) -> str:
    if math.isnan(v):
        return "N/A"
    if v >= 1e6:
        return f"{v:.1e}"
    if v >= 1:
        return f"{v:.3f}"
    return f"{v:.5f}"
