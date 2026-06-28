"""图表生成：评估结果可视化。"""

import numpy as np
import math
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_error_vs_time(summary: dict, config: dict, output_dir: Path):
    """绘制误差随时间变化曲线。"""
    horizons = config['evaluation']['horizons']
    models = list(summary.keys())
    modes = ['A', 'B', 'C', 'D']
    mode_labels = {'A': 'Teacher Forcing', 'B': 'Open-loop', 'C': 'Hybrid', 'D': 'Closed-loop'}

    for mode in modes:
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_name in models:
            mae_values = []
            for h in horizons:
                h_data = summary[model_name].get(mode, {}).get(str(h), {})
                mae_values.append(h_data.get('mae_mean', float('nan')))
            ax.plot(horizons, mae_values, 'o-', label=model_name, markersize=4)

        ax.set_xlabel('Horizon (steps)')
        ax.set_ylabel('MAE (rad)')
        ax.set_title(f'Error vs Horizon — Mode {mode}: {mode_labels[mode]}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fpath = output_dir / f'error_vs_time_mode_{mode}.png'
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  保存: {fpath}")


def plot_state_trajectories(eval_result: dict, config: dict, output_dir: Path,
                            title_suffix: str = ''):
    """绘制状态轨迹对比图。"""
    states_model = eval_result.get('states_model')
    states_real = eval_result.get('states_real')
    if states_model is None or states_real is None:
        return

    state_names = config['evaluation']['states']
    state_units = config['evaluation']['state_units']
    state_degrees = config['evaluation']['state_degrees']
    dt = config['evaluation']['dt']

    n_steps = min(len(states_model), len(states_real))
    time_axis = np.arange(n_steps) * dt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (name, unit, deg) in enumerate(zip(state_names, state_units, state_degrees)):
        ax = axes[i]
        model_vals = states_model[:n_steps, i]
        real_vals = states_real[:n_steps, i]

        if deg:
            model_vals = np.degrees(model_vals)
            real_vals = np.degrees(real_vals)
            unit = 'deg'

        ax.plot(time_axis, real_vals, 'b-', label='Real', linewidth=1.5)
        ax.plot(time_axis, model_vals, 'r--', label='Model', linewidth=1.0)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{name} ({unit})')
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'State Trajectories {title_suffix}', fontsize=14)
    fig.tight_layout()
    return fig


def plot_model_comparison(summary: dict, config: dict, output_dir: Path):
    """绘制模型对比柱状图。"""
    horizons = config['evaluation']['horizons']
    models = list(summary.keys())
    modes = ['A', 'B', 'C', 'D']
    mode_labels = {'A': 'Teacher Forcing', 'B': 'Open-loop', 'C': 'Hybrid', 'D': 'Closed-loop'}

    for horizon in horizons:
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        for idx, mode in enumerate(modes):
            ax = axes[idx]
            mae_values = []
            model_labels = []
            for model_name in models:
                h_data = summary[model_name].get(mode, {}).get(str(horizon), {})
                mae = h_data.get('mae_mean', float('nan'))
                if not math.isnan(mae):
                    mae_values.append(mae)
                    model_labels.append(model_name.replace('_', '\n'))

            if mae_values:
                bars = ax.bar(range(len(mae_values)), mae_values)
                ax.set_xticks(range(len(model_labels)))
                ax.set_xticklabels(model_labels, fontsize=7, rotation=45, ha='right')
                ax.set_ylabel('MAE (rad)')
                ax.set_title(f'Mode {mode}')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'Model Comparison — {horizon} steps', fontsize=14)
        fig.tight_layout()
        fpath = output_dir / f'model_comparison_{horizon}steps.png'
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  保存: {fpath}")


def plot_state_metric_heatmap(summary: dict, config: dict, output_dir: Path):
    """绘制逐状态指标热力图。"""
    # 只取 Mode C 500步的结果
    models = list(summary.keys())
    state_names = ['phi', 'delta', 'phi_dot', 'delta_dot']

    data = []
    for model_name in models:
        row = []
        # 需要从per-seed结果中获取逐状态指标
        # 这里用overall MAE作为占位
        mode_c = summary[model_name].get('C', {}).get('500', {})
        row.append(mode_c.get('mae_mean', float('nan')))
        data.append(row)

    fig, ax = plt.subplots(figsize=(8, 6))
    data_arr = np.array(data)
    im = ax.imshow(data_arr.reshape(-1, 1), cmap='YlOrRd', aspect='auto')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels(['MAE (Mode C, 500 steps)'])
    fig.colorbar(im)
    fig.tight_layout()

    fpath = output_dir / 'state_metric_heatmap.png'
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  保存: {fpath}")


def generate_all_plots(results: dict, summary: dict, config: dict):
    """生成所有图表。"""
    output_dir = Path(__file__).parent.parent / config['output']['plots_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    print("生成图表...")

    # 1. 误差vs时间
    plot_error_vs_time(summary, config, output_dir)

    # 2. 模型对比
    plot_model_comparison(summary, config, output_dir)

    # 3. 状态指标热力图
    plot_state_metric_heatmap(summary, config, output_dir)

    print("图表生成完成")
