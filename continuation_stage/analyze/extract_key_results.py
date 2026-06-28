"""从 full_evaluation_v2.json 提取关键结果并生成汇总。

用法:
    python continuation_stage/analyze/extract_key_results.py
"""

import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_PATH = PROJECT_ROOT / 'continuation_stage' / 'results' / 'full_evaluation_v2.json'
OUTPUT_PATH = PROJECT_ROOT / 'continuation_stage' / 'results' / 'key_results_summary.md'


def main():
    if not RESULTS_PATH.exists():
        print(f"结果文件不存在: {RESULTS_PATH}")
        print("请先运行 run_full_evaluation_v2.py")
        return

    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    config = data['config']
    seeds = config['seeds']
    horizons = config['horizons']
    models = config['models']
    modes = config['modes']
    results = data['data']

    lines = [
        "# 评估v2 关键结果汇总",
        "",
        "> **生成时间**: 2026-06-25",
        "> **配置**: " + f"{len(seeds)} seeds, {len(horizons)} horizons, {len(models)} models, {len(modes)} modes",
        "",
        "---",
        "",
    ]

    # 每个模式的结果表
    for mode in modes:
        lines.append(f"## Mode {mode}")
        lines.append("")

        # 表头
        header = "| Model |"
        separator = "|-------|"
        for h in horizons:
            header += f" {h}步 |"
            separator += "------|"
        lines.append(header)
        lines.append(separator)

        # 每个模型
        for model_name in models:
            row = f"| {model_name} |"
            for h in horizons:
                mae_values = []
                for seed in seeds:
                    entry = results.get(str(seed), {}).get(str(h), {}).get(model_name, {}).get(mode, {})
                    mae = entry.get('metrics', {}).get('overall', {}).get('mae')
                    if mae is not None and not np.isnan(mae) and mae < 1e6:
                        mae_values.append(mae)
                if mae_values:
                    mean_mae = np.mean(mae_values)
                    std_mae = np.std(mae_values) if len(mae_values) > 1 else 0
                    if mean_mae > 10:
                        row += f" {mean_mae:.1f}±{std_mae:.1f} |"
                    else:
                        row += f" {mean_mae:.3f}±{std_mae:.3f} |"
                else:
                    row += " N/A |"
            lines.append(row)

        lines.append("")

    # 逐状态详细结果 (Mode B, 最大时域)
    lines.append("## 逐状态详细结果 (Mode B, 最大时域)")
    lines.append("")
    max_h = max(horizons)
    lines.append(f"| Model | phi | delta | phi_dot | delta_dot |")
    lines.append("|-------|-----|-------|---------|-----------|")

    for model_name in models:
        row = f"| {model_name} |"
        for state in ['phi', 'delta', 'phi_dot', 'delta_dot']:
            values = []
            for seed in seeds:
                entry = results.get(str(seed), {}).get(str(max_h), {}).get(model_name, {}).get('B', {})
                mae = entry.get('metrics', {}).get(state, {}).get('mae')
                if mae is not None and not np.isnan(mae) and mae < 1e6:
                    values.append(mae)
            if values:
                row += f" {np.mean(values):.4f} |"
            else:
                row += " N/A |"
        lines.append(row)

    lines.append("")

    # 消融实验对比
    lines.append("## 消融实验对比 (DAgger 轮数)")
    lines.append("")
    lines.append("| 配置 | Mode B 100步 | Mode C 100步 | Mode D 1000步 |")
    lines.append("|------|-------------|-------------|---------------|")

    ablation_models = [m for m in models if 'ensemble' in m]
    for model_name in ablation_models:
        row = f"| {model_name} |"
        for mode, h in [('B', 100), ('C', 100), ('D', 1000)]:
            values = []
            for seed in seeds:
                entry = results.get(str(seed), {}).get(str(h), {}).get(model_name, {}).get(mode, {})
                mae = entry.get('metrics', {}).get('overall', {}).get('mae')
                if mae is not None and not np.isnan(mae) and mae < 1e6:
                    values.append(mae)
            if values:
                row += f" {np.mean(values):.3f} |"
            else:
                row += " N/A |"
        lines.append(row)

    lines.append("")
    lines.append("---")
    lines.append(f"*总评估次数: {data.get('total_evaluations', 'N/A')}*")
    lines.append(f"*总耗时: {data.get('total_time', 0):.1f}s*")

    # 写入文件
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"结果汇总已保存: {OUTPUT_PATH}")

    # 同时打印到控制台
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
