"""用评估v2的实际Mode B结果更新时域推荐报告。

用法:
    python continuation_stage/analyze/update_horizon_recommendation.py
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_PATH = PROJECT_ROOT / 'continuation_stage' / 'results' / 'full_evaluation_v2.json'
REPORT_PATH = PROJECT_ROOT / 'continuation_stage' / 'reports' / 'CORRECTED_MODEL_HORIZON_RECOMMENDATION.md'


def main():
    if not RESULTS_PATH.exists():
        print(f"结果文件不存在: {RESULTS_PATH}")
        return

    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    config = data['config']
    seeds = config['seeds']
    horizons = config['horizons']
    models = config['models']
    results = data['data']

    # 提取 Mode B 结果
    mode_b_results = {}
    for model_name in models:
        mode_b_results[model_name] = {}
        for h in horizons:
            mae_values = []
            for seed in seeds:
                entry = results.get(str(seed), {}).get(str(h), {}).get(model_name, {}).get('B', {})
                mae = entry.get('metrics', {}).get('overall', {}).get('mae')
                if mae is not None and not np.isnan(mae) and mae < 1e6:
                    mae_values.append(mae)
            if mae_values:
                mode_b_results[model_name][h] = {
                    'mean': float(np.mean(mae_values)),
                    'std': float(np.std(mae_values)),
                    'n': len(mae_values),
                }
            else:
                mode_b_results[model_name][h] = None

    # 生成报告
    now = datetime.now().strftime('%Y-%m-%d')
    lines = [
        "# 修正模型时域推荐 (v2)",
        "",
        f"> **生成时间**: {now}",
        f"> **依据**: Mode B (固定动作开环) 实际评估结果",
        f"> **种子数**: {len(seeds)}",
        "",
        "---",
        "",
        "## 1. 时域推荐原则",
        "",
        "**不得依据 Mode D 近零误差直接推荐 MPC 时域。**",
        "",
        "Mode D 的零误差是因为 LQR 控制器完美补偿，不是模型本身完美。因此:",
        "- Mode B (固定动作开环) 是世界模型多步预测的主要依据",
        "- Mode A 用于单步误差参考",
        "- Mode C 用于漂移参考",
        "- Mode D 用于闭环稳定性参考",
        "",
        "---",
        "",
        "## 2. Mode B 实际结果",
        "",
    ]

    # Mode B 结果表
    header = "| 模型 |"
    separator = "|------|"
    for h in horizons:
        header += f" {h}步 |"
        separator += "------|"
    lines.append(header)
    lines.append(separator)

    for model_name in models:
        row = f"| {model_name} |"
        for h in horizons:
            r = mode_b_results[model_name].get(h)
            if r:
                if r['mean'] > 10:
                    row += f" {r['mean']:.1f}±{r['std']:.1f} |"
                else:
                    row += f" {r['mean']:.3f}±{r['std']:.3f} |"
            else:
                row += " 发散 |"
        lines.append(row)

    lines.extend([
        "",
        "---",
        "",
        "## 3. 时域等级",
        "",
        "| 等级 | MAE 阈值 |",
        "|------|----------|",
        "| 强可信 | < 0.01 rad |",
        "| 可用 | < 0.1 rad |",
        "| 谨慎使用 | < 0.5 rad |",
        "| 不推荐 | ≥ 0.5 rad |",
        "",
    ])

    # 每个模型的推荐
    lines.append("## 4. 模型时域推荐")
    lines.append("")

    for model_name in models:
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("| 时域 | MAE | 等级 |")
        lines.append("|------|-----|------|")

        for h in horizons:
            r = mode_b_results[model_name].get(h)
            if r:
                mae = r['mean']
                if mae < 0.01:
                    level = "强可信"
                elif mae < 0.1:
                    level = "可用"
                elif mae < 0.5:
                    level = "谨慎使用"
                else:
                    level = "不推荐"
                lines.append(f"| {h}步 | {mae:.4f} rad | {level} |")
            else:
                lines.append(f"| {h}步 | 发散 | 不推荐 |")
        lines.append("")

    # 分场景推荐
    lines.extend([
        "---",
        "",
        "## 5. 分场景推荐",
        "",
        "### 5.1 MPC 候选动作 rollout",
        "",
    ])

    for model_name in models:
        max_ok_h = 0
        for h in sorted(horizons):
            r = mode_b_results[model_name].get(h)
            if r and r['mean'] < 0.1:
                max_ok_h = h
        if max_ok_h > 0:
            lines.append(f"- **{model_name}**: ≤{max_ok_h}步 (MAE < 0.1 rad)")
        else:
            lines.append(f"- **{model_name}**: 不推荐 (所有时域 MAE ≥ 0.1 rad)")

    lines.extend([
        "",
        "### 5.2 MPPI 时域",
        "",
    ])

    for model_name in models:
        max_ok_h = 0
        for h in sorted(horizons):
            r = mode_b_results[model_name].get(h)
            if r and r['mean'] < 0.1:
                max_ok_h = h
        if max_ok_h > 0:
            lines.append(f"- **{model_name}**: {max_ok_h}步 (MAE < 0.1 rad)")
        else:
            lines.append(f"- **{model_name}**: 不推荐")

    lines.extend([
        "",
        "---",
        "",
        "## 6. 结论",
        "",
        "1. **Mode B 是世界模型评估的主要依据**: 固定动作开环测试",
        "2. **不要用 Mode D 推荐 MPC 时域**: Mode D 的零误差是控制器补偿，不是模型精度",
        "3. **时域应基于误差容忍度选择**",
        "",
        "---",
        f"",
        f"*报告时间: {now}*",
        f"*基于 {len(seeds)} 个种子的评估结果*",
    ])

    # 写入文件
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"时域推荐报告已更新: {REPORT_PATH}")


if __name__ == '__main__':
    main()
