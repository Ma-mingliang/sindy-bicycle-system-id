# 恢复指令

## RUN_ID: 20260628_175824_neural_ode_72h

---

## 恢复步骤

### 1. 检查当前状态
```bash
cd D:/系统辨识作业/sindy_bicycle
git status
git branch --show-current
cat research_72h/runtime/GOAL_STATE.json
cat research_72h/runtime/CURRENT_STATUS.md
```

### 2. 确认数据文件
```bash
ls -la data/stage2_dataset_150k.npz
ls -la continuation_stage_v9/checkpoints/BEST_NEURAL_ODE_V9.pt
```

### 3. 确认 Python 环境
```bash
"E:/Anaconda/envs/DL/python.exe" --version
"E:/Anaconda/envs/DL/python.exe" -c "import torch; print(torch.__version__)"
```

### 4. 继续当前阶段
根据 CURRENT_STATUS.md 中的阶段继续工作。

### 5. 更新状态
每完成一个步骤，更新：
- research_72h/runtime/GOAL_STATE.json
- research_72h/runtime/CURRENT_STATUS.md
- research_72h/runtime/EXECUTION_LEDGER.md

---

## 关键路径

1. 上下文恢复 → 2. v9复现 → 3. 失效机理 → 4. 论文搜索 → 5. 子代理审计 → 6. 候选设计 → 7. 快速筛选 → 8. 完整训练 → 9. 多seed验证 → 10. 消融/OOD → 11. 独立复核 → 12. GitHub交付 → 13. 本地归档

---

## 退出条件

所有退出条件必须满足，参考 `01_CLAUDE_CODE_72H_GOAL_主任务.md` 第25节。
