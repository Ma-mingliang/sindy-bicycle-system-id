import os
path = r"D:\系统辨识作业\sindy_bicycle\continuation_stage_v9udit_v9_env_state_dynamics\B_state_normalization.md"
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    f.write(open(r"D:\系统辨识作业\sindy_bicycle\continuation_stage_v9udit_v9_env_state_dynamics\_content.txt", "r", encoding="utf-8").read())
print(f"Written to {path}")