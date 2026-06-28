"""配置加载：从YAML读取所有评估参数。"""

import yaml
from pathlib import Path
from typing import Any


def load_config(yaml_path: str) -> dict:
    """加载YAML配置文件。"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_seeds(config: dict) -> list:
    """获取评估种子列表。"""
    return config['evaluation']['seeds']


def get_horizons(config: dict) -> list:
    """获取评估时域列表。"""
    return config['evaluation']['horizons']


def get_model_configs(config: dict) -> list:
    """获取模型配置列表。"""
    return config['models']


def get_state_std(config: dict):
    """获取状态归一化标准差。"""
    import numpy as np
    return np.array(config['normalization']['state_std'])


def get_action_std(config: dict) -> float:
    """获取动作归一化标准差。"""
    return config['normalization']['action_std']


def get_delta_std(config: dict):
    """获取增量归一化标准差。"""
    import numpy as np
    return np.array(config['normalization']['delta_std'])


def get_dt(config: dict) -> float:
    """获取控制周期。"""
    return config['evaluation']['dt']


def get_max_steps(config: dict) -> int:
    """获取最大步数。"""
    return config['evaluation']['max_steps']


def get_lqr_K(config: dict):
    """获取LQR增益向量。"""
    import numpy as np
    return np.array(config['evaluation']['lqr']['K_lqr'])


def get_disturbance_config(config: dict) -> tuple:
    """获取扰动配置: (range, seed_offset)。"""
    d = config['evaluation']['disturbance']
    return d['range'], d['seed_offset']


def get_phi_limit(config: dict) -> float:
    """获取安全横滚角限制。"""
    import math
    val = config['evaluation']['safety']['phi_limit']
    if isinstance(val, str) and 'pi' in val:
        return eval(val, {'pi': math.pi, '__builtins__': {}})
    return float(val)


def get_project_root() -> Path:
    """获取项目根目录。"""
    return Path(__file__).parent.parent


def resolve_data_path(config: dict, key: str) -> Path:
    """解析数据文件的绝对路径。"""
    root = get_project_root()
    return root / config['data'][key]['file']
