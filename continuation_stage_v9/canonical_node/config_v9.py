"""V9 Configuration."""
import numpy as np

STATE_NAMES_7D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
STATE_DIM = 7
ACTION_DIM = 1
IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]

DATA_PATH = 'D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz'
V8_PATH = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8'
V9_PATH = 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v9'

ROLLOUT_HORIZONS = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
N_EVAL_SEGMENTS = 5
SEGMENT_LENGTH = 1100  # Need 1000+ steps

# Survival thresholds
STATE_LIMIT = 100.0  # Numerical divergence
PHYSICAL_LIMITS = {
    'e_y': 5.0, 'e_psi': np.pi, 'v': 5.0,
    'theta': np.pi, 'theta_dot': 10.0,
    'delta': np.pi/2, 'delta_dot': 10.0,
}

# Neural ODE search space
NEURAL_ODE_GRID = [
    {'hidden': 128, 'depth': 2, 'activation': 'tanh', 'lr': 1e-3, 'wd': 0},
    {'hidden': 128, 'depth': 2, 'activation': 'tanh', 'lr': 3e-4, 'wd': 1e-5},
    {'hidden': 128, 'depth': 3, 'activation': 'tanh', 'lr': 1e-3, 'wd': 0},
    {'hidden': 64, 'depth': 3, 'activation': 'tanh', 'lr': 1e-3, 'wd': 0},
    {'hidden': 128, 'depth': 2, 'activation': 'silu', 'lr': 1e-3, 'wd': 0},
    {'hidden': 128, 'depth': 2, 'activation': 'tanh', 'lr': 1e-4, 'wd': 1e-4},
]

# Residual scales to test
RESIDUAL_SCALES = [0.05, 0.1, 0.2, 0.3]

# Time-domain decay gammas
DECAY_GAMMAS = [0.9, 0.95, 0.98]

# Short-time residual K values
SHORT_TIME_KS = [1, 5, 10, 20]
