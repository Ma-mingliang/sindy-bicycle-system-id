"""V9 Data loader - reuses V8 data loading with extensions."""
import sys
sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle/continuation_stage_v8')

import numpy as np
from canonical_7d.data_loader import load_7d_data as _v8_load, get_test_segments as _v8_segments
from canonical_7d.config import DataConfig


def load_7d_data(data_path='D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz', seed=42):
    """Load 7D data using V8 loader."""
    cfg = DataConfig(data_path=data_path, seed=seed)
    return _v8_load(cfg)


def get_test_segments(data, n_segments=5, segment_length=1100, seed=42):
    """Get test segments for long-horizon evaluation."""
    return _v8_segments(data, n_segments=n_segments, segment_length=segment_length, seed=seed)
