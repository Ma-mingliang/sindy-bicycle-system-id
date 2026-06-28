# -*- coding: utf-8 -*-
"""Mode B Implementation Verification Tests."""
import sys, os, json, hashlib, csv, ast, time
import numpy as np
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import methods_common as mc
import methods_evaluate as me
from continuation_stage.evaluate.eval_modes_v2 import generate_reference_trajectory, run_mode_b_fixed, run_mode_a, run_mode_c, run_mode_d
from evaluate.eval_models import RealDynamics, LinearizedModel, SINDy4D
OUTPUT_DIR = SCRIPT_DIR

def build_models():
    print("Building models...")
    t0 = time.time()
    states, actions, deltas, s_std, a_std, d_std = mc.generate_training_data(30000)
    print(f"  Training data: {time.time()-t0:.1f}s")
    models = {}
    models["real_dynamics"] = RealDynamics()
    models["linearized_model"] = LinearizedModel()
    sindy = SINDy4D("meijaard_sindy_v35.npz")
    sindy.train(states, actions, deltas, s_std, a_std, d_std)
    models["sindy_4d"] = sindy
    print(f"  Models ready: {list(models.keys())} ({time.time()-t0:.1f}s)")
    return models
