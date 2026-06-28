"""Quick MPPI diagnostics: load checkpoint, run 5 episodes, report fallback reasons."""
import sys, os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tracking_env import PathTrackingEnv
from residual_mppi import ConservativeEnsembleResidualMPPI

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'residual_mppi_ensemble_v1')

def main():
    sys.stdout.reconfigure(line_buffering=True)

    sindy_data = np.load(os.path.join(os.path.dirname(__file__), 'sindy_model_improved.npz'))
    sindy_Xi = sindy_data['coefficients']
    action_scale = float(sindy_data['action_scale'])

    cfg = {
        'obs_dim': 8, 'action_dim': 1, 'mlp_dim': 256,
        'horizon': 5, 'num_samples': 128, 'num_elites': 16,
        'iterations': 6, 'temperature': 0.5,
        'epsilon_max': 0.1, 'epsilon_std': 0.08,
        'gamma': 0.95, 'lambda_res': 2.0, 'lambda_smooth': 5.0,
        'lambda_uncertainty': 5.0, 'uncertainty_threshold': None,
        'margin': 0.1, 'ensemble_size': 3,
    }

    ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
    agent = ConservativeEnsembleResidualMPPI(cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
    agent.load(ckpt_path)

    # Load threshold from checkpoint
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'uncertainty_threshold_p95' in state_dict:
        agent.uncertainty_threshold = state_dict['uncertainty_threshold_p95']
        print(f"  Uncertainty threshold (p95): {agent.uncertainty_threshold:.6f}")
    if 'val_uncertainty_p90' in state_dict:
        print(f"  Val uncertainty p90: {state_dict['val_uncertainty_p90']:.6f}")
        print(f"  Val uncertainty p95: {state_dict.get('val_uncertainty_p95', 'N/A')}")

    env = PathTrackingEnv(max_episode_steps=1500)

    print("\n  === MPPI Diagnostics (10 episodes) ===\n")

    for ep in range(10):
        obs, _ = env.reset(seed=ep)
        agent.reset_planning()
        total_r = 0
        nf_count = 0
        fb_count = 0
        eps_list = []
        unc_list = []

        for step in range(1500):
            eps, best_G, zero_G, unc = agent.act(obs, eval_mode=True)
            if np.allclose(eps, 0):
                fb_count += 1
            else:
                nf_count += 1
                eps_list.append(np.abs(eps).item())
            unc_list.append(unc)
            obs, r, t, tr, info = env.step(eps)
            total_r += r
            if t or tr:
                break

        n = step + 1
        fb = agent.get_fb_reasons()
        eps_arr = np.array(eps_list) if eps_list else np.array([0])

        print(f"  Ep {ep:>2d} ({n:>4d} steps): ret={total_r:>7.1f}  "
              f"fb={fb_count/n:.0%}  nf={nf_count:>3d}  "
              f"|eps|={eps_arr.mean():.4f}  unc={np.mean(unc_list):.6f}")
        print(f"           fb_reasons: reward={fb.get('one_step_reward',0):.3f}  "
              f"risk={fb.get('risk',0):.3f}  "
              f"uncertainty={fb.get('uncertainty',0):.3f}  "
              f"margin={fb.get('margin',0):.3f}")

    env.close()

if __name__ == '__main__':
    main()
