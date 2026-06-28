"""Stage 4: MPPI diagnostics with stanley_ref injection mode.

Uses the stanley_ref-trained world model and runs MPPI with stanley_ref injection.
"""
import sys, os
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tracking_env import PathTrackingEnv
from residual_mppi import ConservativeEnsembleResidualMPPI

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'residual_mppi_stanley_ref')


def run_episode(env, agent, seed, max_steps=1500):
    obs, _ = env.reset(seed=seed)
    if agent is not None:
        agent.reset_planning()
    total_r = 0
    early_term = False
    ey_list, epsi_list, theta_list, theta_dot_list = [], [], [], []
    u_list = []

    for step in range(max_steps):
        if agent is not None:
            eps, best_G, zero_G, unc = agent.act(obs, eval_mode=True)
            action = eps
        else:
            action = np.array([0.0])
        obs, r, t, tr, info = env.step(action)
        total_r += r

        raw = info.get('raw_state', np.zeros(8))
        ey_list.append(abs(raw[0]))
        epsi_list.append(abs(raw[1]))
        theta_list.append(abs(raw[3]))
        theta_dot_list.append(abs(raw[4]))
        u_list.append(abs(info.get('u_total', 0)))

        if t:
            early_term = True
        if t or tr:
            break

    n = step + 1
    fb = agent.get_fb_reasons() if agent else {}
    return {
        'return': total_r, 'length': n,
        'early_term': early_term,
        'ey_rms': float(np.sqrt(np.mean(np.array(ey_list)**2))),
        'epsi_rms': float(np.sqrt(np.mean(np.array(epsi_list)**2))),
        'theta_rms': float(np.sqrt(np.mean(np.array(theta_list)**2))),
        'theta_max': float(np.max(np.array(theta_list))) if theta_list else 0.0,
        'control_energy': float(np.mean(np.array(u_list)**2)),
        'fb': fb,
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    sindy_data = np.load(os.path.join(os.path.dirname(__file__), 'sindy_model_improved.npz'))
    sindy_Xi = sindy_data['coefficients']
    action_scale = float(sindy_data['action_scale'])

    ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    unc_threshold = state_dict.get('val_uncertainty_p95', 0.001)

    print("=" * 100)
    print("  Stage 4: MPPI Diagnostics (stanley_ref injection)")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Uncertainty threshold: {unc_threshold:.6f}")
    print(f"  Timestamp: {timestamp}")
    print("=" * 100)

    # Test multiple margin values
    margins = [0.01, 0.005, 0.001, 0.0]
    n_episodes = 10

    base_cfg = {
        'obs_dim': 8, 'action_dim': 1, 'mlp_dim': 256,
        'horizon': 5, 'num_samples': 128, 'num_elites': 16,
        'iterations': 6, 'temperature': 0.5,
        'epsilon_max': 0.1, 'epsilon_std': 0.08,
        'gamma': 0.95,
        'lambda_res': 2.0, 'lambda_smooth': 5.0,
        'lambda_uncertainty': 5.0,
        'uncertainty_threshold': None,
        'ensemble_size': 3,
        'risk_mode': 'tolerance',
        'ey_tol': 0.02, 'theta_tol': 0.02,
    }

    header = (f"  {'margin':>7} | {'fb%':>5} | {'rwd':>5} | {'risk':>5} | {'unc':>5} | "
              f"{'marg':>5} | {'nf':>6} | {'|e|':>7} | {'e_max':>7} | "
              f"{'pred':>7} | {'real':>7} | {'ET':>3} | {'len':>5} | "
              f"{'ey_rms':>7} | {'ep_rms':>7} | {'th_rms':>7} | {'th_max':>7}")
    print(f"\n{header}")
    print(f"  {'-'*7} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*5} | "
          f"{'-'*5} | {'-'*6} | {'-'*7} | {'-'*7} | "
          f"{'-'*7} | {'-'*7} | {'-'*3} | {'-'*5} | "
          f"{'-'*7} | {'-'*7} | {'-'*7} | {'-'*7}")

    # Load best LQR config if available
    best_lqr_path = os.path.join(os.path.dirname(__file__), 'configs', 'best_lqr.json')
    lqr_kwargs = {}
    if os.path.exists(best_lqr_path):
        import json
        with open(best_lqr_path) as f:
            best = json.load(f)
        lqr_kwargs = {'lqr_Q': best['Q_diag'], 'lqr_R': best['R_val']}
        print(f"  LQR: {best['label']} (Q={best['Q_diag']}, R={best['R_val']})")

    env = PathTrackingEnv(max_episode_steps=1500, residual_injection='stanley_ref',
                          **lqr_kwargs)

    for margin in margins:
        cfg = {**base_cfg, 'margin': margin}
        agent = ConservativeEnsembleResidualMPPI(
            cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
        agent.load(ckpt_path)
        agent.uncertainty_threshold = unc_threshold

        zero_rets, mppi_rets = [], []
        all_fb, all_early, all_len = [], [], []
        all_ey, all_ep, all_th, all_th_max = [], [], [], []

        for ep in range(n_episodes):
            zero_r = run_episode(env, None, seed=ep)
            mppi_r = run_episode(env, agent, seed=ep)

            zero_rets.append(zero_r['return'])
            mppi_rets.append(mppi_r['return'])
            all_early.append(mppi_r['early_term'])
            all_fb.append(mppi_r['fb'])
            all_len.append(mppi_r['length'])
            all_ey.append(mppi_r['ey_rms'])
            all_ep.append(mppi_r['epsi_rms'])
            all_th.append(mppi_r['theta_rms'])
            all_th_max.append(mppi_r['theta_max'])

        zero_ret = np.array(zero_rets)
        mppi_ret = np.array(mppi_rets)
        adv = mppi_ret - zero_ret

        avg_fb = {}
        for key in ['one_step_reward', 'risk', 'uncertainty', 'margin']:
            avg_fb[key] = float(np.mean([r.get(key, 0) for r in all_fb]))

        fb_rate = float(np.mean([r.get('total_steps',0)-r.get('nonfallback',0)
                                  for r in all_fb])) / max(float(np.mean([r.get('total_steps',1) for r in all_fb])),1)
        nf = float(np.mean([r.get('nonfallback',0) for r in all_fb]))
        eps_mean = float(np.mean([r.get('eps_abs_mean',0) for r in all_fb]))
        eps_max = max(r.get('eps_abs_max',0) for r in all_fb)
        pred_adv = float(np.mean([r.get('predicted_adv_mean',0) for r in all_fb]))

        print(f"  {margin:>7.3f} | {fb_rate:>5.0%} | {avg_fb['one_step_reward']:>.3f} | "
              f"{avg_fb['risk']:>.3f} | {avg_fb['uncertainty']:>.3f} | {avg_fb['margin']:>.3f} | "
              f"{nf:>6.1f} | {eps_mean:>.5f} | {eps_max:>.5f} | "
              f"{pred_adv:>+7.3f} | {np.mean(adv):>+7.2f} | "
              f"{int(np.sum(all_early)):>3} | {np.mean(all_len):>5.0f} | "
              f"{np.mean(all_ey):>.4f} | {np.mean(all_ep):>.4f} | "
              f"{np.mean(all_th):>.4f} | {np.max(all_th_max):>.4f}")

    env.close()
    print(f"\n  Baseline (50 seeds, Q500_D50_R02): ET=5/20, ey_rms=2.966, epsi_rms=1.193, theta_rms=1.463")
    print("=" * 100)


if __name__ == '__main__':
    main()
