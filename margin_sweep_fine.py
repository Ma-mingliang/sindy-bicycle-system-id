"""Fine-grained margin sweep: risk_mode=tolerance fixed, margin=0.02, 0.005."""
import sys, os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_tracking_env import PathTrackingEnv
from residual_mppi import ConservativeEnsembleResidualMPPI

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'residual_mppi_ensemble_v1')


def run_episode(env, agent, seed, max_steps=1500):
    obs, _ = env.reset(seed=seed)
    if agent is not None:
        agent.reset_planning()
    total_r = 0
    early_term = False
    ey_list, theta_list, epsi_list = [], [], []

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
        'fb': fb,
    }


def eval_config(sindy_Xi, action_scale, cfg, env, unc_threshold, n_episodes=10):
    agent = ConservativeEnsembleResidualMPPI(
        cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
    ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
    agent.load(ckpt_path)
    agent.uncertainty_threshold = unc_threshold

    zero_rets, mppi_rets = [], []
    all_fb, all_early = [], []
    all_nf, all_eps_mean, all_eps_max, all_clip = [], [], [], []
    all_pred_adv, all_ey_rms, all_theta_rms, all_epsi_rms = [], [], [], []

    for ep in range(n_episodes):
        zero_r = run_episode(env, None, seed=ep)
        mppi_r = run_episode(env, agent, seed=ep)

        zero_rets.append(zero_r['return'])
        mppi_rets.append(mppi_r['return'])
        all_early.append(mppi_r['early_term'])
        all_fb.append(mppi_r['fb'])
        all_ey_rms.append(mppi_r['ey_rms'])
        all_epsi_rms.append(mppi_r['epsi_rms'])
        all_theta_rms.append(mppi_r['theta_rms'])

        fb = mppi_r['fb']
        all_nf.append(fb.get('nonfallback', 0))
        all_eps_mean.append(fb.get('eps_abs_mean', 0))
        all_eps_max.append(fb.get('eps_abs_max', 0))
        all_clip.append(fb.get('clip_rate', 0))
        all_pred_adv.append(fb.get('predicted_adv_mean', 0))

    zero_ret = np.array(zero_rets)
    mppi_ret = np.array(mppi_rets)
    adv = mppi_ret - zero_ret

    avg_fb = {}
    for key in ['one_step_reward', 'risk', 'uncertainty', 'margin']:
        avg_fb[key] = float(np.mean([r.get(key, 0) for r in all_fb]))

    return {
        'zero_mean': float(np.mean(zero_ret)),
        'mppi_mean': float(np.mean(mppi_ret)),
        'real_advantage': float(np.mean(adv)),
        'early_term_count': int(np.sum(all_early)),
        'fb_rate': float(np.mean([r.get('total_steps', 0) - r.get('nonfallback', 0)
                                   for r in all_fb])) / max(float(np.mean([r.get('total_steps', 1) for r in all_fb])), 1),
        'fb_by_reward': avg_fb['one_step_reward'],
        'fb_by_risk': avg_fb['risk'],
        'fb_by_uncertainty': avg_fb['uncertainty'],
        'fb_by_margin': avg_fb['margin'],
        'nf_steps': float(np.mean(all_nf)),
        'eps_abs_mean': float(np.mean(all_eps_mean)),
        'eps_abs_max': float(np.max(all_eps_max)) if any(x > 0 for x in all_eps_max) else 0.0,
        'clip_rate': float(np.mean(all_clip)),
        'predicted_adv_mean': float(np.mean(all_pred_adv)),
        'ey_rms': float(np.mean(all_ey_rms)),
        'epsi_rms': float(np.mean(all_epsi_rms)),
        'theta_rms': float(np.mean(all_theta_rms)),
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)

    sindy_data = np.load(os.path.join(os.path.dirname(__file__), 'sindy_model_improved.npz'))
    sindy_Xi = sindy_data['coefficients']
    action_scale = float(sindy_data['action_scale'])

    ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    unc_threshold = state_dict.get('uncertainty_threshold_p95', 0.01)

    env = PathTrackingEnv(max_episode_steps=1500)
    n_eps = 10

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

    margins = [0.02, 0.005]

    print("=" * 120)
    print("  Fine-Grained Margin Sweep")
    print(f"  Fixed: risk_mode=tolerance, ey_tol=0.02, theta_tol=0.02")
    print(f"  horizon=5, epsilon_max=0.1, lambda_res=2.0, lambda_smooth=5.0")
    print(f"  uncertainty_threshold={unc_threshold:.6f}")
    print(f"  one_step_reward: diagnostic-only")
    print(f"  Episodes per config: {n_eps}")
    print("=" * 120)

    header = (f"  {'margin':<8} | {'fb%':>5} | {'rwd':>5} | {'risk':>5} | {'unc':>5} | "
              f"{'marg':>5} | {'nf':>6} | {'|e|':>7} | {'e_max':>7} | {'clip':>5} | "
              f"{'pred':>7} | {'real':>7} | {'ET':>3} | {'ey_rms':>7} | {'ep_rms':>7} | {'th_rms':>7}")
    print(header)
    print(f"  {'-'*8} | {'-'*5} | {'-'*5} | {'-'*5} | {'-'*5} | "
          f"{'-'*5} | {'-'*6} | {'-'*7} | {'-'*7} | {'-'*5} | "
          f"{'-'*7} | {'-'*7} | {'-'*3} | {'-'*7} | {'-'*7} | {'-'*7}")

    results = {}
    for margin in margins:
        cfg = {**base_cfg, 'margin': margin}
        r = eval_config(sindy_Xi, action_scale, cfg, env, unc_threshold, n_eps)
        results[margin] = r
        print(f"  {margin:<8} | {r['fb_rate']:>5.0%} | {r['fb_by_reward']:>.3f} | "
              f"{r['fb_by_risk']:>.3f} | {r['fb_by_uncertainty']:>.3f} | {r['fb_by_margin']:>.3f} | "
              f"{r['nf_steps']:>6.1f} | {r['eps_abs_mean']:>.5f} | {r['eps_abs_max']:>.5f} | "
              f"{r['clip_rate']:>.3f} | "
              f"{r['predicted_adv_mean']:>+7.3f} | {r['real_advantage']:>+7.2f} | "
              f"{r['early_term_count']:>3} | {r['ey_rms']:>.4f} | {r['epsi_rms']:>.4f} | {r['theta_rms']:>.4f}")

    # Judgment
    print("\n" + "=" * 120)
    print("  JUDGMENT")
    print("=" * 120)

    for margin, r in results.items():
        nf_up = r['nf_steps'] > 1.0
        adv_pos = r['real_advantage'] > 0
        et_ok = r['early_term_count'] <= 5
        ey_ok = r['ey_rms'] < 3.0

        status = []
        status.append(f"nf={'UP' if nf_up else 'FLAT'}({r['nf_steps']:.1f})")
        status.append(f"adv={'+' if adv_pos else '~'}({r['real_advantage']:+.2f})")
        status.append(f"ET={'OK' if et_ok else 'WARN'}({r['early_term_count']})")
        status.append(f"ey={'OK' if ey_ok else 'WORSE'}({r['ey_rms']:.3f})")

        verdict = "PASS" if (nf_up and adv_pos and et_ok) else "INSUFFICIENT"
        print(f"  margin={margin}: {verdict} — {', '.join(status)}")

    # Compare with baseline (margin=0.1 from risk_ablation)
    print(f"\n  Baseline (margin=0.1, risk=strict): fb=100%, nf=0.4, real_adv=+0.00, ey_rms=2.958")

    env.close()
    print("\n" + "=" * 120)


if __name__ == '__main__':
    main()
