"""Comprehensive residual diagnostic: 4 phases.

Phase 1: Margin gate per-step logging (advantage distribution)
Phase 2: Forced residual test (fixed epsilon, no MPPI)
Phase 3: No-margin MPPI diagnostic
Phase 4: Injection position comparison (A/B/C)
"""
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
        u_list.append(abs(info.get('u_total', 0.0)))

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
        'control_energy': float(np.mean(np.array(u_list)**2)),
        'u_abs_mean': float(np.mean(np.array(u_list))),
        'u_abs_max': float(np.max(np.array(u_list))) if u_list else 0.0,
        'fb': fb,
    }


# ============================================================
# Phase 1: Margin gate per-step diagnostic
# ============================================================
def phase1_margin_diagnostic(sindy_Xi, action_scale, unc_threshold, n_episodes=10):
    print("\n" + "=" * 100)
    print("  PHASE 1: Margin Gate Per-Step Diagnostic")
    print("  Purpose: explain why pred_adv > margin but margin gate still triggers")
    print("=" * 100)

    cfg = {
        'obs_dim': 8, 'action_dim': 1, 'mlp_dim': 256,
        'horizon': 5, 'num_samples': 128, 'num_elites': 16,
        'iterations': 6, 'temperature': 0.5,
        'epsilon_max': 0.1, 'epsilon_std': 0.08,
        'gamma': 0.95,
        'lambda_res': 2.0, 'lambda_smooth': 5.0,
        'lambda_uncertainty': 5.0,
        'uncertainty_threshold': None,
        'margin': 0.005,  # smallest margin tested
        'ensemble_size': 3,
        'risk_mode': 'tolerance',
        'ey_tol': 0.02, 'theta_tol': 0.02,
    }

    agent = ConservativeEnsembleResidualMPPI(
        cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
    ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
    agent.load(ckpt_path)
    agent.uncertainty_threshold = unc_threshold

    env = PathTrackingEnv(max_episode_steps=1500)

    # Collect per-step data
    all_adv_raw = []  # raw advantage (G_raw - zero_G) for ALL steps
    all_adv_pen = []  # penalized advantage (G_total - zero_G)
    margin_pass_count = 0
    margin_fail_count = 0
    all_eps = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        agent.reset_planning()

        for step in range(1500):
            # Temporarily patch act() to collect pre-gate data
            # We use the agent normally but collect from fb_reasons
            eps, best_G, zero_G, unc = agent.act(obs, eval_mode=True)
            action = eps
            obs, r, t, tr, info = env.step(action)

            # Check if this step passed margin gate
            fb = agent._fb_reason
            if fb['nonfallback'] > 0:
                # This step passed (since nonfallback just incremented)
                margin_pass_count += 1
            else:
                margin_fail_count += 1

            if t or tr:
                break

        fb = agent.get_fb_reasons()
        all_adv_raw.append(fb.get('step_adv_raw_mean', 0))
        all_adv_pen.append(fb.get('step_adv_pen_mean', 0))

    total_steps = margin_pass_count + margin_fail_count

    print(f"\n  Config: margin=0.005, risk=tolerance, ey_tol=0.02")
    print(f"  Episodes: {n_episodes}")
    print(f"  Total steps: {total_steps}")
    print(f"  Margin pass: {margin_pass_count} ({margin_pass_count/max(total_steps,1):.1%})")
    print(f"  Margin fail: {margin_fail_count} ({margin_fail_count/max(total_steps,1):.1%})")

    print(f"\n  Advantage stats (per-episode mean, over ALL steps):")
    adv_raw = np.array(all_adv_raw)
    adv_pen = np.array(all_adv_pen)

    print(f"    adv_raw (G_raw - zero_G):")
    print(f"      mean={adv_raw.mean():+.5f}, std={adv_raw.std():.5f}")
    print(f"      p10={np.percentile(adv_raw,10):+.5f}, p25={np.percentile(adv_raw,25):+.5f}")
    print(f"      p50={np.percentile(adv_raw,50):+.5f}, p75={np.percentile(adv_raw,75):+.5f}")
    print(f"      p90={np.percentile(adv_raw,90):+.5f}, p95={np.percentile(adv_raw,95):+.5f}")
    print(f"      max={adv_raw.max():+.5f}")

    print(f"\n    adv_pen (G_total - zero_G):")
    print(f"      mean={adv_pen.mean():+.5f}, std={adv_pen.std():.5f}")
    print(f"      p10={np.percentile(adv_pen,10):+.5f}, p25={np.percentile(adv_pen,25):+.5f}")
    print(f"      p50={np.percentile(adv_pen,50):+.5f}, p75={np.percentile(adv_pen,75):+.5f}")
    print(f"      p90={np.percentile(adv_pen,90):+.5f}, p95={np.percentile(adv_pen,95):+.5f}")
    print(f"      max={adv_pen.max():+.5f}")

    penalty_effect = adv_raw.mean() - adv_pen.mean()
    print(f"\n    Penalty effect: {penalty_effect:+.5f}")
    print(f"    (penalty makes advantage {penalty_effect:.5f} lower)")

    print(f"\n  DIAGNOSIS:")
    if adv_raw.mean() > 0.005 and adv_pen.mean() < 0:
        print(f"    BUG CONFIRMED: raw advantage is positive ({adv_raw.mean():+.5f})")
        print(f"    but penalized advantage is negative ({adv_pen.mean():+.5f}).")
        print(f"    Margin gate was comparing penalized vs unpenalized — FIXED.")
    elif adv_raw.mean() < 0.005:
        print(f"    raw advantage is very small ({adv_raw.mean():+.5f}).")
        print(f"    MPPI cannot find meaningful improvement over zero action.")
        print(f"    Problem is not the gate — it's the planning quality.")
    else:
        print(f"    raw advantage = {adv_raw.mean():+.5f}, margin = 0.005")
        print(f"    Gate should pass if raw_adv > margin.")

    env.close()
    return adv_raw, adv_pen


# ============================================================
# Phase 2: Forced residual test
# ============================================================
def phase2_forced_residual(sindy_Xi, action_scale, unc_threshold, n_episodes=10):
    print("\n" + "=" * 100)
    print("  PHASE 2: Forced Residual Test")
    print("  Purpose: verify residual channel can change trajectory")
    print("=" * 100)

    base_cfg = {
        'obs_dim': 8, 'action_dim': 1, 'mlp_dim': 256,
        'horizon': 5, 'num_samples': 128, 'num_elites': 16,
        'iterations': 6, 'temperature': 0.5,
        'epsilon_max': 0.1, 'epsilon_std': 0.08,
        'gamma': 0.95,
        'lambda_res': 2.0, 'lambda_smooth': 5.0,
        'lambda_uncertainty': 5.0,
        'uncertainty_threshold': None,
        'margin': 0.005,
        'ensemble_size': 3,
        'risk_mode': 'tolerance',
        'ey_tol': 0.02, 'theta_tol': 0.02,
    }

    epsilons = [-0.10, -0.05, 0.00, 0.05, 0.10]
    env = PathTrackingEnv(max_episode_steps=1500)

    header = (f"  {'eps':>6} | {'ret':>8} | {'ret_std':>8} | {'len':>6} | {'ET':>3} | "
              f"{'ey_rms':>7} | {'ep_rms':>7} | {'th_rms':>7} | {'E[u]':>7} | "
              f"{'|u|':>7} | {'|u|max':>7}")
    print(header)
    print(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*6} | {'-'*3} | "
          f"{'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7}")

    results = {}
    for eps_val in epsilons:
        cfg = {**base_cfg, 'forced_epsilon': eps_val}
        agent = ConservativeEnsembleResidualMPPI(
            cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
        ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
        agent.load(ckpt_path)
        agent.uncertainty_threshold = unc_threshold

        rets, lengths, ets = [], [], []
        ey_rs, ep_rs, th_rs = [], [], []
        u_energy, u_abs, u_max = [], [], []

        for ep in range(n_episodes):
            r = run_episode(env, agent, seed=ep)
            rets.append(r['return'])
            lengths.append(r['length'])
            ets.append(r['early_term'])
            ey_rs.append(r['ey_rms'])
            ep_rs.append(r['epsi_rms'])
            th_rs.append(r['theta_rms'])
            u_energy.append(r['control_energy'])
            u_abs.append(r['u_abs_mean'])
            u_max.append(r['u_abs_max'])

        results[eps_val] = {
            'ret_mean': np.mean(rets), 'ret_std': np.std(rets),
            'len_mean': np.mean(lengths), 'ET': int(np.sum(ets)),
            'ey_rms': np.mean(ey_rs), 'epsi_rms': np.mean(ep_rs),
            'theta_rms': np.mean(th_rs), 'energy': np.mean(u_energy),
            'u_abs': np.mean(u_abs), 'u_max': np.max(u_max),
        }
        r = results[eps_val]
        print(f"  {eps_val:>+6.2f} | {r['ret_mean']:>+8.2f} | {r['ret_std']:>8.2f} | "
              f"{r['len_mean']:>6.0f} | {r['ET']:>3} | "
              f"{r['ey_rms']:>.4f} | {r['epsi_rms']:>.4f} | {r['theta_rms']:>.4f} | "
              f"{r['energy']:>.4f} | {r['u_abs']:>.4f} | {r['u_max']:>.4f}")

    # Judgment
    baseline = results[0.00]
    print(f"\n  DIAGNOSIS:")
    max_delta_ey = max(abs(r['ey_rms'] - baseline['ey_rms']) for r in results.values())
    max_delta_th = max(abs(r['theta_rms'] - baseline['theta_rms']) for r in results.values())

    if max_delta_ey < 0.001 and max_delta_th < 0.001:
        print(f"    RESIDUAL CHANNEL INEFFECTIVE: max |delta ey_rms|={max_delta_ey:.4f}")
        print(f"    epsilon_max=0.1 has no effect on trajectory.")
        print(f"    alpha=0.15 may be too small, or dynamics don't respond to torque residual.")
    elif max_delta_ey > 0.01:
        print(f"    RESIDUAL CHANNEL WORKS: max |delta ey_rms|={max_delta_ey:.4f}")
        best_eps = min(results.keys(), key=lambda e: results[e]['ey_rms'])
        print(f"    Best ey_rms at eps={best_eps:+.2f}: {results[best_eps]['ey_rms']:.4f}")
        print(f"    vs baseline eps=0: {baseline['ey_rms']:.4f}")
    else:
        print(f"    WEAK EFFECT: max |delta ey_rms|={max_delta_ey:.4f}")
        print(f"    Residual has some effect but very small.")

    env.close()
    return results


# ============================================================
# Phase 3: No-margin MPPI diagnostic
# ============================================================
def phase3_no_margin(sindy_Xi, action_scale, unc_threshold, n_episodes=10):
    print("\n" + "=" * 100)
    print("  PHASE 3: No-Margin MPPI Diagnostic")
    print("  Purpose: test if MPPI produces real trajectory difference without margin gate")
    print("=" * 100)

    cfg = {
        'obs_dim': 8, 'action_dim': 1, 'mlp_dim': 256,
        'horizon': 5, 'num_samples': 128, 'num_elites': 16,
        'iterations': 6, 'temperature': 0.5,
        'epsilon_max': 0.1, 'epsilon_std': 0.08,
        'gamma': 0.95,
        'lambda_res': 2.0, 'lambda_smooth': 5.0,
        'lambda_uncertainty': 5.0,
        'uncertainty_threshold': None,
        'margin': 0.005,
        'margin_gate_off': True,  # KEY: disable margin gate
        'ensemble_size': 3,
        'risk_mode': 'hard_limit_only',  # relaxed risk
        'ey_hard_limit': 4.0, 'theta_hard_limit': 1.2,
    }

    agent = ConservativeEnsembleResidualMPPI(
        cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
    ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
    agent.load(ckpt_path)
    agent.uncertainty_threshold = unc_threshold

    env = PathTrackingEnv(max_episode_steps=1500)

    zero_rets, mppi_rets = [], []
    all_fb, all_early = [], []
    all_ey, all_ep, all_th = [], [], []
    all_u_energy = []

    for ep in range(n_episodes):
        zero_r = run_episode(env, None, seed=ep)
        mppi_r = run_episode(env, agent, seed=ep)

        zero_rets.append(zero_r['return'])
        mppi_rets.append(mppi_r['return'])
        all_early.append(mppi_r['early_term'])
        all_fb.append(mppi_r['fb'])
        all_ey.append(mppi_r['ey_rms'])
        all_ep.append(mppi_r['epsi_rms'])
        all_th.append(mppi_r['theta_rms'])
        all_u_energy.append(mppi_r['control_energy'])

    zero_ret = np.array(zero_rets)
    mppi_ret = np.array(mppi_rets)
    adv = mppi_ret - zero_ret

    avg_fb = {}
    for key in ['one_step_reward', 'risk', 'uncertainty', 'margin']:
        avg_fb[key] = float(np.mean([r.get(key, 0) for r in all_fb]))

    print(f"\n  Config: margin_gate_off=True, risk=hard_limit_only")
    print(f"  Episodes: {n_episodes}")
    print(f"\n  {'metric':<30} | {'value':>10}")
    print(f"  {'-'*30} | {'-'*10}")
    print(f"  {'zero_return_mean':<30} | {np.mean(zero_ret):>+10.2f}")
    print(f"  {'mppi_return_mean':<30} | {np.mean(mppi_ret):>+10.2f}")
    print(f"  {'real_advantage':<30} | {np.mean(adv):>+10.2f}")
    print(f"  {'early_term_count':<30} | {int(np.sum(all_early)):>10}")
    print(f"  {'fb_rate':<30} | {float(np.mean([r.get('total_steps',0)-r.get('nonfallback',0) for r in all_fb]))/max(float(np.mean([r.get('total_steps',1) for r in all_fb])),1):>10.1%}")
    print(f"  {'nf_steps':<30} | {float(np.mean([r.get('nonfallback',0) for r in all_fb])):>10.1f}")
    print(f"  {'eps_abs_mean':<30} | {float(np.mean([r.get('eps_abs_mean',0) for r in all_fb])):>10.4f}")
    print(f"  {'eps_abs_max':<30} | {max(r.get('eps_abs_max',0) for r in all_fb):>10.4f}")
    print(f"  {'clip_rate':<30} | {float(np.mean([r.get('clip_rate',0) for r in all_fb])):>10.3f}")
    print(f"  {'predicted_adv_mean(raw)':<30} | {float(np.mean([r.get('step_adv_raw_mean',0) for r in all_fb])):>+10.5f}")
    print(f"  {'predicted_adv_mean(pen)':<30} | {float(np.mean([r.get('step_adv_pen_mean',0) for r in all_fb])):>+10.5f}")
    print(f"  {'ey_rms_mppi':<30} | {np.mean(all_ey):>10.4f}")
    print(f"  {'theta_rms_mppi':<30} | {np.mean(all_th):>10.4f}")
    print(f"  {'epsi_rms_mppi':<30} | {np.mean(all_ep):>10.4f}")
    print(f"  {'control_energy_mppi':<30} | {np.mean(all_u_energy):>10.4f}")

    # Zero baseline stats
    zero_ey, zero_th, zero_ep, zero_u = [], [], [], []
    for ep in range(n_episodes):
        r = run_episode(env, None, seed=ep)
        raw = r.get('fb', {})
        zero_ey.append(r['ey_rms'])
        zero_th.append(r['theta_rms'])
        zero_ep.append(r['epsi_rms'])
        zero_u.append(r['control_energy'])

    print(f"\n  {'ey_rms_zero':<30} | {np.mean(zero_ey):>10.4f}")
    print(f"  {'theta_rms_zero':<30} | {np.mean(zero_th):>10.4f}")
    print(f"  {'epsi_rms_zero':<30} | {np.mean(zero_ep):>10.4f}")
    print(f"  {'control_energy_zero':<30} | {np.mean(zero_u):>10.4f}")

    delta_ey = np.mean(all_ey) - np.mean(zero_ey)
    delta_th = np.mean(all_th) - np.mean(zero_th)

    print(f"\n  DIAGNOSIS:")
    if np.mean(adv) > 0 and int(np.sum(all_early)) <= 5:
        print(f"    POSITIVE: no-margin MPPI shows real_advantage={np.mean(adv):+.2f}")
        print(f"    margin gate was too conservative. Consider adjusting margin scale.")
    elif np.mean(adv) < -5:
        print(f"    NEGATIVE: no-margin MPPI hurts performance (adv={np.mean(adv):+.2f})")
        print(f"    margin gate is necessary. MPPI predictions are unreliable.")
    else:
        print(f"    NEUTRAL: real_advantage ≈ {np.mean(adv):+.2f}")
        print(f"    MPPI residual has minimal real effect even without margin gate.")

    if abs(delta_ey) < 0.001:
        print(f"    ey_rms unchanged ({delta_ey:+.4f}) — residual doesn't affect trajectory.")
    elif delta_ey < -0.01:
        print(f"    ey_rms improved ({delta_ey:+.4f}) — residual helps path tracking.")
    else:
        print(f"    ey_rms worsened ({delta_ey:+.4f}) — residual hurts path tracking.")

    env.close()
    return np.mean(adv), np.mean(all_ey), np.mean(zero_ey)


# ============================================================
# Phase 4: Injection position comparison
# ============================================================
def phase4_injection_comparison(sindy_Xi, action_scale, unc_threshold, n_episodes=10):
    print("\n" + "=" * 100)
    print("  PHASE 4: Residual Injection Position Comparison")
    print("  A: final-action residual (current)")
    print("  B: Stanley reference residual")
    print("  C: theta_target residual")
    print("=" * 100)

    modes = ['final_action', 'stanley_ref', 'theta_target']
    mode_labels = {'final_action': 'A: final_action', 'stanley_ref': 'B: stanley_ref',
                   'theta_target': 'C: theta_target'}

    # Test with margin_gate_off and margin=0.005
    margin_configs = [0.005, None]  # None = gate off

    for margin_cfg in margin_configs:
        label = "margin=OFF" if margin_cfg is None else f"margin={margin_cfg}"
        print(f"\n  --- {label} ---")

        header = (f"  {'mode':<20} | {'ret':>8} | {'adv':>8} | {'ET':>3} | {'len':>6} | "
                  f"{'ey_rms':>7} | {'ep_rms':>7} | {'th_rms':>7} | {'fb%':>5} | "
                  f"{'nf':>5} | {'|e|':>7} | {'e_max':>7}")
        print(header)
        print(f"  {'-'*20} | {'-'*8} | {'-'*8} | {'-'*3} | {'-'*6} | "
              f"{'-'*7} | {'-'*7} | {'-'*7} | {'-'*5} | {'-'*5} | {'-'*7} | {'-'*7}")

        for mode in modes:
            cfg = {
                'obs_dim': 8, 'action_dim': 1, 'mlp_dim': 256,
                'horizon': 5, 'num_samples': 128, 'num_elites': 16,
                'iterations': 6, 'temperature': 0.5,
                'epsilon_max': 0.1, 'epsilon_std': 0.08,
                'gamma': 0.95,
                'lambda_res': 2.0, 'lambda_smooth': 5.0,
                'lambda_uncertainty': 5.0,
                'uncertainty_threshold': None,
                'margin': margin_cfg if margin_cfg is not None else 0.005,
                'margin_gate_off': margin_cfg is None,
                'ensemble_size': 3,
                'risk_mode': 'tolerance',
                'ey_tol': 0.02, 'theta_tol': 0.02,
            }

            agent = ConservativeEnsembleResidualMPPI(
                cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)
            ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
            agent.load(ckpt_path)
            agent.uncertainty_threshold = unc_threshold

            env = PathTrackingEnv(max_episode_steps=1500,
                                  residual_injection=mode)

            zero_rets, mppi_rets = [], []
            all_early, all_fb = [], []
            all_ey, all_ep, all_th = [], [], []

            for ep in range(n_episodes):
                zero_r = run_episode(env, None, seed=ep)
                mppi_r = run_episode(env, agent, seed=ep)

                zero_rets.append(zero_r['return'])
                mppi_rets.append(mppi_r['return'])
                all_early.append(mppi_r['early_term'])
                all_fb.append(mppi_r['fb'])
                all_ey.append(mppi_r['ey_rms'])
                all_ep.append(mppi_r['epsi_rms'])
                all_th.append(mppi_r['theta_rms'])

            zero_ret = np.array(zero_rets)
            mppi_ret = np.array(mppi_rets)
            adv = mppi_ret - zero_ret

            fb_rate = float(np.mean([r.get('total_steps',0)-r.get('nonfallback',0)
                                      for r in all_fb])) / max(float(np.mean([r.get('total_steps',1) for r in all_fb])),1)
            nf = float(np.mean([r.get('nonfallback',0) for r in all_fb]))
            eps_mean = float(np.mean([r.get('eps_abs_mean',0) for r in all_fb]))
            eps_max = max(r.get('eps_abs_max',0) for r in all_fb)

            print(f"  {mode_labels[mode]:<20} | {np.mean(mppi_ret):>+8.2f} | "
                  f"{np.mean(adv):>+8.2f} | {int(np.sum(all_early)):>3} | "
                  f"{float(np.mean([r.get('total_steps',1) for r in all_fb])):>6.0f} | "
                  f"{np.mean(all_ey):>.4f} | {np.mean(all_ep):>.4f} | {np.mean(all_th):>.4f} | "
                  f"{fb_rate:>5.0%} | {nf:>5.1f} | {eps_mean:>.5f} | {eps_max:>.5f}")

            env.close()

    print(f"\n  DIAGNOSIS:")
    print(f"    Compare ey_rms across A/B/C to find best injection position.")
    print(f"    If B or C shows lower ey_rms with similar theta_rms, prefer reference-level injection.")

    return None


# ============================================================
# Main
# ============================================================
def main():
    sys.stdout.reconfigure(line_buffering=True)

    sindy_data = np.load(os.path.join(os.path.dirname(__file__), 'sindy_model_improved.npz'))
    sindy_Xi = sindy_data['coefficients']
    action_scale = float(sindy_data['action_scale'])

    ckpt_path = os.path.join(CKPT_DIR, 'checkpoint_ensemble_mppi.pt')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    unc_threshold = state_dict.get('uncertainty_threshold_p95', 0.01)

    print("=" * 100)
    print("  COMPREHENSIVE RESIDUAL DIAGNOSTIC")
    print(f"  uncertainty_threshold={unc_threshold:.6f}")
    print("=" * 100)

    # Phase 1
    phase1_margin_diagnostic(sindy_Xi, action_scale, unc_threshold, n_episodes=10)

    # Phase 2
    phase2_forced_residual(sindy_Xi, action_scale, unc_threshold, n_episodes=10)

    # Phase 3
    phase3_no_margin(sindy_Xi, action_scale, unc_threshold, n_episodes=10)

    # Phase 4
    phase4_injection_comparison(sindy_Xi, action_scale, unc_threshold, n_episodes=10)

    print("\n" + "=" * 100)
    print("  ALL PHASES COMPLETE")
    print("=" * 100)


if __name__ == '__main__':
    main()
