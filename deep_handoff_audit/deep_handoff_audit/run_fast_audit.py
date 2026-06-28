"""Fast GP+Ensemble audit with reduced samples.

Uses fewer GP samples and shorter evaluation to complete in reasonable time.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import json
import time
import torch


def run_fast_audit():
    """Run fast GP+Ensemble evaluation."""
    print("=" * 70)
    print("FAST GP+ENSEMBLE AUDIT")
    print("=" * 70)

    from methods_common import generate_training_data, real_step, K_lqr, dt
    from methods_classic import GPMethod
    from methods_nn import ResidualNet
    from methods_evaluate import make_tau_func

    # Get normalization constants
    states, actions, deltas, state_std, action_std, delta_std = generate_training_data(5000, seed=42)

    print(f"\nNormalization constants:")
    print(f"  state_std: {state_std}")
    print(f"  action_std: {action_std:.4f}")
    print(f"  delta_std: {delta_std}")

    # 1. Train GP baseline with fewer samples
    print("\n1. Training GP baseline (5000 samples)...")
    t0 = time.time()
    gp = GPMethod()
    gp.max_samples = 2000  # Reduce for speed
    gp.train(states, actions, deltas, state_std, action_std, delta_std)
    print(f"   GP trained in {time.time()-t0:.1f}s")

    # 2. Compute GP residuals
    print("\n2. Computing GP residuals...")
    gp_deltas = []
    for i in range(len(states)):
        s_next_gp = gp.predict(states[i], actions[i])
        gp_delta = s_next_gp - states[i]
        gp_deltas.append(gp_delta)
    gp_deltas = np.array(gp_deltas)
    residuals = (deltas - gp_deltas) / delta_std
    print(f"   Residual mean: {residuals.mean(axis=0)}")
    print(f"   Residual std: {residuals.std(axis=0)}")

    # 3. Train ensemble of NNs
    print("\n3. Training ensemble of NNs...")
    n_models = 5
    nn_models = []
    for i in range(n_models):
        torch.manual_seed(i * 42)
        np.random.seed(i * 42)
        model = ResidualNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        X = torch.FloatTensor(np.column_stack([
            states / state_std,
            actions / action_std
        ]))
        Y = torch.FloatTensor(residuals)

        model.train()
        for epoch in range(50):
            idx = np.random.permutation(len(X))
            for batch_start in range(0, len(X), 256):
                batch_idx = idx[batch_start:batch_start+256]
                x_batch = X[batch_idx]
                y_batch = Y[batch_idx]
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = torch.nn.functional.mse_loss(pred, y_batch)
                loss.backward()
                optimizer.step()

        nn_models.append(model)
        print(f"   Model {i} trained")

    # 4. Define ensemble prediction
    residual_scale = 0.3

    def ensemble_predict(s, tau):
        s_next_gp = gp.predict(s, tau)
        s_norm = s / state_std
        a_norm = tau / action_std
        x = torch.FloatTensor(np.concatenate([s_norm, [a_norm]]))

        nn_residuals = []
        for model in nn_models:
            model.eval()
            with torch.no_grad():
                nn_res = model(x).numpy()
            nn_residuals.append(nn_res)

        mean_residual = np.mean(nn_residuals, axis=0)
        s_next = s_next_gp + mean_residual * delta_std * residual_scale
        return s_next

    # 5. Run evaluation with fewer steps
    print("\n4. Running evaluations (200 steps, 3 segments)...")
    n_runs = 3
    n_segments = 3
    n_steps = 200

    all_results = []
    for run_i in range(n_runs):
        run_seed = 42 + run_i
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        segment_maes = []
        for seg_i in range(n_segments):
            tau_func = make_tau_func(seg_i, n_steps)
            rng = np.random.RandomState(run_seed * 100 + seg_i)
            phi_init = rng.uniform(-0.25, 0.25)
            s0 = np.array([phi_init, 0.0, 0.0, 0.0])

            s_real = s0.copy()
            s_model = s0.copy()
            phi_errors = []

            for step in range(n_steps):
                tau = tau_func(step, s_real)
                s_real = real_step(s_real, tau)
                s_model = ensemble_predict(s_model, tau)

                if abs(s_real[0]) > math.pi / 3:
                    break

                phi_errors.append(abs(s_model[0] - s_real[0]))

            if phi_errors:
                segment_maes.append(np.mean(phi_errors))

        if segment_maes:
            run_mae = np.mean(segment_maes)
        else:
            run_mae = float('nan')

        all_results.append({
            'run': run_i,
            'seed': run_seed,
            'config': f'{n_models} models, 0 DAgger, scale={residual_scale}',
            'mae_rad': float(run_mae),
            'segment_maes': [float(m) for m in segment_maes],
        })
        print(f"   Run {run_i}: seed={run_seed}, MAE={run_mae:.4f} rad")

    # 6. Run with DAgger
    print("\n5. Running with DAgger (2 rounds)...")
    dagger_rounds = 2
    for round_i in range(dagger_rounds):
        print(f"\n   DAgger round {round_i+1}/{dagger_rounds}:")
        dagger_states = []
        dagger_actions = []
        dagger_deltas = []

        for seg_i in range(2):
            tau_func = make_tau_func(100 + round_i * 2 + seg_i, n_steps)
            phi_init = np.random.uniform(-0.25, 0.25)
            s = np.array([phi_init, 0.0, 0.0, 0.0])

            for step in range(n_steps):
                tau = tau_func(step, s)
                s_real = real_step(s, tau)
                delta_real = s_real - s

                dagger_states.append(s.copy())
                dagger_actions.append(tau)
                dagger_deltas.append(delta_real)

                s = s_real
                if abs(s[0]) > math.pi / 3:
                    break

        dagger_states = np.array(dagger_states)
        dagger_actions = np.array(dagger_actions)
        dagger_deltas = np.array(dagger_deltas)

        print(f"   Collected {len(dagger_states)} DAgger samples")

        # Compute GP residuals for DAgger data
        gp_deltas_d = []
        for i in range(len(dagger_states)):
            s_next_gp = gp.predict(dagger_states[i], dagger_actions[i])
            gp_delta = s_next_gp - dagger_states[i]
            gp_deltas_d.append(gp_delta)
        gp_deltas_d = np.array(gp_deltas_d)
        residuals_d = (dagger_deltas - gp_deltas_d) / delta_std

        # Combine with original data
        all_states = np.vstack([states, dagger_states])
        all_actions = np.concatenate([actions, dagger_actions])
        all_residuals = np.vstack([residuals, residuals_d])

        X_aug = torch.FloatTensor(np.column_stack([
            all_states / state_std,
            all_actions / action_std
        ]))
        Y_aug = torch.FloatTensor(all_residuals)

        for i, model in enumerate(nn_models):
            torch.manual_seed(i * 42 + round_i * 1000)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model.train()
            for epoch in range(30):
                idx = np.random.permutation(len(X_aug))
                for batch_start in range(0, len(X_aug), 256):
                    batch_idx = idx[batch_start:batch_start+256]
                    x_batch = X_aug[batch_idx]
                    y_batch = Y_aug[batch_idx]
                    optimizer.zero_grad()
                    pred = model(x_batch)
                    loss = torch.nn.functional.mse_loss(pred, y_batch)
                    loss.backward()
                    optimizer.step()

        print(f"   Models retrained")

    # 7. Final evaluation after DAgger
    print("\n6. Final evaluation after DAgger...")
    final_results = []
    for run_i in range(n_runs):
        run_seed = 42 + run_i
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        segment_maes = []
        for seg_i in range(n_segments):
            tau_func = make_tau_func(seg_i, n_steps)
            rng = np.random.RandomState(run_seed * 100 + seg_i)
            phi_init = rng.uniform(-0.25, 0.25)
            s0 = np.array([phi_init, 0.0, 0.0, 0.0])

            s_real = s0.copy()
            s_model = s0.copy()
            phi_errors = []

            for step in range(n_steps):
                tau = tau_func(step, s_real)
                s_real = real_step(s_real, tau)
                s_model = ensemble_predict(s_model, tau)

                if abs(s_real[0]) > math.pi / 3:
                    break

                phi_errors.append(abs(s_model[0] - s_real[0]))

            if phi_errors:
                segment_maes.append(np.mean(phi_errors))

        if segment_maes:
            run_mae = np.mean(segment_maes)
        else:
            run_mae = float('nan')

        final_results.append({
            'run': run_i,
            'seed': run_seed,
            'config': f'{n_models} models, {dagger_rounds} DAgger, scale={residual_scale}',
            'mae_rad': float(run_mae),
            'segment_maes': [float(m) for m in segment_maes],
        })
        print(f"   Run {run_i}: seed={run_seed}, MAE={run_mae:.4f} rad")

    # Statistics
    maes = [r['mae_rad'] for r in final_results if not np.isnan(r['mae_rad'])]
    stats = {
        'n_runs': len(maes),
        'mean_rad': float(np.mean(maes)) if maes else float('nan'),
        'std_rad': float(np.std(maes)) if maes else float('nan'),
        'min_rad': float(np.min(maes)) if maes else float('nan'),
        'max_rad': float(np.max(maes)) if maes else float('nan'),
    }

    print(f"\n7. FINAL STATISTICS ({n_models} models, {dagger_rounds} DAgger):")
    print(f"   N runs: {stats['n_runs']}")
    print(f"   Mean: {stats['mean_rad']:.4f} rad")
    print(f"   Std: {stats['std_rad']:.4f} rad")
    print(f"   Min: {stats['min_rad']:.4f} rad")
    print(f"   Max: {stats['max_rad']:.4f} rad")

    return {
        'gp_plus_ensemble_no_dagger': {'runs': all_results},
        'gp_plus_ensemble_with_dagger': {'runs': final_results, 'statistics': stats},
    }


if __name__ == '__main__':
    results = run_fast_audit()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fast_audit_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
