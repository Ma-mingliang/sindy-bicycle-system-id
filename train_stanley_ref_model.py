"""Stage 3: Train ensemble world model on stanley_ref data.

Trains 3-member ensemble with bootstrap sampling.
Evaluates model quality and saves checkpoint with schema.
"""
import sys, os
from datetime import datetime
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from residual_mppi import EnsembleWorldModel


def main():
    sys.stdout.reconfigure(line_buffering=True)

    # Parse dataset size from command line
    dataset_size = sys.argv[1] if len(sys.argv) > 1 else '30k'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load SINDy model
    sindy_data = np.load(os.path.join(os.path.dirname(__file__), 'sindy_model_improved.npz'))
    sindy_Xi = sindy_data['coefficients']
    action_scale = float(sindy_data['action_scale'])

    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), 'data', f'stanley_ref_dataset_{dataset_size}.npz')
    data = np.load(data_path)
    obs = torch.tensor(data['obs'], dtype=torch.float32)
    action = torch.tensor(data['action'], dtype=torch.float32)
    next_obs = torch.tensor(data['next_obs'], dtype=torch.float32)

    print("=" * 80)
    print(f"  Stage 3: Training Ensemble World Model (stanley_ref)")
    print(f"  Dataset: {dataset_size} ({len(obs)} samples)")
    print(f"  Timestamp: {timestamp}")
    print("=" * 80)

    # Train ensemble
    ensemble = EnsembleWorldModel(
        obs_dim=8, action_dim=1, mlp_dim=256,
        sindy_Xi=sindy_Xi, action_scale=action_scale,
        ensemble_size=3,
    )

    from residual_mppi import ConservativeEnsembleResidualMPPI
    cfg = {
        'obs_dim': 8, 'action_dim': 1, 'mlp_dim': 256,
        'ensemble_size': 3,
    }
    agent = ConservativeEnsembleResidualMPPI(
        cfg, sindy_Xi, action_scale, obs_dim=8, action_dim=1)

    # Create replay buffer
    class SimpleBuffer:
        def __init__(self, obs, action, next_obs):
            self.obs = obs.numpy()
            self.action = action.numpy()
            self.next_obs = next_obs.numpy()
            self.size = len(obs)

    buffer = SimpleBuffer(obs, action, next_obs)

    # Train
    print("\n  Training ensemble...")
    results = agent.train_world_model(buffer, epochs=50, batch_size=256, val_ratio=0.2)

    # Model errors
    print("\n  Computing model errors...")
    errors = agent.compute_model_errors(buffer)
    print(f"    SINDy error: {errors['sindy_error']:.6f}")
    print(f"    Model error: {errors['model_error']:.6f}")
    print(f"    Model/SINDy ratio: {errors['model_error']/max(errors['sindy_error'],1e-8):.3f}")

    # Uncertainty percentiles
    print("\n  Computing uncertainty percentiles...")
    val_unc = agent.compute_val_uncertainty_percentiles(buffer, val_ratio=0.2)
    print(f"    Val p50: {val_unc['p50']:.6f}")
    print(f"    Val p90: {val_unc['p90']:.6f}")
    print(f"    Val p95: {val_unc['p95']:.6f}")
    print(f"    Val p99: {val_unc['p99']:.6f}")

    # Rollout validation
    print("\n  Validating rollout...")
    rollout = agent.validate_rollout(buffer, horizon=5, n_samples=100)
    print(f"    One-step error: {rollout['one_step_error']:.6f}")
    print(f"    5-step error: {rollout['multi_step_error'][-1]:.6f}")
    print(f"    Diverges: {rollout['diverges']}")

    # Save checkpoint
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'ensemble_{dataset_size}.pt')

    agent.save(ckpt_path, extra_meta={
        'injection_mode': 'stanley_ref',
        'train_data_source': f'stanley_ref_dataset_{dataset_size}.npz',
        'reward_version': 'hrrl_v1',
        'created_at': timestamp,
        'dataset_size': dataset_size,
        'n_samples': len(obs),
        'sindy_error': errors['sindy_error'],
        'model_error': errors['model_error'],
        'val_uncertainty_p95': val_unc['p95'],
        'rollout_h5_error': float(rollout['multi_step_error'][-1]),
        'rollout_diverges': rollout['diverges'],
    })

    print(f"\n  Saved: {ckpt_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
