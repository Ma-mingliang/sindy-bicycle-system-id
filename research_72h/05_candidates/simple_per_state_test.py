"""Simple Per-State Test - Quick validation of the approach."""
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'D:/系统辨识作业/sindy_bicycle')

STATE_NAMES_7D = ['e_y', 'e_psi', 'v', 'theta', 'theta_dot', 'delta', 'delta_dot']
STATE_DIM = 7
ACTION_DIM = 1
IDX_7D_FROM_8D = [0, 1, 2, 3, 4, 6, 7]
DT = 1.0 / 30.0


class SimpleODEFunc(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim),
        )
        nn.init.zeros_(self.net[-1].bias)
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)

    def forward(self, x):
        return self.net(x)


def load_data():
    data = np.load('D:/系统辨识作业/sindy_bicycle/data/stage2_dataset_150k.npz', allow_pickle=True)
    obs = data['obs'][:, IDX_7D_FROM_8D]
    next_obs = data['next_obs'][:, IDX_7D_FROM_8D]
    deltas = next_obs - obs
    action = data['action']

    state_std = np.std(obs, axis=0)
    state_std[state_std < 1e-10] = 1.0
    action_std = np.std(action)
    if action_std < 1e-10:
        action_std = 1.0
    delta_std = np.std(deltas, axis=0)
    delta_std[delta_std < 1e-10] = 1.0

    return obs, action, deltas, state_std, action_std, delta_std


def main():
    print("=" * 60)
    print("Simple Per-State Test")
    print("=" * 60)

    print("\n1. Loading data...")
    obs, action, deltas, state_std, action_std, delta_std = load_data()
    print(f"   Data shape: {obs.shape}")
    print(f"   State std: {state_std}")

    # Prepare input
    X = np.hstack([obs / state_std, action.reshape(-1, 1) / action_std])
    print(f"   Input shape: {X.shape}")

    # Test GP for e_y
    print("\n2. Testing GP for e_y...")
    Y_ey = deltas[:, 0] / delta_std[0]

    # Sample subset
    n_samples = 1000
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_sub = X[indices]
    Y_sub = Y_ey[indices]

    # Normalize
    X_mean = X_sub.mean(axis=0)
    X_std = X_sub.std(axis=0) + 1e-8
    X_scaled = (X_sub - X_mean) / X_std

    kernel = Matern(nu=2.5, length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-3)
    gp.fit(X_scaled, Y_sub)
    print(f"   GP trained successfully")

    # Test prediction
    X_test = X_scaled[:10]
    Y_pred = gp.predict(X_test)
    Y_true = Y_sub[:10]
    print(f"   Prediction sample: {Y_pred[:5]}")
    print(f"   True sample: {Y_true[:5]}")
    print(f"   MSE: {np.mean((Y_pred - Y_true)**2):.6f}")

    # Test NODE for e_y
    print("\n3. Testing NODE for e_y...")
    model = SimpleODEFunc(STATE_DIM + ACTION_DIM, 1, hidden=64)
    X_torch = torch.FloatTensor(X_sub)
    Y_torch = torch.FloatTensor(Y_sub).unsqueeze(1)

    # Simple training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        pred = model(X_torch)
        loss = nn.functional.mse_loss(pred, Y_torch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}: loss={loss.item():.6f}")

    print(f"   NODE trained successfully")

    print("\n4. Summary:")
    print("   - GP and NODE can both be trained for e_y")
    print("   - Ready to implement full per-state hybrid model")
    print("   - Key insight: Use physics for e_y, e_psi; NODE/GP for others")


if __name__ == '__main__':
    main()
