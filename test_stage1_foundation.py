import math

import numpy as np

from path_tracking_env import PathTrackingEnv


def test_residual_scale_is_explicit_and_reported():
    env = PathTrackingEnv(
        max_episode_steps=2,
        residual_injection="theta_target",
        residual_scale=0.2,
    )
    env.reset(seed=0)

    _, _, _, _, info = env.step(np.array([0.5], dtype=np.float32))

    assert math.isclose(info["residual_scale"], 0.2, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(info["epsilon"], 0.1, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(info["theta_target_delta"], 0.1, rel_tol=0.0, abs_tol=1e-12)
    env.close()


def test_step_reports_raw_and_clipped_theta_separately():
    env = PathTrackingEnv(max_episode_steps=2, residual_injection="stanley_ref")
    env.reset(seed=0)

    env._theta = 2.0
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))

    assert "theta_raw" in info
    assert "theta_clipped" in info
    assert "theta_raw_abs_exceeds_fall_limit" in info
    assert abs(info["theta_raw"]) >= abs(info["theta_clipped"])
    assert abs(info["theta_clipped"]) <= 1.57
    env.close()


def test_stanley_ref_reports_reference_authority_terms():
    env = PathTrackingEnv(
        max_episode_steps=2,
        residual_injection="stanley_ref",
        residual_scale=0.2,
    )
    env.reset(seed=1)

    _, _, _, _, info = env.step(np.array([0.5], dtype=np.float32))

    assert math.isclose(info["epsilon"], 0.1, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(info["delta_ref_delta"], 0.1, rel_tol=0.0, abs_tol=1e-12)
    assert "delta_stanley_base" in info
    assert "delta_ref" in info
    assert "theta_target_base" in info
    assert "theta_target_delta" in info
    env.close()


if __name__ == "__main__":
    test_residual_scale_is_explicit_and_reported()
    test_step_reports_raw_and_clipped_theta_separately()
    test_stanley_ref_reports_reference_authority_terms()
