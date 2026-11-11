"""Convert robot observations to the 8D state and validate 7D actions."""

import numpy as np


def state_vector(obs: dict[str, np.ndarray]) -> np.ndarray:
    """EEF xyz [m], axis-angle [rad], and two gripper joint positions [m]."""
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-8:
        raise ValueError("Expected a finite xyzw end-effector quaternion")
    # Match the LIBERO processor's xyzw -> axis-angle convention (including w < 0).
    quat = quat / np.linalg.norm(quat)
    vector_norm = np.linalg.norm(quat[:3])
    if vector_norm < 1e-10:
        rotvec = np.zeros(3)
    else:
        angle = 2 * np.arctan2(vector_norm, quat[3])
        rotvec = quat[:3] * (angle / vector_norm)
    state = np.concatenate((obs["robot0_eef_pos"], rotvec, obs["robot0_gripper_qpos"]))
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(f"Expected finite state shape (8,), got {state.shape}")
    return state.astype(np.float32)


def validate_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError(f"Expected finite action shape (7,), got {action.shape}")
    return np.clip(action, -1.0, 1.0)
