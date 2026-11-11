"""Controller settings and state conversion for the Lift exercise."""

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


def lift_controller() -> dict:
    """Configure world-frame OSC_POSE with the limits used in the exercise."""
    from robosuite.controllers import load_composite_controller_config

    controller = load_composite_controller_config(controller="BASIC")
    controller["body_parts"]["right"].update(
        type="OSC_POSE",
        input_type="delta",
        input_ref_frame="world",
        impedance_mode="fixed",
        input_min=-1,
        input_max=1,
        output_min=[-0.05] * 3 + [-0.5] * 3,
        output_max=[0.05] * 3 + [0.5] * 3,
    )
    return controller
