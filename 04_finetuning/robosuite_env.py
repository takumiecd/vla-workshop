"""Standard robosuite Lift/Panda environment with an explicit 7D control contract."""

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from config import EnvConfig


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


def validate_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError(f"Expected finite action shape (7,), got {action.shape}")
    return np.clip(action, low, high)


class RobosuiteEnv:
    def __init__(self, config: EnvConfig, seed: int, max_steps: int) -> None:
        # Set the backend before importing robosuite / MuJoCo / PyOpenGL.
        os.environ["MUJOCO_GL"] = config.renderer
        if config.renderer in {"egl", "osmesa"}:
            os.environ["PYOPENGL_PLATFORM"] = config.renderer
        import robosuite as suite
        from robosuite import macros
        from robosuite.controllers import load_composite_controller_config

        # Keep raw observations in the documented OpenGL convention on every machine.
        macros.IMAGE_CONVENTION = "opengl"
        controller = load_composite_controller_config(controller="BASIC")
        arm = controller["body_parts"]["right"]
        arm.update(
            type="OSC_POSE",
            input_type="delta",
            input_ref_frame="world",
            impedance_mode="fixed",
            input_min=-1,
            input_max=1,
            output_min=[-0.05] * 3 + [-0.5] * 3,
            output_max=[0.05] * 3 + [0.5] * 3,
        )
        self.config = config
        self.controller_config = controller
        self.env = suite.make(
            env_name=config.task,
            robots=config.robot,
            controller_configs=controller,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=False,
            camera_names=config.cameras,
            camera_heights=config.image_size,
            camera_widths=config.image_size,
            camera_depths=False,
            control_freq=config.control_freq,
            horizon=max_steps + config.settle_steps,
            ignore_done=False,
            hard_reset=False,
            reward_shaping=False,
            seed=seed,
        )
        self.low, self.high = self.env.action_spec
        if self.low.shape != (7,):
            self.close()
            raise ValueError(f"Expected a 7D OSC_POSE action space, got {self.low.shape}")

    def reset(self) -> dict[str, np.ndarray]:
        obs = self.env.reset()
        for _ in range(self.config.settle_steps):
            obs, _, _, _ = self.env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
        return obs

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        action = validate_action(action, self.low, self.high)
        obs, reward, done, info = self.env.step(action)
        # Success stays on the evaluation side; it is never a policy input.
        info = dict(info, success=bool(self.env._check_success()))
        return obs, float(reward), bool(done), info

    def video_frame(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.ascontiguousarray(obs[f"{self.config.cameras[0]}_image"][::-1])

    def describe(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "controller": self.controller_config,
            "action_low": self.low.tolist(),
            "action_high": self.high.tolist(),
            "state_order": [
                "eef_x",
                "eef_y",
                "eef_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "gripper_left",
                "gripper_right",
            ],
            "action_order": ["dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "gripper"],
            "action_semantics": "world-frame normalized OSC_POSE delta; gripper -1 open, +1 close",
            "raw_image_convention": "opengl; video vertically flipped; policy transform in config",
        }

    def save_initial_state(self, folder: Path, obs: dict[str, np.ndarray]) -> None:
        """Save raw observations and simulator state for debugging, outside the policy."""
        (folder / "model.xml").write_text(self.env.sim.model.get_xml())
        np.savez_compressed(
            folder / "initial_observation.npz",
            **obs,
            sim_state=self.env.sim.get_state().flatten(),
        )

    def close(self) -> None:
        self.env.close()
