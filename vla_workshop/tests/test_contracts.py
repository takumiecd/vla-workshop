"""Regression checks for boundaries where a silent mismatch breaks robot learning."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vla_workshop.config import EnvConfig, PolicyConfig, RunConfig, load_config
from vla_workshop.envs.robosuite_env import state_vector, validate_action
from vla_workshop.evaluation import rollout
from vla_workshop.policies.smolvla import transform_image, validate_features


def observation():
    return {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.9]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.02, -0.02]),
        "agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
    }


@pytest.mark.parametrize("angle", [0.0, np.pi / 2, np.pi, 1.5 * np.pi])
def test_state_preserves_pose_and_libero_axis_angle_convention(angle):
    obs = observation()
    obs["robot0_eef_quat"] = np.array([np.sin(angle / 2), 0.0, 0.0, np.cos(angle / 2)])
    state = state_vector(obs)
    np.testing.assert_allclose(state[:3], obs["robot0_eef_pos"])
    np.testing.assert_allclose(state[3:6], [angle, 0, 0], atol=1e-6)
    np.testing.assert_allclose(
        Rotation.from_rotvec(state[3:6]).as_matrix(),
        Rotation.from_quat(obs["robot0_eef_quat"]).as_matrix(),
        atol=1e-6,
    )
    np.testing.assert_allclose(state[-2:], obs["robot0_gripper_qpos"])


def test_image_rotation_keeps_rgb_channels_and_pixel_scale():
    image = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    result = transform_image(image, "rotate180")
    np.testing.assert_array_equal(result[:, 0, 0], [1, 1, 1])
    np.testing.assert_array_equal(result[:, 1, 1], [1, 0, 0])
    assert result.shape == (3, 2, 2)
    assert result.dtype == np.float32 and result.flags.c_contiguous


@pytest.mark.parametrize("action", [np.zeros(6), np.zeros((1, 7)), np.full(7, np.nan)])
def test_invalid_actions_are_rejected_before_simulation(action):
    with pytest.raises(ValueError, match="finite action"):
        validate_action(action, -np.ones(7), np.ones(7))


def test_action_bounds_are_explicit():
    action = np.array([2, -2, 0, 0.5, 0, 0, 1.5])
    np.testing.assert_array_equal(
        validate_action(action, -np.ones(7), np.ones(7)),
        [1, -1, 0, 0.5, 0, 0, 1],
    )


def test_incompatible_base_checkpoint_is_rejected_without_padding():
    policy_config = SimpleNamespace(
        input_features={"observation.state": SimpleNamespace(shape=(6,))},
        output_features={"action": SimpleNamespace(shape=(6,))},
    )
    with pytest.raises(ValueError, match="feature mismatch"):
        validate_features(policy_config, {}, 256)


def test_output_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "configs" / "lift.json"
    monkeypatch.chdir(tmp_path)
    config = load_config(source)
    assert Path(config.output_dir) == source.parent.parent / "outputs"


@pytest.mark.parametrize("fail", [False, True])
def test_rollout_records_termination_and_closes_environment(tmp_path, monkeypatch, fail):
    instances = []

    class FakeEnv:
        def __init__(self, config, seed, max_steps):
            self.config, self.seed = config, seed
            self.low, self.high = -np.ones(7), np.ones(7)
            self.closed = False
            instances.append(self)

        def reset(self):
            return observation()

        def step(self, action):
            return observation(), 1.0, False, {"success": True}

        def video_frame(self, obs):
            return obs["agentview_image"]

        def describe(self):
            return {"seed": self.seed}

        def save_initial_state(self, folder, obs):
            np.savez_compressed(folder / "initial_observation.npz", **obs)

        def close(self):
            self.closed = True

    class TestPolicy(rollout.HoldPolicy):
        resets = 0

        def reset(self, *, seed):
            self.resets += 1

        def select_action(self, obs, instruction):
            return np.full(7, np.nan) if fail else super().select_action(obs, instruction)

    monkeypatch.setattr(rollout, "RobosuiteEnv", FakeEnv)
    config = RunConfig(
        env=EnvConfig(cameras=["agentview"], image_size=16),
        policy=PolicyConfig(path="unused"),
        output_dir=str(tmp_path),
        episodes=2,
    )
    policy = TestPolicy()
    if fail:
        with pytest.raises(ValueError):
            rollout.run(config, policy, "test")
        result = next(tmp_path.iterdir())
    else:
        result = rollout.run(config, policy, "test")
        assert policy.resets == 2
        steps = [
            json.loads(line)
            for line in (result / "episode_000/steps.jsonl").read_text().splitlines()
        ]
        assert len(steps) == 1 and steps[0]["success"]
        assert len(steps[0]["state_before"]) == 8
        assert (result / "episode_000/video.mp4").stat().st_size > 0
    summary = json.loads((result / "summary.json").read_text())
    assert summary["status"] == ("failed" if fail else "completed")
    assert all(instance.closed for instance in instances)
