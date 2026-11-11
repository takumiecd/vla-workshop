"""Check split integrity, paired comparisons, and interrupted simulation recording."""

import copy
import json

import numpy as np
import pytest
from vla_workshop.course import common, libero
from vla_workshop.course.compare import compare_reports
from vla_workshop.evaluation.rollout import HoldPolicy


def specification():
    return {
        "suite": "libero_spatial",
        "task_ids": [0],
        "max_steps": 2,
        "settle_steps": 0,
        "seed": 0,
        "renderer": "osmesa",
        "policy": {},
        "initial_states": {"validation": [0, 1], "test": [10], "collection": [20]},
    }


def test_disjoint_splits_reject_leakage_and_invalid_ids():
    common.episode_split({"train_episodes": [0, 1], "validation_episodes": [2]})
    for train, validation in [([0], [0]), ([True], [1]), ([], [1]), ([0, 0], [1])]:
        with pytest.raises(ValueError):
            common.episode_split({"train_episodes": train, "validation_episodes": validation})
    config = specification()
    assert libero.evaluation_cases(config, "validation") == [(0, 0), (0, 1)]
    config["initial_states"]["collection"] = [0]
    with pytest.raises(ValueError, match="overlap"):
        libero.evaluation_cases(config, "validation")


def observation():
    return {
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
        "agentview_image": np.zeros((16, 16, 3), np.uint8),
    }


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_records_completed_cases_and_closes_on_failure(tmp_path, monkeypatch, fail):
    monkeypatch.setattr(common, "PROJECT", tmp_path)
    instances = []

    class FakeEnvironment:
        def __init__(self, config, task, initial, seed):
            self.obs = observation()
            self.low, self.high = -np.ones(7), np.ones(7)
            self.instruction = "pick up the cube"
            self.initial_state_hash = f"initial-{initial}"
            self.closed = False
            instances.append(self)

        def step(self, action):
            if fail:
                raise RuntimeError("simulator failed")
            return self.obs, True, True

        def close(self):
            self.closed = True

    if fail:
        with pytest.raises(RuntimeError, match="simulator failed"):
            libero.evaluate(specification(), policy=HoldPolicy(), environment=FakeEnvironment)
        folder = next((tmp_path / "outputs").iterdir())
    else:
        folder = libero.evaluate(specification(), policy=HoldPolicy(), environment=FakeEnvironment)
    report = json.loads((folder / "report.json").read_text())
    assert all(instance.closed for instance in instances)
    assert report["status"] == ("failed" if fail else "completed")
    if fail:
        assert "success_rate" not in report
    else:
        assert report["success_rate"] == 1 and len(report["episodes"]) == 2
        assert all(row["steps"] == 1 for row in report["episodes"])
        for log in folder.glob("task_*/steps.jsonl"):
            row = json.loads(log.read_text())
            assert len(row["state_before"]) == 8 and len(row["env_action"]) == 7


def report():
    return {
        "status": "completed",
        "conditions_sha256": "same",
        "environment_code_sha256": "same-environment-code",
        "split": "validation",
        "assets_revision": "fixed",
        "versions": {"mujoco": "3.3.2"},
        "episodes": [
            {
                "task_id": 0,
                "initial_state": 0,
                "initial_state_sha256": "pose",
                "instruction": "task",
                "seed": 0,
                "success": False,
            }
        ],
    }


def test_comparison_pairs_the_same_physical_initial_states():
    baseline, candidate = report(), report()
    candidate["episodes"][0]["success"] = True
    assert compare_reports(baseline, candidate)["difference_percentage_points"] == 100
    for field, value in [
        ("seed", 1),
        ("initial_state_sha256", "different"),
        ("instruction", "other"),
    ]:
        changed = copy.deepcopy(candidate)
        changed["episodes"][0][field] = value
        with pytest.raises(ValueError, match="differs"):
            compare_reports(baseline, changed)
    for field, value in [("status", "failed"), ("split", "test"), ("episodes", [])]:
        changed = copy.deepcopy(candidate)
        changed[field] = value
        with pytest.raises(ValueError):
            compare_reports(baseline, changed)
    candidate["episodes"][0]["success"] = "false"
    with pytest.raises(ValueError, match="booleans"):
        compare_reports(baseline, candidate)


def test_optimizer_updates_finite_loss_only():
    torch = pytest.importorskip("torch")
    from vla_workshop.course.train import optimizer_update

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, batch):
            return self.weight.square() * batch["scale"], {}

    policy = TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    optimizer_update(policy, lambda x: x, {"scale": torch.tensor(1.0)}, optimizer, 10.0)
    assert policy.weight.item() == pytest.approx(0.8)
    with pytest.raises(ValueError, match="non-finite"):
        optimizer_update(
            policy, lambda x: x, {"scale": torch.tensor(float("nan"))}, optimizer, 10.0
        )
    assert policy.weight.item() == pytest.approx(0.8)
