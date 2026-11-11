"""Pinned LIBERO scenes and reproducible, sequential evaluation."""

import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from vla_workshop.config import PolicyConfig
from vla_workshop.envs.robosuite_env import state_vector, validate_action
from vla_workshop.evaluation.rollout import package_versions, source_metadata
from vla_workshop.policies.smolvla import SmolVLA

from .common import PROJECT, digest, output_directory, write_json

ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"


def prepare_libero(renderer: str = "egl") -> Path:
    """Prepare an isolated config before LIBERO can prompt, and pin scene assets."""
    from huggingface_hub import snapshot_download

    if renderer not in {"egl", "osmesa"}:
        raise ValueError("LIBERO renderer must be egl or osmesa")
    # The isolated interpreter cannot use the notebook kernel's inline backend.
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MUJOCO_GL"] = renderer
    os.environ["PYOPENGL_PLATFORM"] = renderer
    spec = importlib.util.find_spec("libero")
    if spec is None:
        raise ImportError("Use the workshop LIBERO runtime")
    package = Path(next(iter(spec.submodule_search_locations))) / "libero"
    config_dir = PROJECT / "outputs" / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    assets = Path(
        snapshot_download(
            "lerobot/libero-assets",
            repo_type="dataset",
            revision=ASSETS_REVISION,
        )
    )
    # JSON is valid YAML. Create this before importing libero.libero to avoid input().
    write_json(
        config_dir / "config.yaml",
        {
            "benchmark_root": str(package),
            "bddl_files": str(package / "bddl_files"),
            "init_states": str(package / "init_files"),
            "datasets": str(PROJECT / "datasets"),
            "assets": str(assets),
        },
    )
    import libero.libero as library

    # hf-libero 0.1.4 otherwise fetches unversioned assets from its helper.
    library._assets_path_cache = str(assets)
    return assets


class LiberoEnvironment:
    """Expose raw camera/state observations to the same SmolVLA adapter as lesson 01."""

    def __init__(self, config: dict, task_id: int, initial_state: int, seed: int):
        prepare_libero(config["renderer"])
        from lerobot.sim.integrations.libero import get_task_init_states
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suites = benchmark.get_benchmark_dict()
        if config["suite"] not in suites:
            raise ValueError(f"Unknown suite: {config['suite']}")
        suite = suites[config["suite"]]()
        if not 0 <= task_id < suite.n_tasks:
            raise ValueError(f"Invalid task id: {task_id}")
        states = get_task_init_states(suite, task_id)
        if not 0 <= initial_state < len(states):
            raise ValueError(f"Invalid initial state: {initial_state}")
        task = suite.get_task(task_id)
        self.instruction = task.language
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(
                Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            ),
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
        )
        try:
            self.env.seed(seed)
            self.env.reset()
            self.obs = self.env.set_init_state(states[initial_state])
            for _ in range(config["settle_steps"]):
                self.obs, _, _, _ = self.env.step([0, 0, 0, 0, 0, 0, -1])
            self.initial_state_hash = hashlib.sha256(
                np.asarray(self.env.sim.get_state().flatten()).tobytes()
            ).hexdigest()
            self.low, self.high = self.env.env.action_spec
            if self.low.shape != (7,):
                raise ValueError("Expected 7D LIBERO actions")
        except BaseException:
            self.close()
            raise

    def step(self, action: np.ndarray) -> tuple[dict, bool, bool]:
        action = validate_action(action, self.low, self.high)
        self.obs, _, done, _ = self.env.step(action)
        return self.obs, bool(self.env.check_success()), bool(done)

    def close(self) -> None:
        self.env.close()


def evaluation_cases(config: dict, split: str) -> list[tuple[int, int]]:
    states = config["initial_states"]
    if split not in {"validation", "test", "collection"}:
        raise ValueError(f"Unknown split: {split}")
    for name in ["validation", "test", "collection"]:
        values = states[name]
        if (
            not values
            or len(set(values)) != len(values)
            or any(type(i) is not int or i < 0 for i in values)
        ):
            raise ValueError(f"Invalid {name} initial states")
    for left, right in [
        ("validation", "test"),
        ("validation", "collection"),
        ("test", "collection"),
    ]:
        if set(states[left]) & set(states[right]):
            raise ValueError(f"Initial states overlap: {left}, {right}")
    tasks = config["task_ids"]
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or any(type(i) is not int or i < 0 for i in tasks)
    ):
        raise ValueError("Task ids must be nonempty, unique and nonnegative")
    if config["max_steps"] < 1 or config["settle_steps"] < 0:
        raise ValueError("Invalid step limit")
    return [(task, initial) for task in tasks for initial in states[split]]


def evaluate(
    config: dict, split: str = "validation", policy=None, environment=LiberoEnvironment
) -> Path:
    """Run all configured cases; failed executions never become a success-rate report."""
    cases = evaluation_cases(config, split)
    policy = policy if policy is not None else SmolVLA(PolicyConfig(**config["policy"]), 256)
    folder = output_directory("libero_" + split)
    conditions = {key: value for key, value in config.items() if key != "policy"}
    report = {
        "status": "running",
        "split": split,
        "config": config,
        "conditions_sha256": digest(conditions),
        "environment_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assets_revision": ASSETS_REVISION,
        "policy": policy.metadata,
        "evaluation_scope": "Selected LIBERO tasks and initial states; not a full benchmark score",
        "versions": package_versions(),
        "source": source_metadata(),
        "episodes": [],
    }
    write_json(folder / "report.json", report)
    try:
        for task, initial in cases:
            seed = config["seed"] + task * 10000 + initial
            np.random.seed(seed)
            env = environment(config, task, initial, seed)
            try:
                policy.reset(seed=seed)
                episode_dir = folder / f"task_{task:02d}_init_{initial:03d}"
                episode_dir.mkdir()
                success = False
                started = time.perf_counter()
                with (
                    imageio.get_writer(episode_dir / "video.mp4", fps=20) as video,
                    (episode_dir / "steps.jsonl").open("w") as log,
                ):
                    video.append_data(env.obs["agentview_image"][::-1, ::-1])
                    for step in range(config["max_steps"]):
                        state = state_vector(env.obs)
                        before_inference = time.perf_counter()
                        raw = policy.select_action(env.obs, env.instruction)
                        policy_seconds = time.perf_counter() - before_inference
                        action = validate_action(raw, env.low, env.high)
                        _, success, done = env.step(action)
                        log.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "policy_seconds": policy_seconds,
                                    "state_before": state.tolist(),
                                    "policy_action": raw.tolist(),
                                    "env_action": action.tolist(),
                                    "success": success,
                                    "done": done,
                                }
                            )
                            + "\n"
                        )
                        log.flush()
                        video.append_data(env.obs["agentview_image"][::-1, ::-1])
                        if step % 20 == 0:
                            print(f"task={task} init={initial} step={step + 1}", flush=True)
                        if success or done:
                            break
                report["episodes"].append(
                    {
                        "task_id": task,
                        "initial_state": initial,
                        "seed": seed,
                        "initial_state_sha256": env.initial_state_hash,
                        "seconds": time.perf_counter() - started,
                        "instruction": env.instruction,
                        "steps": step + 1,
                        "success": success,
                    }
                )
                write_json(folder / "report.json", report)
            finally:
                env.close()
        report.update(
            status="completed", successes=sum(row["success"] for row in report["episodes"])
        )
        report["success_rate"] = report["successes"] / len(cases)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(folder / "report.json", report)
    print(f"Results: {folder}", flush=True)
    return folder
