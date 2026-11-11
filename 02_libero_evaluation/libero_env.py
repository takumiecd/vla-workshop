"""Prepare LIBERO assets and open a scene at a known initial state."""

import importlib.util
import os
from pathlib import Path

from common import PROJECT, write_json

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
    # hf-libero checks its installed assets directory before downloading anything.
    # This directory is inside this lesson's isolated Python environment.
    local_assets = package / "assets"
    if not local_assets.exists():
        local_assets.symlink_to(assets, target_is_directory=True)
    if local_assets.resolve() != assets.resolve():
        raise RuntimeError("LIBERO assets must point to the pinned snapshot")
    return assets


def open_scene(config, task_id, initial_state, seed):
    prepare_libero(config["renderer"])
    from lerobot.sim.integrations.libero import get_task_init_states
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[config["suite"]]()
    if not 0 <= task_id < suite.n_tasks:
        raise ValueError(f"Invalid task id: {task_id}")
    states = get_task_init_states(suite, task_id)
    if not 0 <= initial_state < len(states):
        raise ValueError(f"Invalid initial state: {initial_state}")
    task = suite.get_task(task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        ),
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
    )
    try:
        env.seed(seed)
        env.reset()
        obs = env.set_init_state(states[initial_state])
        for _ in range(config["settle_steps"]):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    except BaseException:
        env.close()
        raise
    return env, obs, task.language


def scene_cases(config: dict, split: str) -> list[tuple[int, int]]:
    states = config["initial_states"]
    if split not in states:
        raise ValueError(f"Missing initial states for {split}")
    for name, values in states.items():
        if (
            not values
            or len(set(values)) != len(values)
            or any(type(i) is not int or i < 0 for i in values)
        ):
            raise ValueError(f"Invalid {name} initial states")
    all_states = [value for values in states.values() for value in values]
    if len(all_states) != len(set(all_states)):
        raise ValueError("Initial states overlap between splits")
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
