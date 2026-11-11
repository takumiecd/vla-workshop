"""One rollout loop for environment inspection and model inference."""

import hashlib
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any
from uuid import uuid4

import imageio.v2 as imageio
import numpy as np

from vla_workshop.config import RunConfig
from vla_workshop.envs.robosuite_env import RobosuiteEnv, state_vector, validate_action
from vla_workshop.policies.base import Policy


class HoldPolicy:
    """Zero pose delta and open gripper, for simulator connectivity checks."""

    def __init__(self) -> None:
        self.metadata = {"type": "hold", "description": "environment check; no learned model"}

    def reset(self, *, seed: int) -> None:
        pass

    def select_action(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        return np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def package_versions() -> dict[str, str | None]:
    result = {}
    for name in (
        "vla-workshop",
        "numpy",
        "robosuite",
        "mujoco",
        "lerobot",
        "torch",
        "torchvision",
        "transformers",
        "imageio",
        "imageio-ffmpeg",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def source_metadata() -> dict[str, Any]:
    """Identify the executed code and the simulator's pinned upstream source."""
    package = Path(__file__).resolve().parents[1]
    hashes = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    try:
        direct_url = distribution("robosuite").read_text("direct_url.json")
    except PackageNotFoundError:
        direct_url = None
    return {"package_sha256": hashes, "robosuite_source": json.loads(direct_url or "null")}


def run_episode(
    config: RunConfig,
    policy: Policy,
    env: RobosuiteEnv,
    folder: Path,
    episode: int,
    seed: int,
) -> dict[str, Any]:
    """Record one episode; the caller owns environment creation and cleanup."""
    folder.mkdir()
    obs = env.reset()
    policy.reset(seed=seed)
    write_json(folder / "environment.json", env.describe())
    env.save_initial_state(folder, obs)
    for camera in config.env.cameras:
        imageio.imwrite(folder / f"{camera}.png", obs[f"{camera}_image"][::-1])
    success = False
    total_reward = 0.0
    reason = "max_steps"
    started = time.perf_counter()
    with (
        imageio.get_writer(
            folder / "video.mp4", fps=config.env.control_freq, macro_block_size=1
        ) as video,
        (folder / "steps.jsonl").open("w") as log,
    ):
        video.append_data(env.video_frame(obs))
        for step in range(config.max_steps):
            state_before = state_vector(obs).tolist()
            selected_at = time.perf_counter()
            raw_action = policy.select_action(obs, config.instruction)
            policy_seconds = time.perf_counter() - selected_at
            action = validate_action(raw_action, env.low, env.high)
            obs, reward, done, info = env.step(action)
            success = success or info["success"]
            total_reward += reward
            log.write(
                json.dumps(
                    {
                        "step": step,
                        "state_before": state_before,
                        "policy_action": raw_action.tolist(),
                        "env_action": action.tolist(),
                        "action_clipped": bool(np.any(raw_action != action)),
                        "state_after": state_vector(obs).tolist(),
                        "reward": reward,
                        "success": info["success"],
                        "done": done,
                        "policy_seconds": policy_seconds,
                    }
                )
                + "\n"
            )
            log.flush()
            video.append_data(env.video_frame(obs))
            if success or done:
                reason = "success" if success else "environment_done"
                break
    summary = {
        "episode": episode,
        "seed": seed,
        "steps": step + 1,
        "success": success,
        "reward": total_reward,
        "reason": reason,
        "seconds": time.perf_counter() - started,
    }
    write_json(folder / "summary.json", summary)
    return summary


def run(config: RunConfig, policy: Policy, mode: str) -> Path:
    run_dir = Path(config.output_dir) / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"_{mode}_{uuid4().hex[:8]}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "config": asdict(config),
        "policy": policy.metadata,
        "versions": package_versions(),
        "source": source_metadata(),
        "mode": mode,
        "evaluation_scope": "robosuite Lift connectivity test; not a LIBERO score",
    }
    write_json(run_dir / "run.json", metadata)
    summaries = []
    try:
        for episode in range(config.episodes):
            seed = config.seed + episode
            np.random.seed(seed)
            env = RobosuiteEnv(config.env, seed=seed, max_steps=config.max_steps)
            try:
                summary = run_episode(
                    config,
                    policy,
                    env,
                    run_dir / f"episode_{episode:03d}",
                    episode,
                    seed,
                )
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
            finally:
                env.close()
        write_json(
            run_dir / "summary.json",
            {
                "status": "completed",
                "episodes": summaries,
                "success_rate": sum(s["success"] for s in summaries) / len(summaries),
                "note": "Lift result only; not a LIBERO benchmark result",
            },
        )
    except BaseException as error:
        write_json(
            run_dir / "summary.json",
            {
                "status": "failed",
                "episodes": summaries,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        print(f"Incomplete run saved to {run_dir}", flush=True)
        raise
    print(f"Results: {run_dir}", flush=True)
    return run_dir
