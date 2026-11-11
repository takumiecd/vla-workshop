"""Paths, provenance, and small validated configurations shared by lessons."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

PROJECT = Path(__file__).resolve().parent


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def output_directory(kind: str) -> Path:
    folder = (
        PROJECT
        / "outputs"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"_{kind}_{uuid4().hex[:8]}")
    )
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def episode_split(config: dict) -> tuple[list[int], list[int]]:
    train = config["train_episodes"]
    validation = config["validation_episodes"]
    for name, values in [("train", train), ("validation", validation)]:
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} episodes must be nonempty and unique")
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(f"Invalid {name} episode index")
    if set(train) & set(validation):
        raise ValueError("Train and validation episodes overlap")
    return train, validation
