"""Paths, provenance, and small validated configurations shared by lessons."""

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


def output_directory(kind: str) -> Path:
    folder = (
        PROJECT
        / "outputs"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"_{kind}_{uuid4().hex[:8]}")
    )
    folder.mkdir(parents=True, exist_ok=False)
    return folder
