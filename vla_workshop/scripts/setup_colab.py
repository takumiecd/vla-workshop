"""Install the workshop's locked environment for a Colab T4 runtime."""

import argparse
import hashlib
import json
import subprocess
import sys
import venv
from pathlib import Path
from typing import TextIO


def require_t4() -> str:
    """Check the assigned GPU after Colab has connected; this cannot allocate a GPU."""
    try:
        names = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
            )
            .strip()
            .splitlines()
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Colabの「ランタイム → ランタイムのタイプを変更」で「T4 GPU」を選択し、"
            "接続してから実行してください。"
        ) from error
    if len(names) != 1 or "T4" not in names[0].upper():
        raise RuntimeError(f"T4 GPUを選択してください。割り当てられたGPU: {names}")
    return names[0]


def run_logged(command: list[str], log: TextIO) -> None:
    """Stream installer output into both the notebook and the saved log."""
    print("Running:", " ".join(command), flush=True)
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError("セットアップに失敗しました。outputs/setup.logを確認してください。")


def setup(project: Path) -> Path:
    """Install exact dependency versions in an isolated environment and check them."""
    if sys.version_info[:2] not in {(3, 12), (3, 13)}:
        raise RuntimeError("Python 3.12または3.13のColabランタイムを使ってください。")
    gpu = require_t4()
    lock = (
        project / "requirements" / f"colab-py{sys.version_info.major}{sys.version_info.minor}.txt"
    )
    lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest()
    environment = project / ".venv"
    python = environment / "bin" / "python"
    output = project / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    record = output / "setup.json"
    previous = json.loads(record.read_text()) if record.exists() else {}
    ready = (
        python.exists()
        and previous.get("status") == "completed"
        and previous.get("lock_sha256") == lock_hash
        and previous.get("host_python_minor") == sys.version_info.minor
    )
    if not ready:
        # .venv contains generated packages only. Rebuild an older or incomplete installation.
        # Colab may omit ensurepip. The host pip can bootstrap a pip-less venv.
        venv.EnvBuilder(with_pip=False, clear=environment.exists()).create(environment)
        commands = [
            [sys.executable, "-m", "pip", "--python", str(python), "install", "pip==26.3"],
            [str(python), "-m", "pip", "install", "-r", str(lock)],
            [str(python), "-m", "pip", "install", "--no-deps", "-e", str(project)],
        ]
    else:
        commands = []
    commands.extend(
        [
            [str(python), "-m", "pip", "check"],
            [
                str(python),
                str(project / "scripts/check_environment.py"),
                "--output",
                str(output / "environment.json"),
            ],
        ]
    )
    record.write_text(json.dumps({"status": "running", "lock_sha256": lock_hash}) + "\n")
    with (output / "setup.log").open("w") as log:
        for command in commands:
            run_logged(command, log)
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    (output / "requirements-resolved.txt").write_text(freeze)
    record.write_text(
        json.dumps(
            {
                "python": str(python),
                "host_python": sys.version,
                "host_python_minor": sys.version_info.minor,
                "gpu": gpu,
                "lock_file": lock.name,
                "lock_sha256": lock_hash,
                "status": "completed",
            },
            indent=2,
        )
        + "\n"
    )
    print("Setup complete:", python, flush=True)
    return python


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    setup(parser.parse_args().project.resolve())
