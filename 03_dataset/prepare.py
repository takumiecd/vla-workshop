"""Prepare only the supplied lesson folder in an isolated Python environment."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import venv
from pathlib import Path


def run(command: list[str], log) -> None:
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise RuntimeError("準備に失敗しました。outputs/setup.logを確認してください")


def prepare(project: Path, cpu: bool = False) -> Path:
    if sys.version_info[:2] not in {(3, 12), (3, 13)}:
        raise RuntimeError("Python 3.12または3.13を選んでください")
    if not cpu:
        try:
            names = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise RuntimeError("この回はT4 GPUを選択し、接続してから実行してください") from error
        if "T4" not in names:
            raise RuntimeError("T4 GPUランタイムを選んでください")
    lock = project / f"requirements-py{sys.version_info.major}{sys.version_info.minor}.txt"
    signature = hashlib.sha256(lock.read_bytes()).hexdigest()
    python = project / ".venv/bin/python"
    output = project / "outputs"
    output.mkdir(exist_ok=True)
    record = output / "setup.json"
    previous = json.loads(record.read_text()) if record.exists() else {}
    commands = []
    if (
        not python.exists()
        or previous.get("lock_sha256") != signature
        or previous.get("status") != "completed"
    ):
        venv.EnvBuilder(with_pip=False, clear=python.parent.parent.exists()).create(
            python.parent.parent
        )
        build_packages = ["setuptools==81.0.0", "wheel==0.48.0"]
        build_packages += [
            line for line in lock.read_text().splitlines() if line.startswith("cmake==")
        ]
        commands = [
            [sys.executable, "-m", "pip", "--python", str(python), "install", "pip==26.3"],
            [str(python), "-m", "pip", "install", *build_packages],
            [str(python), "-m", "pip", "install", "--no-build-isolation", "-r", str(lock)],
        ]
        if "hf-libero==" in lock.read_text() and not Path("/usr/include/EGL/egl.h").exists():
            commands[0:0] = [
                ["apt-get", "update", "-qq"],
                ["apt-get", "install", "-y", "-qq", "libegl-dev", "libgl-dev"],
            ]
        if shutil.which("gcc") is None:
            commands[0:0] = [
                ["apt-get", "update", "-qq"],
                ["apt-get", "install", "-y", "-qq", "build-essential", "linux-libc-dev"],
            ]
    commands.append([str(python), "-m", "pip", "check"])
    check = [
        str(python),
        str(project / "check_environment.py"),
        "--output",
        str(output / "environment.json"),
    ]
    if cpu:
        check.append("--no-cuda")
    commands.append(check)
    record.write_text(json.dumps({"status": "running", "lock_sha256": signature}) + "\n")
    with (output / "setup.log").open("w") as log:
        for command in commands:
            run(command, log)
    record.write_text(
        json.dumps(
            {"status": "completed", "lock_sha256": signature, "python": sys.version, "cpu": cpu}
        )
        + "\n"
    )
    return python


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    prepare(args.project.resolve(), cpu=args.cpu)
