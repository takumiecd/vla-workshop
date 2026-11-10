"""Create a reproducible workshop environment without changing the notebook kernel."""

import argparse
import json
import subprocess
import sys
import venv
from pathlib import Path


def setup(project: Path) -> Path:
    """Reuse Colab's CUDA stack and install workshop libraries into a local venv."""
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use a Colab runtime with Python 3.12 for this workshop.")
    subprocess.run(["nvidia-smi"], check=True)
    # Check before pip can replace a large, already-working CUDA installation.
    probe = (
        "import json,torch,torchvision,numpy; "
        "from packaging.version import Version; "
        "assert torch.cuda.is_available(), 'CUDA is unavailable'; "
        "assert Version('2.7') <= Version(torch.__version__) < Version('2.16'); "
        "assert Version('0.22') <= Version(torchvision.__version__) < Version('0.30'); "
        "assert Version('2.0') <= Version(numpy.__version__) < Version('2.3'); "
        "print(json.dumps(dict(torch=torch.__version__,torchvision=torchvision.__version__,"
        "numpy=numpy.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))))"
    )
    base_environment = json.loads(subprocess.check_output([sys.executable, "-c", probe], text=True))
    environment = project / ".venv"
    python = environment / "bin" / "python"
    if not python.exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    commands = [
        [str(python), "-m", "pip", "install", "-e", f"{project}[smolvla]"],
        [
            str(python),
            "-c",
            (
                "import json,torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; "
                "print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,"
                "gpu=torch.cuda.get_device_name(0))))"
            ),
        ],
    ]
    log_path = project / "outputs" / "setup.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        for command in commands:
            print("Running:", " ".join(command), flush=True)
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            log.flush()
            if result.returncode:
                print(log_path.read_text()[-10000:], flush=True)
                raise RuntimeError(f"Setup failed; see {log_path}")
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    (log_path.parent / "requirements-resolved.txt").write_text(freeze)
    (log_path.parent / "setup.json").write_text(
        json.dumps(
            {
                "python": str(python),
                "host_python": sys.version,
                "base_environment": base_environment,
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
