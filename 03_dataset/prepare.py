"""Install the pinned dependencies into this lesson's Python environment."""

import hashlib
import subprocess
import sys
import venv
from pathlib import Path


def run(*args):
    subprocess.run(args, check=True)


def prepare():
    if sys.version_info[:2] not in {(3, 12), (3, 13)}:
        raise RuntimeError("Python 3.12または3.13のColabランタイムを選んでください")
    folder = Path(__file__).resolve().parent
    requirements = folder / "requirements.txt"
    signature = hashlib.sha256(requirements.read_bytes()).hexdigest()
    environment = folder / ".venv"
    python = str(environment / "bin/python")
    stamp = environment / "requirements.sha256"

    if not stamp.exists() or stamp.read_text() != signature:
        venv.EnvBuilder(with_pip=False, clear=environment.exists()).create(environment)
        run(sys.executable, "-m", "pip", "--python", python, "install", "pip==26.3")
        run(python, "-m", "pip", "install", "setuptools==81.0.0", "wheel==0.48.0", "cmake==3.31.10")
        run(python, "-m", "pip", "install", "--no-build-isolation", "-r", str(requirements))
        stamp.write_text(signature)
    run(python, "-m", "pip", "check")
    (folder / "outputs").mkdir(exist_ok=True)


if __name__ == "__main__":
    prepare()
