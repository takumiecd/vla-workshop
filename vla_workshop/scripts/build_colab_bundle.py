"""Package only workshop source and documentation for upload to Colab."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def build_bundle(project: Path, destination: Path) -> Path:
    """Build a source-only ZIP with per-file hashes, excluding data and credentials."""
    patterns = (
        "README.md",
        "pyproject.toml",
        ".gitignore",
        "colab.ipynb",
        "src/**/*.py",
        "configs/*.json",
        "lessons/**/*.md",
        "lessons/**/*.py",
        "docs/*.md",
        "scripts/*.py",
        "tests/*.py",
    )
    files = sorted({path for pattern in patterns for path in project.glob(pattern)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(project).as_posix()
            content = path.read_bytes()
            archive.writestr(f"vla_workshop/{relative}", content)
            manifest[relative] = hashlib.sha256(content).hexdigest()
        archive.writestr("vla_workshop/bundle_manifest.json", json.dumps(manifest, indent=2))
    print(destination)
    print("SHA256:", hashlib.sha256(destination.read_bytes()).hexdigest())
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    build_bundle(project, args.output or project / "outputs/colab/vla_workshop.zip")
