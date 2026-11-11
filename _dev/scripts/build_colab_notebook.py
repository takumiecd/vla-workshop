"""Build transparent lesson folders: an ipynb and its adjacent Python/config files."""

import argparse
import json
import shutil
import subprocess
import textwrap
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lesson_content import EXERCISES

SOURCES = {
    "config.py": "src/vla_workshop/config.py",
    "robosuite_env.py": "src/vla_workshop/envs/robosuite_env.py",
    "policy_base.py": "src/vla_workshop/policies/base.py",
    "smolvla.py": "src/vla_workshop/policies/smolvla.py",
    "rollout.py": "src/vla_workshop/evaluation/rollout.py",
    "common.py": "src/vla_workshop/course/common.py",
    "libero_env.py": "src/vla_workshop/course/libero.py",
    "dataset_tools.py": "src/vla_workshop/course/data.py",
    "collection.py": "src/vla_workshop/course/collect.py",
    "training.py": "src/vla_workshop/course/train.py",
    "comparison.py": "src/vla_workshop/course/compare.py",
}
IMPORTS = {
    str(Path(source).with_suffix("")).removeprefix("src/").replace("/", "."): name[:-3]
    for name, source in SOURCES.items()
}
IMPORTS.update({".common": "common", ".libero": "libero_env", ".data": "dataset_tools"})
BASE = ["config.py", "robosuite_env.py", "policy_base.py", "rollout.py"]
BENCHMARK = [*BASE, "smolvla.py", "common.py", "libero_env.py"]
TRAINING = [
    "config.py",
    "robosuite_env.py",
    "smolvla.py",
    "common.py",
    "dataset_tools.py",
    "training.py",
]
LESSONS = {
    "00_environment": ("環境と観測の確認", BASE, ["lift.json"], "base"),
    "01_smolvla_inference": ("SmolVLAの推論", [*BASE, "smolvla.py"], ["lift.json"], "base"),
    "02_libero_evaluation": ("LIBEROのベースライン評価", BENCHMARK, ["evaluation.json"], "libero"),
    "03_dataset": (
        "データセットの確認と再生",
        ["common.py", "dataset_tools.py"],
        ["dataset.json"],
        "data",
    ),
    "04_finetuning": (
        "既存データでfine-tuning",
        [*TRAINING, "policy_base.py", "rollout.py", "libero_env.py"],
        ["train.json", "evaluation.json"],
        "libero",
    ),
    "05_collection": (
        "モデルの動作を収集・検証する",
        [*BENCHMARK, "dataset_tools.py", "collection.py", "collector.py"],
        ["collection.json"],
        "libero",
    ),
    "06_collected_finetuning": (
        "自分のデータで再学習",
        [*TRAINING, "policy_base.py", "rollout.py", "libero_env.py"],
        ["train.json", "evaluation.json"],
        "libero",
    ),
    "07_improvement": (
        "改善実験と知見共有",
        [*BENCHMARK, "comparison.py", "experiment.py"],
        ["experiment.json"],
        "libero",
    ),
}


def cell(kind: str, source: str, identifier: str, form=False) -> dict:
    value = {
        "cell_type": kind,
        "id": identifier,
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(True),
    }
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    if form:
        value["metadata"]["cellView"] = "form"
    return value


def material_source(project: Path, name: str) -> str:
    source = project / SOURCES[name] if name in SOURCES else project / "materials" / name
    text = source.read_text()
    for original, replacement in IMPORTS.items():
        text = text.replace(f"from {original} import", f"from {replacement} import")
    if name == "common.py":
        text = text.replace(
            "Path(__file__).resolve().parents[3]", "Path(__file__).resolve().parent"
        )
    if name == "rollout.py":
        text = text.replace('package.rglob("*.py")', 'package.glob("*.py")')
        text = text.replace(
            "Path(__file__).resolve().parents[1]", "Path(__file__).resolve().parent"
        )
    return text


def notebook(slug: str, title: str, files: list[str], profile: str) -> dict:
    cpu = profile == "data"
    runtime = "CPU（ハードウェアアクセラレータなし）" if cpu else "T4 GPU"
    number = slug[:2]
    intro = f"""# {number}: {title}

配布された**この回のフォルダ一式**を使います。ipynbの隣のPython・JSONファイルを読み込みます。
ライブラリと公開モデル・データは実行時に取得します。教材リポジトリのcloneや、隠れたコードの展開は行いません。

1. この回のフォルダをDriveに置き、このipynbをColabで開きます。
2. ランタイムを **{runtime}** にし、最初のセルにフォルダのパスを指定します。
3. 説明→準備→設定→実行→結果確認→保存の順で進めます。

00・01はLift、02以降はLIBEROを扱います。短い接続確認と、成功率を比較する評価は区別します。
T4で短い動作確認を実施しています。長時間学習や最終評価は各実習で行います。

今回使うファイル：
""" + "\n".join(f"- `{name}`" for name in files)
    prepare = (
        """# @title この回の配布ファイルを準備する
import json
import shutil
import subprocess
import os
import shutil
import subprocess
import sys
from pathlib import Path
from google.colab import drive, files

USE_DRIVE = True  # @param {type:"boolean"}
LESSON_DIR = "LESSON_PATH"  # @param {type:"string"}
REQUIRED_FILES = REQUIRED_LIST

if USE_DRIVE:
    drive.mount("/content/drive")
    source = Path(LESSON_DIR)
else:
    # ipynb以外の今回の配布ファイルを、まとめて選んでアップロードします。
    os.chdir("/content")
    files.upload()
    source = Path("/content")
missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
if missing:
    raise FileNotFoundError(f"配布ファイルが不足: {missing}。LESSON_DIRを確認してください。")
PROJECT = Path("/content/lesson_NUMBER")
PROJECT.mkdir(exist_ok=True)
for name in REQUIRED_FILES:
    shutil.copy2(source / name, PROJECT / name)
os.chdir(PROJECT)
print("この回の作業場所:", PROJECT)
""".replace("LESSON_PATH", f"/content/drive/MyDrive/vla_Workshop/{slug}")
        .replace("REQUIRED_LIST", repr(files))
        .replace("NUMBER", number)
    )
    setup = '''# @title ライブラリを準備する
LESSON_ENV = {
    **os.environ,
    "MPLBACKEND": "Agg",
    "PYTHONPATH": str(PROJECT),
    "PYTHONNOUSERSITE": "1",
}
command = [sys.executable, str(PROJECT / "prepare.py"), "--project", str(PROJECT)]
CPU_ARGUMENT
subprocess.run(command, env=LESSON_ENV, check=True)
PYTHON = PROJECT / ".venv/bin/python"


def run_python(program):
    """セルに書いたPythonを、この回の専用Pythonで実行する。"""
    last_result = None
    with subprocess.Popen([str(PYTHON), "-u", "-c", program], cwd=PROJECT, env=LESSON_ENV,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("Results: "):
                last_result = Path(line.removeprefix("Results: ").strip())
        if process.wait():
            raise RuntimeError("実行に失敗しました。直前のエラーを確認してください。")
    return last_result
'''.replace("CPU_ARGUMENT", 'command.append("--cpu")' if cpu else "")
    cells = [
        cell("markdown", intro, "intro"),
        cell("code", prepare, "files", True),
        cell(
            "markdown",
            (
                "## ライブラリの準備\n\nこの回だけの専用Pythonを作ります。"
                "初回はダウンロードに時間がかかります。教材のPythonパッケージはインストールしません。"
            ),
            "setup-note",
        ),
        cell("code", setup, "setup", True),
    ]
    for index, (heading, source) in enumerate(EXERCISES[slug]):
        cells += [
            cell("markdown", "## " + heading, f"step-{index}-note"),
            cell("code", source, f"step-{index}"),
        ]
    cells += [
        cell(
            "markdown",
            (
                "## 確認課題\n\n- 入力・出力と、変更した条件を説明できますか。\n"
                "- 結果のファイルと動画を開き、失敗や不足を確認しましたか。\n"
                "- 次の回で使う保存先を記録しましたか。\n\n"
                "05では採否を確認し、06では学習・検証のエピソードを分けます。"
                "07では検証結果で候補を決めてから最終評価を行います。"
            ),
            "questions",
        ),
        cell(
            "markdown",
            (
                "## 結果をDriveに保存する\n\n動画・ログ・モデルなど、この回のoutputsを保存します。"
                "次の回でデータ・モデルを指定できるよう、フォルダのまま保存します。"
                "保存後にランタイムを接続解除して削除してください。"
            ),
            "save-note",
        ),
        cell(
            "code",
            """# @title 結果を保存する
from datetime import UTC, datetime
from uuid import uuid4

RESULTS_DIR = "/content/drive/MyDrive/vla_Workshop/results"  # @param {type:"string"}
drive.mount("/content/drive")
destination = Path(RESULTS_DIR)
destination.mkdir(parents=True, exist_ok=True)
name = "LESSON_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
saved = destination / name
shutil.copytree(PROJECT / "outputs", saved)
print("保存先:", saved)
for info in sorted(saved.glob("*/dataset/meta/info.json")):
    print("06のDATASET_ROOT:", info.parents[1])
for model in sorted(saved.glob("*/pretrained_model/config.json")):
    print("07のpolicy.path:", model.parent)
""".replace("LESSON", slug),
            "save",
            True,
        ),
    ]
    metadata = {
        "colab": {"name": slug + ".ipynb", "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "workshop": {"distribution": "adjacent-files", "lesson": slug, "files": files},
    }
    if not cpu:
        metadata.update(accelerator="GPU")
        metadata["colab"]["gpuType"] = "T4"
    return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}


def format_material(project: Path, path: Path, text: str) -> str:
    """Normalize generated imports/cells with the same formatter as their source."""
    command = ["ruff"] if shutil.which("ruff") else ["uvx", "--offline", "ruff"]
    options = ["--config", str(project / "pyproject.toml"), "--stdin-filename", str(path), "-"]
    initial = subprocess.run(
        [*command, "format", *options], input=text, text=True, capture_output=True, check=True
    )
    text = initial.stdout
    checked = subprocess.run(
        [*command, "check", "--fix", *options], input=text, text=True, capture_output=True
    )
    if checked.returncode:
        raise RuntimeError(checked.stderr)
    formatted = subprocess.run(
        [*command, "format", *options],
        input=checked.stdout,
        text=True,
        capture_output=True,
        check=True,
    )
    return formatted.stdout


def build(project: Path, check=False) -> None:
    generated = {}
    for slug, (title, modules, configs, profile) in LESSONS.items():
        folder = project.parent / slug
        contents = {name: material_source(project, name) for name in modules}
        contents["prepare.py"] = (project / "scripts/prepare_lesson.py").read_text()
        checker = "check_environment.py" if profile == "base" else "check_libero_environment.py"
        contents["check_environment.py"] = (project / "scripts" / checker).read_text()
        for minor in [12, 13]:
            source = f"colab-py3{minor}.txt" if profile == "base" else f"libero-py3{minor}.txt"
            contents[f"requirements-py3{minor}.txt"] = (
                project / "requirements" / source
            ).read_text()
        for name in configs:
            contents[name] = (project / "configs" / name).read_text()
        files = sorted(contents)
        contents[slug + ".ipynb"] = (
            json.dumps(notebook(slug, title, files, profile), ensure_ascii=False, indent=2) + "\n"
        )
        contents["README.md"] = f"""# {slug[:2]}: {title}

このディレクトリを一式で配布します。ipynbの先頭で、同じフォルダのファイルを準備します。
ipynbだけを配布すると動きません。

モデル・データ・依存ライブラリは別途ダウンロードします。
長い補助処理は隣のPythonファイルにあり、教材リポジトリ全体を読み込む操作はありません。

実行方法・設定・確認課題は[{slug}.ipynb]({slug}.ipynb)にまとめています。
"""
        for name, data in list(contents.items()):
            if name.endswith((".py", ".ipynb")):
                contents[name] = format_material(project, folder / name, data)

        generated.update({folder / name: data for name, data in contents.items()})
    for path, expected in generated.items():
        if check:
            if not path.exists() or path.read_text() != expected:
                raise SystemExit(f"Stale material: {path}")
        else:
            path.parent.mkdir(exist_ok=True)
            path.write_text(expected)
    if not check:
        output = project / "outputs/distribution"
        output.mkdir(parents=True, exist_ok=True)
        for slug in LESSONS:
            with ZipFile(output / f"{slug}.zip", "w", ZIP_DEFLATED) as archive:
                for path in generated:
                    if path.parent.name == slug:
                        archive.write(path, f"{slug}/{path.name}")
        with ZipFile(output / "vla_workshop_00-07.zip", "w", ZIP_DEFLATED) as archive:
            for path in generated:
                archive.write(path, path.relative_to(project.parent))
    print(f"{'Verified' if check else 'Generated'} {len(LESSONS)} lesson folders")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    build(Path(__file__).resolve().parents[1], check=parser.parse_args().check)
