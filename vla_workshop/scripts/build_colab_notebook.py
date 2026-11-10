"""Generate independent Colab lessons with the same embedded workshop program."""

import argparse
import base64
import copy
import hashlib
import io
import json
import textwrap
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

COLAB_ROOT = (
    "https://colab.research.google.com/github/takumiecd/vla-workshop/blob/main/vla_workshop"
)
LESSONS = (
    ("00_environment", "環境と観測の確認", list(range(10)) + [14, 15]),
    ("01_smolvla_inference", "CUDAでSmolVLAを推論する", list(range(6)) + list(range(10, 16))),
)


def bundle_program(project: Path) -> tuple[bytes, dict[str, str]]:
    """Bundle readable source; omit notebooks, generated data, and credentials."""
    patterns = (
        "pyproject.toml",
        "README.md",
        "src/**/*.py",
        "configs/*.json",
        "requirements/*.txt",
        "lessons/**/*.py",
        "docs/*.md",
        "scripts/setup_colab.py",
        "scripts/check_environment.py",
    )
    paths = sorted({p for pattern in patterns for p in project.glob(pattern)})
    buffer = io.BytesIO()
    manifest = {}
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in paths:
            name = path.relative_to(project).as_posix()
            data = path.read_bytes()
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, data)
            manifest[name] = hashlib.sha256(data).hexdigest()
    return buffer.getvalue(), manifest


def embedded_source(bundle: bytes) -> str:
    """Return a standard-library-only preparation cell with no repository fetch."""
    digest = hashlib.sha256(bundle).hexdigest()
    chunks = "\n".join(
        f'    "{part}"' for part in textwrap.wrap(base64.b64encode(bundle).decode(), 96)
    )
    return (
        """# @title 内蔵プログラムを準備する（実行するだけ）
# 実習コードはこのipynbに同梱されています。教材リポジトリの取得は不要です。
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

BUNDLE = (
"""
        + chunks
        + '''
)
EXPECTED_SHA256 = "'''
        + digest
        + """"
data = base64.b64decode(BUNDLE)
assert hashlib.sha256(data).hexdigest() == EXPECTED_SHA256, "内蔵コードの照合に失敗しました"
PROJECT = Path("/content/vla_workshop")
PROJECT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    for name in archive.namelist():
        assert (PROJECT / name).resolve().is_relative_to(PROJECT.resolve()), name
    archive.extractall(PROJECT)
(PROJECT / "outputs").mkdir(exist_ok=True)
(PROJECT / "outputs/source_bundle.json").write_text(json.dumps({"sha256": EXPECTED_SHA256}) + "\\n")
print("内蔵プログラムの準備ができました:", PROJECT)
"""
    )


def introduction(slug: str, title: str) -> str:
    goal = (
        "モデルを使わずにLift環境を動かし、画像・状態・行動と記録を確認します。"
        if slug.startswith("00")
        else "LIBERO用SmolVLAをLift環境に接続し、観測→CUDA推論→行動実行を確認します。"
    )
    return f"""# {slug[:2]}: {title}

{goal}

1. このノートブックを自分のDriveにコピーします。
2. ランタイム → ランタイムのタイプを変更 → **T4 GPU**を選びます。
3. 上から順に実行し、動画とログを確認します。最初は20ステップです。
4. 最後のセルで結果をDriveに保存し、ランタイムを接続解除して削除します。

この1冊に準備と実習コードを同梱しています。別冊の変数や実行結果、教材リポジトリのcloneは不要です。
ライブラリとモデルは実行時に取得します。Python 3.12／3.13向けで、初回準備には時間がかかります。

T4を要求する設定は入っていますが、空きや利用枠によって接続できない場合があります。
接続前の失敗では教材コードはまだ実行されていません。ノートブックから割り当てを強制することはできません。

確認済みの範囲は[検証状況](https://github.com/takumiecd/vla-workshop/blob/main/vla_workshop/docs/validation.md)を参照してください。
"""


def index_notebook() -> dict:
    links = "\n".join(
        f"- [{slug[:2]}: {title}]({COLAB_ROOT}/lessons/{slug}/colab.ipynb)"
        for slug, title, _ in LESSONS
    )
    source = f"""# VLA Workshop — 実習ノートブック一覧

**1冊＝1つの到達目標**として、次の順に進めます。各冊を自分のDriveにコピーしてください。

{links}

各冊に共通コード・設定・固定依存リストを同梱しているため、ipynbだけで実習を始められます。
ライブラリとモデルの取得にはネットワークが必要です。各冊でT4 GPUを選び、上から実行します。

講師は各冊のipynbを共有Driveに置いて配布することもできます。
共有原本を直接編集せず、各自のDriveにコピーし、結果は各冊の最後のセルで保存します。
この一覧には実行セルがなく、GPUへの接続は不要です。

今後は02: LIBERO評価、03: fine-tuning、04: データ収集、05: 改善実験を追加します。
これらは未実装です。現在の00・01はLiftでの接続確認で、LIBERO成功率の評価とは分けて扱います。

[教材の説明](https://github.com/takumiecd/vla-workshop/blob/main/vla_workshop/README.md) · [検証状況](https://github.com/takumiecd/vla-workshop/blob/main/vla_workshop/docs/validation.md)
"""
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "workshop-index",
                "metadata": {},
                "source": source.splitlines(True),
            }
        ],
        "metadata": {"colab": {"name": "VLA Workshop — 実習一覧"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build(project: Path, check: bool = False) -> None:
    template = json.loads((project / "notebooks/lesson.template.ipynb").read_text())
    bundle, manifest = bundle_program(project)
    generated = {project / "colab.ipynb": index_notebook()}
    for slug, title, indices in LESSONS:
        notebook = copy.deepcopy(template)
        notebook["cells"] = [copy.deepcopy(template["cells"][i]) for i in indices]
        notebook["cells"][0]["source"] = introduction(slug, title).splitlines(True)
        for cell in notebook["cells"]:
            tag = cell.get("metadata", {}).get("tags", [])
            if tag == ["source"]:
                cell["source"] = embedded_source(bundle).splitlines(True)
            if tag == ["export"]:
                cell["source"] = "".join(cell["source"]).replace("LESSON", slug).splitlines(True)
            if cell["cell_type"] == "code":
                cell["source"] = "".join(cell["source"]).rstrip().splitlines(True)
                cell["execution_count"] = None
                cell["outputs"] = []
        notebook["metadata"]["colab"]["name"] = f"{slug}.ipynb"
        notebook["metadata"]["workshop"] = {
            "distribution": "self-contained",
            "lesson": slug,
            "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
            "files": manifest,
        }
        generated[project / "lessons" / slug / "colab.ipynb"] = notebook
    for path, notebook in generated.items():
        encoded = json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"
        if check:
            if not path.exists() or path.read_text() != encoded:
                raise SystemExit(f"Stale notebook: {path}; run scripts/build_colab_notebook.py")
        else:
            path.write_text(encoded)
            print(f"Generated {path.relative_to(project)}")
    if check:
        print("All lesson notebooks match their template and common program")
        return
    # Instructor distribution: only notebooks, no separate program ZIP for participants.
    output = project / "outputs/colab"
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(output / "vla_workshop_notebooks.zip", "w", ZIP_DEFLATED) as archive:
        for path in generated:
            name = "00_START_HERE.ipynb" if path.parent == project else path.parent.name + ".ipynb"
            archive.write(path, name)
    print(f"Packaged {len(generated)} notebooks for shared Drive distribution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    build(Path(__file__).resolve().parents[1], check=parser.parse_args().check)
