"""Keep the distributed lessons explicit and independent of the development package."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_each_notebook_has_adjacent_readable_dependencies():
    notebooks = sorted(ROOT.glob("[0-9][0-9]_*/*.ipynb"))
    assert len(notebooks) == 8
    for notebook in notebooks:
        data = json.loads(notebook.read_text())
        assert data["metadata"]["workshop"]["distribution"] == "adjacent-files"
        for filename in data["metadata"]["workshop"]["files"]:
            assert (notebook.parent / filename).is_file(), (notebook, filename)
        for cell in data["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            assert "base64" not in source and "BUNDLE" not in source
            tree = ast.parse(source)
            assert cell["execution_count"] is None and cell["outputs"] == []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if (
                        node.func.id == "run_python"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                    ):
                        ast.parse(node.args[0].value)
        for helper in notebook.parent.glob("*.py"):
            tree = ast.parse(helper.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert node.level == 0, (helper, "relative package import")
                    assert not (node.module or "").startswith("vla_workshop"), helper
                if isinstance(node, ast.Import):
                    assert all(not alias.name.startswith("vla_workshop") for alias in node.names), (
                        helper
                    )


def test_all_libero_lessons_share_the_same_environment_implementation():
    implementations = [path.read_bytes() for path in ROOT.glob("[0-9][0-9]_*/libero_env.py")]
    assert len(implementations) >= 4 and len(set(implementations)) == 1


def test_source_metadata_reads_only_adjacent_lesson_code(tmp_path, monkeypatch):
    import importlib.util
    import shutil

    source = ROOT / "00_environment"
    for name in ["config.py", "robosuite_env.py", "policy_base.py", "rollout.py"]:
        shutil.copy2(source / name, tmp_path / name)
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "unrelated.py").write_text("not_part_of_the_lesson = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = importlib.util.spec_from_file_location("flat_workshop_rollout", tmp_path / "rollout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = module.source_metadata()["package_sha256"]
    assert set(files) == {"config.py", "robosuite_env.py", "policy_base.py", "rollout.py"}


def test_save_cell_keeps_dataset_and_checkpoint_paths_reusable(tmp_path, capsys):
    import shutil
    from types import SimpleNamespace

    project = tmp_path / "lesson"
    dataset = project / "outputs/collection/dataset/meta/info.json"
    checkpoint = project / "outputs/training/pretrained_model/config.json"
    for path in [dataset, checkpoint]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    notebook = ROOT / "05_collection/05_collection.ipynb"
    source = next(
        cell["source"] for cell in json.loads(notebook.read_text())["cells"] if cell["id"] == "save"
    )
    destination = tmp_path / "drive"
    code = "".join(source).replace("/content/drive/MyDrive/vla_Workshop/results", str(destination))
    exec(
        code,
        {
            "PROJECT": project,
            "Path": Path,
            "shutil": shutil,
            "drive": SimpleNamespace(mount=lambda path: None),
        },
    )
    saved = next(destination.iterdir())
    assert (saved / "collection/dataset/meta/info.json").is_file()
    assert (saved / "training/pretrained_model/config.json").is_file()
    text = capsys.readouterr().out
    assert f"06のDATASET_ROOT: {saved / 'collection/dataset'}" in text
    assert f"07のpolicy.path: {saved / 'training/pretrained_model'}" in text
