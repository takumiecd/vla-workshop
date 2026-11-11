"""Readable notebook exercise cells for each distributed lesson."""

# Cells are ordinary Python executed in the lesson's isolated interpreter.
EXERCISES = {
    "00_environment": [
        (
            "環境を動かす",
            """result = run_python(r\"\"\"
from config import load_config
from rollout import HoldPolicy, run

config = load_config("lift.json")
config.max_steps = 20
run(config, HoldPolicy(), "inspect")
\"\"\")""",
        ),
        (
            "画像・動画・仕様を確認する",
            """from IPython.display import Image, Video, display

display(Image(filename=str(result / "episode_000/agentview.png")))
display(Video(str(result / "episode_000/video.mp4"), embed=True))
print((result / "episode_000/environment.json").read_text())
print((result / "summary.json").read_text())""",
        ),
    ],
    "01_smolvla_inference": [
        (
            "使うモデルとカメラを確認する",
            """policy_config = json.loads((PROJECT / "lift.json").read_text())["policy"]
print("モデル:", policy_config["path"])
print("固定revision:", policy_config["revision"])
print("カメラ:", policy_config["camera_map"])
print("モデルを読み込むコードは、隣のsmolvla.pyにあります。")""",
        ),
        (
            "SmolVLAを読み込み、行動を実行する",
            """result = run_python(r\"\"\"
from config import load_config
from smolvla import SmolVLA
from rollout import run

config = load_config("lift.json")
config.max_steps = 20
# 固定したモデルと、そのモデル付属の前処理・後処理を読み込みます。
policy = SmolVLA(config.policy, config.env.image_size)
run(config, policy, "smolvla")
\"\"\")""",
        ),
        (
            "モデルと動作を確認する",
            """from IPython.display import Video, display

display(Video(str(result / "episode_000/video.mp4"), embed=True))
print((result / "run.json").read_text())
print((result / "summary.json").read_text())
first_step = (result / "episode_000/steps.jsonl").read_text().splitlines()[0]
print("最初の状態と行動:", first_step)""",
        ),
    ],
    "02_libero_evaluation": [
        (
            "評価条件を確認する",
            """config = json.loads((PROJECT / "evaluation.json").read_text())
print(json.dumps(config, indent=2, ensure_ascii=False))
# 最初の接続確認では20ステップ・1初期状態に絞ります。
# 成功率を比較するときはFalseにし、全員で同じ条件を使ってください。
CONNECTION_CHECK = True
if CONNECTION_CHECK:
    config["max_steps"] = 20
    config["initial_states"]["validation"] = [0]
(PROJECT / "active_evaluation.json").write_text(json.dumps(config))""",
        ),
        (
            "LIBEROで推論・評価する",
            """result = run_python(r\"\"\"
from common import read_json
from libero_env import evaluate

evaluate(read_json("active_evaluation.json"), split="validation")
\"\"\")""",
        ),
        (
            "成功率と失敗動画を確認する",
            """from IPython.display import Video, display

report = json.loads((result / "report.json").read_text())
print(json.dumps(report, indent=2, ensure_ascii=False))
for video in sorted(result.glob("task_*/video.mp4")):
    display(Video(str(video), embed=True))""",
        ),
    ],
    "03_dataset": [
        (
            "データと分割を確認する",
            """config = json.loads((PROJECT / "dataset.json").read_text())
# train_episodesとvalidation_episodesは重複させません。
print(json.dumps(config, indent=2, ensure_ascii=False))
(PROJECT / "active_dataset.json").write_text(json.dumps(config))""",
        ),
        (
            "データを読み込み、検証して再生する",
            """result = run_python(r\"\"\"
from common import read_json
from dataset_tools import inspect_dataset

inspect_dataset(read_json("active_dataset.json"))
\"\"\")""",
        ),
        (
            "観測・行動・動画の対応を見る",
            """from IPython.display import Video, display

display(Video(str(result / "replay.mp4"), embed=True))
print((result / "dataset.json").read_text())""",
        ),
    ],
    "04_finetuning": [
        (
            "小規模学習の条件を決める",
            """config = json.loads((PROJECT / "train.json").read_text())
# 03と同じデータ分割で、02で評価した指示を学習します。
# 最初は10ステップ・batch_size=1。VLMを固定してexpertを学習します。
print(json.dumps(config, indent=2, ensure_ascii=False))
(PROJECT / "active_train.json").write_text(json.dumps(config))""",
        ),
        (
            "モデルを読み込み、fine-tuningする",
            """result = run_python(r\"\"\"
from common import read_json
from training import train

# training.pyからsmolvla.pyのモデル読み込みと前処理を使います。
train(read_json("active_train.json"))
\"\"\")""",
        ),
        (
            "学習と検証のlossを確認する",
            """rows = [
    json.loads(line)
    for line in (result / "losses.jsonl").read_text().splitlines()
]
for row in rows:
    print(row)
training_report = json.loads((result / "training.json").read_text())
print("保存したモデル:", training_report["checkpoint"])
print("lossの低下とLIBERO成功率の改善は、分けて確認します。")""",
        ),
        (
            "学習したモデルをLIBEROで評価する",
            """evaluation = json.loads((PROJECT / "evaluation.json").read_text())
evaluation["policy"]["path"] = training_report["checkpoint"]
evaluation["policy"]["revision"] = None
(PROJECT / "trained_evaluation.json").write_text(json.dumps(evaluation))
evaluation_result = run_python(r\"\"\"
from common import read_json
from libero_env import evaluate

evaluate(read_json("trained_evaluation.json"), split="validation")
\"\"\")
print((evaluation_result / "report.json").read_text())""",
        ),
    ],
    "05_collection": [
        (
            "収集条件と補正コードを確認する",
            """config = json.loads((PROJECT / "collection.json").read_text())
print(json.dumps(config, indent=2, ensure_ascii=False))
print((PROJECT / "collector.py").read_text())
# collector.pyを変更する場合は、配布フォルダのファイルを編集して準備セルからやり直します。
# この回はモデルrolloutの収集です。人のリアルタイム操作を記録するUIではありません。
(PROJECT / "active_collection.json").write_text(json.dumps(config))""",
        ),
        (
            "モデルの動作を記録する",
            """result = run_python(r\"\"\"
from common import read_json
from collection import collect

collect(read_json("active_collection.json"), adapter_path="collector.py")
\"\"\")""",
        ),
        (
            "動画を見て採否を決める",
            """from IPython.display import Video, display

collection_report = json.loads((result / "collection.json").read_text())
print(json.dumps(collection_report, indent=2, ensure_ascii=False))
for video in sorted(result.glob("episode_*/video.mp4")):
    print(video.parent.name)
    display(Video(str(video), embed=True))
# 動画・状態・行動を確認したうえで採用するエピソード番号を指定します。
# 06へ進むには、学習2本＋検証1本の最低3本を用意してください。
ACCEPT_EPISODES = []""",
        ),
        (
            "採用した記録をLeRobot形式へ変換する",
            """if not ACCEPT_EPISODES:
    raise ValueError("動画を確認し、ACCEPT_EPISODESを指定してください")
export_code = (
    "from collection import export_accepted\\n"
    + f"export_accepted({str(result)!r}, {ACCEPT_EPISODES!r}, {str(result / 'dataset')!r})"
)
dataset_result = run_python(export_code)
print("学習用データ:", dataset_result)""",
        ),
    ],
    "06_collected_finetuning": [
        (
            "収集したデータを指定する",
            """from google.colab import drive

drive.mount("/content/drive")
# 05の保存セルが表示した「06のDATASET_ROOT」を貼り付けます。
DATASET_ROOT = ""  # @param {type:"string"}
if not DATASET_ROOT or not (Path(DATASET_ROOT) / "meta/info.json").is_file():
    raise FileNotFoundError("05で変換したdatasetフォルダを指定してください")
config = json.loads((PROJECT / "train.json").read_text())
config["dataset"]["root"] = DATASET_ROOT
config["dataset"]["revision"] = None
config["dataset"]["train_episodes"] = [0, 1]
config["dataset"]["validation_episodes"] = [2]
(PROJECT / "active_train.json").write_text(json.dumps(config))
print(json.dumps(config, indent=2, ensure_ascii=False))""",
        ),
        (
            "自分のデータで学習する",
            """result = run_python(r\"\"\"
from common import read_json
from training import train

train(read_json("active_train.json"))
\"\"\")
training_report = json.loads((result / "training.json").read_text())
print(training_report)""",
        ),
        (
            "同じ条件で評価する",
            """evaluation = json.loads((PROJECT / "evaluation.json").read_text())
evaluation["policy"]["path"] = training_report["checkpoint"]
evaluation["policy"]["revision"] = None
(PROJECT / "trained_evaluation.json").write_text(json.dumps(evaluation))
evaluation_result = run_python(r\"\"\"
from common import read_json
from libero_env import evaluate

evaluate(read_json("trained_evaluation.json"), split="validation")
\"\"\")
print((evaluation_result / "report.json").read_text())""",
        ),
    ],
    "07_improvement": [
        (
            "改善する条件を確認する",
            """print((PROJECT / "experiment.py").read_text())
# 配布例はaction chunkから実行する数を1→2に変える実験です。
# 学習したモデルで比べる場合はexperiment.pyでpolicy.pathを指定します。
# 仮説を記録し、評価用の初期状態や試行数は変更しません。""",
        ),
        (
            "ベースラインと候補を同じ条件で評価する",
            """result = run_python(r\"\"\"
import copy
from common import read_json
from comparison import compare
from experiment import configure
from libero_env import evaluate

baseline_config = read_json("experiment.json")
candidate_config = configure(copy.deepcopy(baseline_config))
baseline = evaluate(baseline_config, split="validation")
candidate = evaluate(candidate_config, split="validation")
compare(str(baseline), str(candidate))
\"\"\")""",
        ),
        (
            "改善・悪化した初期状態を確認する",
            """comparison = json.loads((result / "comparison.json").read_text())
print(json.dumps(comparison, indent=2, ensure_ascii=False))
print("試行数、変えた条件、成功例・失敗例を一緒に報告してください。")""",
        ),
        (
            "最終評価（条件を決めた後に実行する）",
            """# 候補を決めてからTrueにします。最終評価を見て条件を選び直さないでください。
RUN_FINAL_TEST = False
if RUN_FINAL_TEST:
    final_result = run_python(r\"\"\"
import copy
from common import read_json
from comparison import compare
from experiment import configure
from libero_env import evaluate

baseline_config = read_json("experiment.json")
candidate_config = configure(copy.deepcopy(baseline_config))
baseline = evaluate(baseline_config, split="test")
candidate = evaluate(candidate_config, split="test")
compare(str(baseline), str(candidate))
\"\"\")
    print((final_result / "comparison.json").read_text())""",
        ),
    ],
}
