# 開発者向けの実行手順

参加者向けの入口は各回の`00_environment/00_environment.ipynb`と`01_smolvla_inference/01_smolvla_inference.ipynb`です。
このページはローカルのLinux＋NVIDIA GPU環境でコードを変更・確認する場合の補足です。

Python 3.12または3.13を使い、この`_dev/`フォルダで実行します。
EGL/OpenGLのランタイムライブラリ（例：`libegl1`、`libgl1`）も必要です。

```bash
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/colab-py313.txt
python -m pip install --no-deps -e .
python -m pip install "ruff>=0.9,<1"
python -m pip check
python -c "import torch; assert torch.cuda.is_available()"
vla-inspect --config configs/lift.json --max-steps 20
vla-rollout --config configs/lift.json --max-steps 20
```

Python 3.12の場合は`colab-py312.txt`を使います。
T4以外でローカル開発する場合も依存リストは同じですが、Colab用のセットアップスクリプトはT4を確認します。
環境確認のみなら`python -m pip install -e .`でも実行できます。
出力先などの相対パスは設定ファイルを基準に解決します。`configs/`直下の設定ではプロジェクトルートが基準です。

## チェック

```bash
python -m pytest -q
ruff check src tests lessons scripts
ruff format --check src tests lessons scripts
```

設定と入出力の意味、正常終了と異常終了、記録・後片付けを確認してから、GPU上で短い実習を実行します。
ノートブックへ処理本体を複製せず、`src/vla_workshop/`の共通コードを呼び出してください。

## ノートブックの生成

説明と実行セルの原稿は`notebooks/lesson.template.ipynb`で管理します。
`src/vla_workshop/`、設定、依存リストを修正した後は、次を実行します。

```bash
python3 scripts/build_colab_notebook.py
python3 scripts/build_colab_notebook.py --check
```

各冊には同じ共通プログラムが自動同梱されます。生成されたipynbを直接修正すると、再生成時に上書きされます。
生成器は親フォルダの各回ディレクトリの00・01を更新します。READMEを実習の一覧に使います。
テンプレートは講師用です。参加者には生成済みの各冊を渡します。
