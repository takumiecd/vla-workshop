# 開発者向けの実行手順

参加者向けの標準入口は`colab.ipynb`です。
このページはローカルのLinux＋NVIDIA GPU環境でコードを変更・確認する場合の補足です。

Python 3.10〜3.12を使い、プロジェクトのルートで実行します。
EGL/OpenGLのランタイムライブラリ（例：`libegl1`、`libgl1`）も必要です。

```bash
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[smolvla,dev]"
python -m pip check
python -c "import torch; assert torch.cuda.is_available()"
vla-inspect --config configs/lift.json --max-steps 20
vla-rollout --config configs/lift.json --max-steps 20
```

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
