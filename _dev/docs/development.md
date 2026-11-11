# 開発用の実行

配布方法・生成手順は[開発用README](../README.md)を参照してください。
参加者は各回のフォルダ一式を使い、同じフォルダのPythonコードを実行します。

開発用パッケージを使う場合は、この`_dev/`で対応する依存関係を選びます。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/colab-py313.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

LIBEROを扱う場合は`requirements/libero-py313.txt`を使います。
Python 3.12では対応する`py312`ファイルを使います。
LIBEROのビルド前に固定版のsetuptools・wheel・CMakeとEGLヘッダーを用意します。
詳細な準備手順は配布用`prepare.py`の原稿である`scripts/prepare_lesson.py`にあります。
パッケージの制約だけで自動選択せず、固定依存リストからインストールしてください。
