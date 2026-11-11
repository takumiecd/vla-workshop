# 教材の開発用ファイル

各回の配布物は、親フォルダの`00_environment/`〜`07_improvement/`にあります。
参加者はその回のフォルダ一式を使い、この`_dev/`を参照しません。

## 編集する場所

- `src/vla_workshop/`：共通処理の原稿。配布時に同じ回の通常のPythonファイルへ変換します。
- `scripts/lesson_content.py`：各回のノートブックに見せる設定・実行・結果確認のセル。
- `materials/`：参加者が編集する`collector.py`と`experiment.py`の原稿。
- `configs/`：各回の初期設定。
- `requirements/`：Lift用とLIBERO用の固定依存リスト。
- `tests/`：入出力、分割、記録、比較の検証。

## 配布物を生成する

このフォルダで実行します。開発環境にはRuff、またはRuffを実行できるuvを用意してください。

```bash
python3 scripts/build_colab_notebook.py
python3 scripts/build_colab_notebook.py --check
```

配布する`.py`は同じ回のファイルだけをimportし、`vla_workshop`パッケージをimportしません。
ノートブックにもコードを隠して埋め込まず、使うファイルと実行する処理を明示します。
各回一式のZIPと、8回分をまとめた`vla_workshop_00-07.zip`は`outputs/distribution/`に生成します。
生成したファイルを直接変更すると、再生成で上書きされます。講師の修正は上記の原稿に反映してください。
参加者は自分に配布されたコピーのPython・JSONを編集できます。

## 実行環境

00・01はrobosuite 1.5.2、02以降はLIBEROが要求するrobosuite 1.4.0を使います。
それぞれの回の専用venvに固定依存を入れ、同じ環境へ混ぜません。
03はCPUでデータを扱います。他の回はT4を確認します。
LIBEROのネイティブビルドにはCMakeなどを先に入れ、ビルド分離を使わずにインストールします。

## 検証する

共通処理の開発用には、対応する固定依存を入れてから`pip install --no-deps -e .`を実行します。
配布物の実行では教材パッケージのインストールは不要です。

```bash
python -m pytest -q
ruff check src tests scripts materials
ruff format --check src tests scripts materials
python3 scripts/build_colab_notebook.py --check
```

[入出力仕様](docs/io_contract.md) · [バージョン](docs/versions.md) ·
[検証状況](docs/validation.md) · [参照元](docs/references.md)
