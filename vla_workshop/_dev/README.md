# 教材の開発用ファイル

参加者は、このフォルダを扱う必要はありません。配布するのは親フォルダの各回ディレクトリにあるipynbだけです。

## 配置

- `src/vla_workshop/`：環境接続、SmolVLA接続、動画・ログの記録。
- `configs/`・`requirements/`：実行条件と固定した依存関係。
- `notebooks/lesson.template.ipynb`：説明と実行セルの原稿。
- `scripts/`：ノートブックの生成とColab内のセットアップ。
- `tests/`：共通処理のテスト。
- `docs/`：入出力仕様、バージョン、参照元、検証記録。

## 更新する

共通処理は`src/`、説明や実行セルはテンプレートを編集します。
このフォルダで次を実行すると、親フォルダの各回ディレクトリに00・01を再生成します。

```bash
python3 scripts/build_colab_notebook.py
python3 scripts/build_colab_notebook.py --check
```

各冊には同じ共通プログラムが同梱され、前の冊の変数やファイルに依存しません。
生成済みのipynbを直接変更した場合は、次の再生成前に変更をテンプレートへ反映してください。

## 配布する

各回ディレクトリのipynbを1回ずつ共有Driveへ置くか、Colabリンクを案内します。
参加者には原本から自分のDriveへコピーしてもらいます。
配布用の設定・サンプルデータが必要になった回では、その回のディレクトリに追加します。
READMEの実習一覧に、公開した回のリンクを追加していきます。未完成のipynbを作る必要はありません。
結果の保存先は最後のセルの`RESULTS_DIR`で指定します。

## 確認する

Python 3.12／3.13の開発環境で実行します。

```bash
python -m pytest -q
ruff check src tests lessons scripts notebooks
ruff format --check src tests lessons scripts notebooks
python3 scripts/build_colab_notebook.py --check
```

[ローカルでの実行](docs/development.md) · [入出力仕様](docs/io_contract.md) ·
[固定バージョン](docs/versions.md) · [検証状況](docs/validation.md) · [参照元](docs/references.md)
