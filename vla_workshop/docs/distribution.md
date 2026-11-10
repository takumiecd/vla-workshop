# GitHubとColabでの配布

## 役割

- GitHub：ノートブック、共通コード、設定、説明、実験の考察を管理する。
- Colab：教材の特定commitを取得して、環境確認・推論・今後の学習を実行する。
- Driveなど：動画、収集データ、学習済み重みを保存する。

現在のGitルートは`self_work/`です。このルートをGitHubへ配布する場合、ノートブックのパスは`vla_workshop/colab.ipynb`です。
この実装作業ではGitHubリポジトリの作成、remote設定、commit、pushはしていません。

## GitHubで配布するとき

1. 教材の共通コード・設定・説明をGitHubへpushし、そのcommit SHAを取得します。
2. `colab.ipynb`の教材取得セルに、配布先の`REPOSITORY`、コードの`COMMIT`、`PROJECT_SUBDIR`を設定します。
3. 設定したノートブックをpushし、READMEに「Open in Colab」リンクを置きます。
4. ノートブックを開いて実行し、表示された教材commitと、00・01の結果を確認します。

取得するコードのcommitと、配布するノートブック自身のcommitは別でも構いません。
全員が使う共通コードの版は`COMMIT`で固定し、配布リンクも必要に応じてノートブックのcommitに固定します。

リンクの形式は以下です。配布先が決まったらプレースホルダーを実際の値に置き換えます。

```text
https://colab.research.google.com/github/<owner>/<repo>/blob/<notebook-ref>/vla_workshop/colab.ipynb
```

ノートブックの取得セルは公開GitHubリポジトリ向けです。
privateで始める場合は、下記のZIP配布も利用できます。トークンを教材や出力に保存する操作はありません。

## ZIPで先に試すとき

プロジェクトのルートで次を実行します。

```bash
python3 scripts/build_colab_bundle.py
```

参加者には`colab.ipynb`と、生成された`outputs/colab/vla_workshop.zip`を渡します。
Colabの「ノートブックを開く」からipynbをアップロードし、教材取得セルでZIPを選択します。
各ファイルのSHA256を照合してからセットアップします。

このZIPは教材ソース専用です。データ、モデル重み、outputs、venv、Git履歴は含みません。

## 参加者の作業

参加者はノートブックを自分のDriveにコピーし、GPUランタイムで上から実行します。
実行結果はColab内の`outputs/`に保存されます。最後のセルでZIPを取得し、Driveなどへ保管してください。
Colabのランタイムを削除するとローカルのファイルは失われるため、提出前に保存した動画・ログを開いて確認します。

教材を更新するときは、コードcommitとノートブック設定をセットで更新します。
同じ比較実験の途中では、参加者ごとに異なる版へ更新しないようにします。
