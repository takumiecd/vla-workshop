# ipynbを配布する

参加者には目的別のipynbを渡します。GitHubは講師の開発・版管理に使い、配布には共有Driveも使えます。
参加者が教材リポジトリをcloneしたり、教材のPythonファイルを別途ダウンロードしたりする操作はありません。
ライブラリとモデルは実行時にネットワークから取得します。

## 配布するファイル

- `lessons/00_environment/colab.ipynb`：環境と観測の確認。
- `lessons/01_smolvla_inference/colab.ipynb`：CUDAによるSmolVLA推論。
- `colab.ipynb`：各冊への案内。実行セルはありません。

各冊には、共通のPythonコード・設定・固定依存リストを同梱しています。
各冊を単独で実行でき、前のノートブックの変数やファイルは不要です。
講師は`src/vla_workshop/`を一か所で修正し、配布前に各冊を再生成します。
内蔵プログラムのSHA-256をノートブックと結果に記録します。

## 共有Driveで配布する

1. 下記の生成コマンドを実行し、`outputs/colab/vla_workshop_notebooks.zip`を展開します。
2. 展開したipynbを共有Driveの教材フォルダに置きます。ZIPを参加者に展開してもらう必要はありません。
3. 参加者には00・01の原本を閲覧できる権限を付けます。
4. 各自「ファイル → ドライブにコピーを保存」で自分用のコピーを作ります。
5. 各冊でT4 GPUを選び、上から実行し、最後のセルで結果をDriveに保存します。

共有原本を全員が直接編集しない運用にします。
結果を共有する場合は、原本とは別の結果フォルダにメンバーごとの保存先を作ります。
保存先のパスはノートブック最後の`RESULTS_DIR`で指定します。
結果ファイル名には実習名・日時・ランダムIDが付き、同時実行での衝突を避けます。

GitHubからも[実習一覧](https://colab.research.google.com/github/takumiecd/vla-workshop/blob/main/vla_workshop/colab.ipynb)を開けます。
共有Drive経由でも同じ内容を実行できます。
共有フォルダの作成・アップロードは、配布先を決めてから行ってください。

## 更新手順

プロジェクトのルートで実行します。

```bash
python3 scripts/build_colab_notebook.py
python3 scripts/build_colab_notebook.py --check
```

生成した各冊をGitHubへpushし、共有Driveにも新しい原本を置きます。
各自の古いDriveコピーは自動更新されません。教材を更新したことを知らせ、新しい原本からコピーしてもらいます。
同じ比較実験では同じ版を使います。

ノートブックに表示結果や認証情報を保存して配布しないでください。
モデル重み・収集データ・動画はノートブックに埋め込まず、Driveなどで管理します。

開発・検証用のソースZIPは`python3 scripts/build_colab_bundle.py`で別途作れます。
これは講師用で、参加者の実習には不要です。
