# VLA Workshop

robosuiteでSmolVLAを動かし、データの確認・学習・収集を体験する教材です。総復習では、集めたデータで学習し、前後のモデルをLIBEROで比べます。
**初回は[一括ZIPをダウンロード](https://github.com/takumiecd/vla-workshop/releases/latest/download/vla_workshop.zip)します。**
更新の案内があったら、[最新版の変更点](https://github.com/takumiecd/vla-workshop/releases/latest)を見て、該当する回のZIPを取り直します。

```text
vla_workshop/
├── 00_environment/              # 環境・観測の確認
├── 01_smolvla_inference/         # SmolVLAの読み込みと推論
├── 02_libero_evaluation/         # LIBERO評価
├── 03_dataset/                  # データ確認・再生
├── 04_finetuning/               # 既存データで学習
├── 05_collection/               # モデルの動作の収集・採否・変換
└── 06_collected_finetuning/      # 集めたデータでLoRAを学習して比較
```

## 取り組む順番

| 回 | 内容・達成条件 | ランタイム | 各回のZIP |
| --- | --- | --- | --- |
| 00 | robosuiteで画像・8次元状態・7次元行動を確認し、動画を保存する | T4（描画） | [00_environment.zip](https://github.com/takumiecd/vla-workshop/releases/latest/download/00_environment.zip) |
| 01 | SmolVLAを読み込み、観測→推論→行動実行をつなぐ | T4 | [01_smolvla_inference.zip](https://github.com/takumiecd/vla-workshop/releases/latest/download/01_smolvla_inference.zip) |
| 02 | LIBEROのタスク・初期状態を固定し、成功率と失敗動画を記録する | T4 | [02_libero_evaluation.zip](https://github.com/takumiecd/vla-workshop/releases/latest/download/02_libero_evaluation.zip) |
| 03 | データを再生し、観測・行動の対応と学習・検証の分割を確認する | CPU | [03_dataset.zip](https://github.com/takumiecd/vla-workshop/releases/latest/download/03_dataset.zip) |
| 04 | LoRAで学習し、学習率を変えたときの違いを確認する | T4 | [04_finetuning.zip](https://github.com/takumiecd/vla-workshop/releases/latest/download/04_finetuning.zip) |
| 05 | モデルを何回か動かし、成功した試行をLeRobot形式へ変換する | T4 | [05_collection.zip](https://github.com/takumiecd/vla-workshop/releases/latest/download/05_collection.zip) |
| 06 | 総復習。収集したデータでLoRAを学習し、元のモデルと比較する | T4 | [06_collected_finetuning.zip](https://github.com/takumiecd/vla-workshop/releases/latest/download/06_collected_finetuning.zip) |

## 初回のセットアップ

1. 一括ZIPを展開し、できた `vla_workshop` フォルダをGoogle Driveの「マイドライブ」直下に置きます。
2. 取り組む回のフォルダを開き、中の `.ipynb` をColabで開きます。
3. 表にあるランタイムを選び、上から順にセルを実行します。教材を別の場所に置いた場合は、最初のセルの `LESSON_DIR` を変更します。

## 1回分を更新する

たとえば04の更新では、編集済みの `04_finetuning` を別の場所に移してから、`04_finetuning.zip` を `vla_workshop` の中に展開します。手元で展開した場合は、できた `04_finetuning` フォルダをDriveの同じ場所に置きます。新しいフォルダのノートブックをColabで開くと、いつもの設定で実行できます。

保存した結果は `vla_workshop/results` にあります。授業で使った版を取り直すときは、[リリース一覧](https://github.com/takumiecd/vla-workshop/releases)から版を選べます。

結果はフォルダのまま保存します。05の保存セルは06で使うデータのパスを表示します。

手元のファイルを使う場合は、最初のセルで `USE_DRIVE` を `False` にして、ノートブック冒頭の一覧にあるファイルをまとめてアップロードします。結果をDriveに保存するときは、最後のセルでDriveに接続します。

00・01では、ノートブックのセルでrobosuiteの環境を作り、観測を取り、行動を送ります。
03ではLeRobotのデータを開き、04・06ではモデルの出力からlossを計算してパラメータを更新します。
設定は同じフォルダのJSONに、実行の流れと公開APIの呼び出しはセルに、環境準備や画像・状態の変換は同じフォルダのPythonにあります。

初回の実行では、ライブラリやその回に必要なモデル・データのダウンロードに時間がかかることがあります。

## 各回のポイント

- **00・01**：Pandaを使って、Lift環境で画像・状態・行動を確認します。00ではシミュレータの画像描画にT4を使います。
- **02**：最初は20ステップの動作確認を選べます。成功率を比較するときは、タスク・初期状態・ステップ上限を全員でそろえます。
  この回はLIBERO-Spatialの1タスクで成功率を測ります。
- **03**：公開データの`lerobot/libero`から、02で評価したタスクと同じ指示のエピソードを再生します。
- **04**：元の重みを固定してLoRAだけを学習します。学習率だけを変え、1回に1サンプルずつ3回更新して比較します。
  正規化は元のモデル付属の統計を使います。最後に検証用の小さなバッチでlossを計算します。
- **05**：モデルでLIBEROを何回か動かし、成功した試行を学習データにします。動画で成功・未達を確認します。
- **06**：05で変換した成功試行を3件以上用意します。最後の1件を検証用、残りを学習用にし、LoRAで学習します。元のモデルと学習後のモデルを同じタスク・開始時の配置で比べます。

学習用・検証用エピソードは分けますが、開始モデルの事前学習データとの重複は不明です。
成功率だけでなく、試行数、変更した条件、データ量、代表動画、失敗傾向と再実行方法を共有してください。
