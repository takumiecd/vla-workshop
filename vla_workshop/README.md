# VLA Workshop

SmolVLAの推論から、データ収集・fine-tuning・LIBEROでの評価と改善までを段階的に学ぶ実習です。

最終目標は、共通のSmolVLAベースラインに対して、指定したLIBEROタスク群の成功率を改善し、その結果と考察を共有することです。

## 取り組む順番

| 順番 | 実習 | 達成条件 | 状態 |
| --- | --- | --- | --- |
| 00 | [環境と観測の確認](lessons/00_environment/README.md) | 環境を起動し、画像・状態・行動を確認する | 実装済み |
| 01 | [SmolVLAの推論](lessons/01_smolvla_inference/README.md) | 推論と制御を接続し、動画・ログを保存する | 実装済み |
| 02 | LIBEROのベースライン評価 | 共通モデルの成功率を同じ条件で再現する | 今後追加 |
| 03 | 既存データでfine-tuning | 学習したモデルで推論し、学習前後を比較する | 今後追加 |
| 04 | データ収集 | デモを収集・確認・変換・再生する | 今後追加 |
| 05 | 改善実験 | 共通の評価条件で改善を検証し、考察を共有する | 今後追加 |

各実習は「目的 → 実行方法 → 確認すること → 達成条件」の形式で整えます。
1冊の中身と公開順は[レッスン構成](docs/course_structure.md)を参照してください。

13件のテストとノートブックの形式・構文チェックは通っています。
Colab GPUでの00・01の実行確認は未完了です。[検証状況](docs/validation.md)を確認してください。

## Google Colabで取り組む

参加者は目的別のノートブックを使います。各冊だけで準備から実行・結果保存まで進められます。

- [00: 環境と観測の確認をColabで開く](https://colab.research.google.com/github/takumiecd/vla-workshop/blob/main/vla_workshop/lessons/00_environment/colab.ipynb)
- [01: SmolVLA推論をColabで開く](https://colab.research.google.com/github/takumiecd/vla-workshop/blob/main/vla_workshop/lessons/01_smolvla_inference/colab.ipynb)

1. 共有Driveのipynb、または上のリンクを開き、自分のDriveにコピーします。
2. ランタイムのタイプを**T4 GPU**に変更します。各冊のGPU設定もT4です。
3. 上から順に準備・実習・確認課題を進めます。最初は20ステップです。
4. 最後のセルで動画・ログをDriveに保存し、ランタイムを接続解除して削除します。

Python 3.12／3.13向けの固定依存リストから、隔離した教材用venvを準備します。
各冊に共通プログラムを同梱しており、教材リポジトリのcloneや教材の.pyの別途取得は不要です。
ライブラリとモデルは実行時に取得するため、ネットワークと初回準備の時間が必要です。

T4指定は割り当て要求です。Colabへの接続前に失敗している場合は教材コードは動いていません。
古いDriveコピーは自動更新されないため、更新後は新しい原本からコピーしてください。

## 講師・配布担当へ

共有Driveに目的別のipynbを置いて配布できます。各自が原本からコピーして実行します。
GitHubでは読みやすい共通ソースを一か所で管理し、各冊への同梱を自動生成します。

```bash
python3 scripts/build_colab_notebook.py
python3 scripts/build_colab_notebook.py --check
```

`outputs/colab/vla_workshop_notebooks.zip`に配布用のipynbがまとまります。
[配布手順](docs/distribution.md)と[開発手順](docs/development.md)を参照してください。
配布先は[takumiecd/vla-workshop](https://github.com/takumiecd/vla-workshop)です。

## 最初の実習の仕様

- 環境：robosuite 1.5.2の公式ソース（commit固定）/ MuJoCo 3.3.2、標準Lift、Panda、20 Hz。
- 制御：BASIC内のOSC_POSE。ワールド座標系の相対移動・相対回転とグリッパーの7次元。
- モデル：`HuggingFaceVLA/smolvla_libero`。チェックポイントとVLM設定・トークナイザーのcommitを固定。
- 入力：RGB画像2枚と8次元状態、言語指示。物体の正解位置・成功判定は入力しません。
- 行動：モデル付属の統計で逆正規化してから、環境の行動範囲にclip。変換前後を両方記録。
- 実行：FP32、CUDA必須、`n_action_steps=1`。1ステップずつ実行し、毎回推論します。

LIBERO用チェックポイントをLiftに接続する実習です。Liftでのタスク成功や、LIBEROでの成功率の再現を保証するものではありません。
ベース版`lerobot/smolvla_base`は状態・行動の仕様が異なるため、この設定では使いません。
別のチェックポイントを使う場合は、次元に加えて座標系・単位・制御方式も確認してください。

画像の向きと状態・行動の意味は[入出力仕様](docs/io_contract.md)を参照してください。
LIBEROの実習は今後追加します。依存関係が異なるため、この環境にLIBEROを追加インストールする手順はまだ提供していません。

## ディレクトリの役割

```text
vla_workshop/
├── colab.ipynb                 # 目的別ノートブックの一覧
├── scripts/                    # セットアップ・ノートブック生成
├── notebooks/                  # 講師用の共通テンプレート
├── lessons/                    # 各冊のcolab.ipynb・説明・実行用スクリプト
├── src/vla_workshop/
│   ├── envs/                   # robosuite・LIBEROとの接続
│   ├── policies/               # SmolVLAとの接続
│   ├── data/                   # 記録・変換・読み込み
│   └── evaluation/             # rollout・評価・動画保存
├── configs/                    # 共通の環境・モデル・学習設定
├── experiments/                # メンバーごとの実験設定と考察
├── datasets/                   # ローカルの収集データ
└── outputs/                    # 動画・ログ・チェックポイント
```

共通処理は`src/vla_workshop/`に置き、実習間・メンバー間でコード一式を複製しません。
各実習のスクリプトは、共通処理を設定付きで呼び出す入口にします。
環境確認・推論・データ収集・学習後の評価で、同じ環境生成と観測・行動の変換処理を使います。
robosuiteとLIBEROの違いは`envs/`内の接続部分で扱い、それぞれの観測・行動仕様を明記します。
主要な依存関係は`pyproject.toml`、間接依存を含む固定版は`requirements/colab-py312.txt`と`requirements/colab-py313.txt`にまとめます。
[バージョン構成](docs/versions.md)も参照してください。実習の共通設定は`configs/lift.json`です。

## 実装方針と参照コード

robosuiteを土台に、環境の起動から観測取得・行動実行を組み立てます。
最初の題材は標準のLift環境とPandaロボットとし、設定を明示して共通の環境生成処理にまとめています。

親ワークスペースの`practice/ikeda/`は、環境設定・画像と状態の取得・動画とログの保存などの実装を参照します。
既存コードはCaP方式で、LLMが`pick()`などを呼ぶコードを生成します。
VLA実習では、SmolVLAが出力する行動を実行する接続を用意します。

参照元と採用した考え方は[参照実装と公式資料](docs/references.md)に記載しています。
この実習の実行に`practice/ikeda/`のコピーや配置は必要ありません。
段階が進んでも同じ共通プログラムを使い、環境や推論処理の別コピーを増やさない方針です。

## 実験の残し方

設定と考察は`experiments/<メンバー名>/<実験名>/`に置きます。
比較には対象タスク、評価用初期状態、試行数、推論設定、モデル・データ・環境のバージョンを記録します。
モデル選択に使う評価と最終評価を分けます。

動画・データ・重みは`datasets/`または`outputs/`に保存し、Gitには含めません。
共有する実験記録には、生成物の保存先と再実行方法を記載します。

## コードを変更するとき

```bash
python -m pytest -q
ruff check src tests lessons scripts
ruff format --check src tests lessons scripts
```

`lessons/`に処理本体を増やさず、共通コードを変更します。
型とdocstringに加えて、観測・行動の意味が変わる変更では`docs/io_contract.md`も更新してください。
テストは入出力の対応、異常値の拒否、エピソード終了・保存・後片付けを確認します。
実機能の確認には00と01の短いrolloutも実行してください。
