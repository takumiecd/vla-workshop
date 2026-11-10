# VLA Workshop

robosuiteでSmolVLAを動かし、データ収集・fine-tuning・LIBEROでの成功率改善へ進む実習教材です。

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takumiecd/vla-workshop/blob/main/vla_workshop/colab.ipynb)

## 参加者へ

1. 上のボタンから実習一覧を開き、00・01それぞれのノートブックを自分のDriveにコピーします。
2. T4 GPUランタイムを選び、上から順番に実行します。
3. 00で環境・観測・記録を確認し、01でCUDAによるSmolVLA推論を実行します。
4. 各冊の最後に結果をDriveへ保存し、ランタイムを接続解除して削除します。

各冊に共通のPythonコードを同梱しています。教材リポジトリのcloneや教材の.pyの別途取得は不要です。ライブラリとモデルは実行時に取得します。共有Driveにipynbを置いて配布することもできます。ノートブックで実行結果を見ながら、各段階のコードを読んで学べます。

- [1レッスンの構成と公開順](vla_workshop/docs/course_structure.md)
- [全体の説明と実習一覧](vla_workshop/README.md)
- [00: 環境と観測の確認](vla_workshop/lessons/00_environment/README.md)
- [01: SmolVLAの推論](vla_workshop/lessons/01_smolvla_inference/README.md)
- [観測と行動の仕様](vla_workshop/docs/io_contract.md)

最初の2段階を実装しています。データ収集・学習・LIBERO評価の実習は順次追加します。
最初の実習はLiftでの接続確認です。LIBEROでの成功率評価とは区別してください。

コードの13件のテストとノートブックの形式・構文を確認しています。
Colab GPUでの実行確認は、CLI接続断とGPU割り当て時の503により未完了です。
現状と確認範囲は[検証状況](vla_workshop/docs/validation.md)を参照してください。
