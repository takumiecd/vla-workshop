# VLA Workshop

Google Colabで、robosuiteとSmolVLAを動かす実習です。各回のディレクトリに教材をまとめます。

```text
vla_workshop/
├── README.md
├── 00_environment/
│   └── 00_environment.ipynb
├── 01_smolvla_inference/
│   └── 01_smolvla_inference.ipynb
└── _dev/                         # 教材を更新するためのコード
```

現在、参加者に配布するのは各回のipynbだけです。
設定やサンプルデータの配布が必要になった回では、そのディレクトリに追加します。

| 回 | 内容 | ノートブック | Colab |
| --- | --- | --- | --- |
| 00 | 環境を動かし、画像・観測・動画を確認する | [00_environment.ipynb](00_environment/00_environment.ipynb) | [開く](https://colab.research.google.com/github/takumiecd/vla-workshop/blob/main/vla_workshop/00_environment/00_environment.ipynb) |
| 01 | SmolVLAで行動を推論し、環境で実行する | [01_smolvla_inference.ipynb](01_smolvla_inference/01_smolvla_inference.ipynb) | [開く](https://colab.research.google.com/github/takumiecd/vla-workshop/blob/main/vla_workshop/01_smolvla_inference/01_smolvla_inference.ipynb) |

## 使い方

1. 今回のipynbをColabで開き、自分のDriveにコピーします。
2. ランタイムのタイプを **T4 GPU** に変更します。
3. 上から順に実行し、画像・動画・ログを確認します。
4. 最後のセルで結果をDriveに保存します。

参加者に配布するのは、その回のipynbだけです。必要な実習コードは同梱しています。
ライブラリとモデルは実行時に取得します。別のPythonファイルやCLIを準備する必要はありません。
共有Driveにも、このipynbをそのまま置いて配布できます。

## 今後の回

LIBEROでの評価、fine-tuning、データ収集、改善実験を、準備できた回から追加します。
現在の00・01はLiftでの接続確認です。LIBEROの成功率評価は今後扱います。

教材の更新用ファイルは[_dev/](./_dev/README.md)にまとめています。
T4への接続は参加者から確認報告があります。一連のGPU推論は未検証です。[検証状況](_dev/docs/validation.md)
