# 参照実装と公式資料

## チームの参照実装

親ワークスペースの`practice/ikeda/`にある以下のファイルを参照しました。
このプロジェクトの実行時依存にはしていません。

| 参照元 | 参照した内容 | この実習での実装 |
| --- | --- | --- |
| `tabletop_env.py` | 環境生成、手先・グリッパー状態の取得、画像の上下方向、評価情報をモデル入力と分ける構成 | 標準Liftを`RobosuiteEnv`にまとめる |
| `robot_api.py` | 7次元のOSC入力とグリッパー開閉の扱い | モデル出力を検証し、直接`env.step`に渡す |
| `run_instruction.py` | 動画・ログ・試行ごとの結果保存 | 共通rolloutで動画、JSONL、実行設定を保存 |

既存のCaPエージェント、色検出、pick/placeの動作関数は、SmolVLAの推論処理には使っていません。

## 公式資料

- [robosuiteコントローラ](https://robosuite.ai/docs/modules/controllers.html)：OSC_POSEと参照座標系。
- [robosuiteの固定commit](https://github.com/ARISE-Initiative/robosuite/tree/95743f6687ad7394ccb6e865fa4e1b99114e580d)：1.5.2でminkを任意依存にした公式修正。PyPI 1.5.2の必須minkはNumPy < 2、LeRobot 0.7はNumPy >= 2を要求するため、OSC_POSE実習ではminkを使わないこのcommitを採用。
- [SmolVLAガイド](https://huggingface.co/docs/lerobot/smolvla)：モデルとfine-tuningの概要。
- [LeRobot 0.7.0](https://github.com/huggingface/lerobot/tree/v0.7.0)：今回の固定API。
- [LIBEROの観測processor](https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/processor/env_processor.py)：画像の180度回転と8次元状態の構成。
- [使用するSmolVLAチェックポイント](https://huggingface.co/HuggingFaceVLA/smolvla_libero/tree/6721902bc4d61e50a3bfdb11dfb4cb626f05d102)：入出力と正規化統計。

新しいAPIへ更新する際は、依存関係・入出力仕様・短い実行確認をまとめて更新します。
