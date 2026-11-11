# Colabで使うバージョン

対象はLinux x86_64、Python 3.12／3.13です。00・01と02〜07は別の専用環境を使います。
LIBEROが要求するrobosuite 1.4.0を、Lift用の1.5.2と同じ環境にインストールしません。

| パッケージ | Lift（00・01） | LIBERO・データ（02〜07） |
| --- | --- | --- |
| NumPy | 2.1.3 | 2.1.3 |
| PyTorch | 2.11.0+cu128 | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 | 0.26.0+cu128 |
| LeRobot | 0.7.0 | 0.7.0 |
| Transformers | 5.22.0 | 5.22.0 |
| MuJoCo | 3.3.2 | 3.3.2 |
| robosuite | 1.5.2の公式commit `95743f6687ad7394ccb6e865fa4e1b99114e580d` | 1.4.0 |
| hf-libero | — | 0.1.4 |
| PyAV | — | 15.1.0 |
| OpenCVの通常版／headless版 | 両方4.13.0.92 | 両方4.13.0.92 |
| CMake | 4.1.3 | 3.31.10 |

PyTorchとtorchvisionは[CUDA 12.8の公式配布元](https://download.pytorch.org/whl/cu128)を使います。
Colab標準のPyTorchやNumPyは継承せず、各回の`requirements-py312.txt`または`requirements-py313.txt`から入れます。
03は同じデータ用ライブラリをCPUで使い、GPUを要求しません。

Liftで使うrobosuiteはminkを任意依存にした公式commitです。
LIBEROのネイティブ拡張をビルドするため、setuptools・wheel・CMakeを先に準備し、ビルド分離を無効にします。
EGLヘッダーが不足する場合は、Colab内に必要なライブラリを入れます。
別プロセスのMatplotlibはAggを使用し、MuJoCoのカメラ描画はEGLを使用します。

## 固定リストの更新

- Liftの入力：[lift.in](../requirements/lift.in)。配布元は`colab-py312.txt`／`colab-py313.txt`。
- LIBEROの入力：[libero.in](../requirements/libero.in)。配布元は`libero-py312.txt`／`libero-py313.txt`。
- 両方で使うライブラリの固定条件：[libero-shared-constraints.txt](../requirements/libero-shared-constraints.txt)。

LIBEROの固定リスト先頭にあるコマンドはリポジトリルートから実行します。
Liftを更新するときは`lift.in`のrobosuite公式commit指定を含め、現在の固定リストと差分を確認します。
更新後は`pip check`、主要モジュールのimport、CUDA計算と短い実行を確認します。

## モデル・データの版

- [SmolVLAチェックポイント](https://huggingface.co/HuggingFaceVLA/smolvla_libero/tree/6721902bc4d61e50a3bfdb11dfb4cb626f05d102)
- [VLM設定・トークナイザー](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Instruct/tree/7b375e1b73b11138ff12fe22c8f2822d8fe03467)
- [LIBEROシーンアセット](https://huggingface.co/datasets/lerobot/libero-assets/tree/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6)
- [LeRobot形式のLIBEROデータ](https://huggingface.co/datasets/lerobot/libero/tree/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4)

実行時の確認範囲は[検証状況](validation.md)を参照してください。
