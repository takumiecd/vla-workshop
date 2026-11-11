# Colab T4のバージョン構成

対象はLinux x86_64、Python 3.12／3.13、NVIDIA T4です。
ノートブックはGPUをT4に指定し、接続後にも`nvidia-smi`で割り当てを確認します。
GPUを割り当てる前の接続失敗は、Pythonパッケージのインストールとは別の段階です。

## 固定する主要パッケージ

| パッケージ | バージョン |
| --- | --- |
| NumPy | 2.1.3 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| CUDAビルド | 12.8 |
| LeRobot | 0.7.0 |
| Transformers | 5.22.0 |
| MuJoCo | 3.3.2 |
| robosuite | 1.5.2の公式commit `95743f6687ad7394ccb6e865fa4e1b99114e580d` |
| OpenCVの通常版／headless版 | 両方4.13.0.92 |

robosuiteとLeRobotが異なるOpenCV配布名を要求するため、両方の版をそろえています。
PyTorchとtorchvisionはCUDA 12.8の公式配布元を指定します。
通常のPyPI版PyTorchに依存するCUDAパッケージとの混在を避けるため、CUDAの接尾辞も含めて固定します。

間接依存を含む102パッケージは、以下に固定しています。

- Python 3.12：[colab-py312.txt](../requirements/colab-py312.txt)
- Python 3.13：[colab-py313.txt](../requirements/colab-py313.txt)

各ファイルの先頭に再生成コマンドを記載しています。
更新時は依存解決だけでなく、インストール、`pip check`、主要モジュールのimport、環境起動を確認します。

## セットアップ時の確認

教材専用のvenvにインストールし、Colab標準のPythonパッケージを継承しません。
インストール後は以下を確認してから実習へ進みます。

1. `pip check`で依存条件の不整合がないこと。
2. NumPy・Torch間のデータ変換。
3. MuJoCo、robosuite、SmolVLAとそのprocessorのimport。
4. T4が割り当てられ、PyTorchのビルドに`sm_75`が含まれること。
5. T4上の小さなCUDA計算が成功すること。

結果は`outputs/environment.json`、`outputs/setup.json`、`outputs/setup.log`に保存します。
実際に確認できた範囲は[検証状況](validation.md)を参照してください。

## ColabのPython更新について

前の教材はPython 3.12だけを受け付けていましたが、Colabの2026.10ランタイムはPython 3.13系です。
現在はPythonのminor版に対応する固定依存リストを選びます。

公式情報：[Colab Runtime Versions](https://research.google.com/colaboratory/runtime-version-faq.html)、
[PyTorch CUDA 12.8配布元](https://download.pytorch.org/whl/cu128)。
