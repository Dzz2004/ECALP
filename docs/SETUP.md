# 成员 A：数据、环境与主结果

目标：RN50，DTD / EuroSAT，Zero-shot CLIP 与官方 ECALP 的同特征对照。
分支：`feat/data-baseline`。官方图构建、重加权和传播实现保持原样。
`main.py` 仅对两处 `torch.load` 增加 `map_location=device`，使 CPU 保存的特征可在 GPU 上运行，
也使 GPU 保存的特征可在 CPU 上读取。该问题已由 CPU 特征的 GPU 小型测试触发。

首次正式实验已于 2026-11-15 完成，实测表格、日志入口和基线口径分析见
[主结果报告](../results/main_results.md)。

## 1. 环境

本组首轮验证环境：Python 3.10、PyTorch 2.6.0+cu124、NumPy 1.26.4、Matplotlib 3.10.0。
**原作者测试版本为 PyTorch 1.13.1；本组当前版本不同，结果必须保留这项差异。**
本项目依赖只涉及预提取特征推理、绘图和下载，不需要重新安装 CLIP 或下载模型权重。

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Linux 默认 PyTorch wheel 包含 CUDA 依赖；需要可兼容的 NVIDIA 驱动。
无 GPU 的 Linux / Windows 组员可先安装 CPU wheel，再安装其余依赖：

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

CPU 可用于检查与运行，正式实验优先单 GPU。不能仅依据沙箱内的
`torch.cuda.is_available()` 判断服务器是否有 GPU；本次沙箱内不可见，沙箱外可用。

当前服务器的 `.venv` 通过 `--system-site-packages` 复用已有 `minimind` Conda 环境，
没有修改那个环境。上述面向组员的安装命令使用独立环境；干净环境的联网安装尚未完成验证。
原版 1.13.1 的 CUDA 安装包下载较慢，CPU 依赖安装未获批准，首轮继续用已有 2.6.0 环境。
实际验证结果与依赖版本记录到 `results/`。

## 2. 官方数据

目录：[作者公布的 Google Drive](https://drive.google.com/drive/folders/10MMCLgnm32XqgS-iPgw0I2Y3y4gzL_47)。
已确认提供 `fine_grained.tar.gz`（页面显示约 870 MB）。下载整个细粒度包，
只提取本组需要的 RN50 特征；不下载风格迁移与扰动数据。

```bash
mkdir -p features_zip
python -m gdown 1-CXxtY77_Zj2Q1T2x0CFN-5a0xjKtz_A -O features_zip/fine_grained.tar.gz --continue
python experiments/prepare_features.py
```

解压脚本适配归档内部的作者绝对目录前缀，只复制指定普通文件，并拒绝覆盖已有文件。
数据与压缩包都已被 `.gitignore` 忽略。预期路径：

```text
features/fine_grained/RN/DTD.pt
features/fine_grained/RN/eurosat.pt
features/fine_grained/RN/few_shots/DTD.pt
```

本次已确认三个文件均在作者归档内：DTD 为 `[1692,1024]` 图像特征、47 类；
EuroSAT 为 `[8100,1024]` 图像特征、10 类。DTD Few-shot 特征为 `[47,16,1024]`，
标签为 `[47,16,47]` one-hot。Few-shot 文件已下载，但本轮仅运行 Zero-shot。
若未来作者归档缺少支持集，脚本会提示并记录到 manifest；可加 `--require-fewshot` 将缺失作为硬错误。

`results/data_manifest.json` 记录原始下载来源、归档及特征文件 SHA-256。
大文件只在本地或共享存储提供，不提交 Git。

## 3. 校验与运行

```bash
# 检查形状、类型、有限值、零向量、标签范围、类别计数及特征范数。
python experiments/run_main_results.py --inspect-only --device cpu

# 正式主结果：按文件原始顺序，单 GPU；把 4 换为自己可用的物理 GPU 编号。
python experiments/run_main_results.py --datasets DTD eurosat --device cuda:4

# 没有 GPU 的组员使用：
python experiments/run_main_results.py --datasets DTD eurosat --device cpu

# 生成主结果图。
python experiments/plot_main_results.py
```

每个数据集的 ECALP 启动独立 `main.py` 进程，避免图与伪标签状态跨实验复用。
统一参数为官方默认：`k_text=3, k_image=8, k_fewshot=8, gamma=10, alpha=1, T=3`；
历史伪标签衰减系数 `beta=0.2` 在原实现中对应 `/5`。

保留原版直接运行入口，用于复核：

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python main.py --task fine_grained --datasets DTD --clip_model RN --mode ZS
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python main.py --task fine_grained --datasets eurosat --clip_model RN --mode ZS
```

CLIP Baseline 将图像特征、类别文本原型归一化后，用余弦相似度最大值分类。
若图像包含多个视图，先按官方实现沿视图维取均值，再归一化。测试标签只用于计算准确率，
不会传给 ECALP。ECALP 的当前样本复制、稀疏图逻辑和输入顺序均沿用官方入口。

## 4. 记录约定

- `results/main_results.csv`：`dataset, clip_model, mode, num_shots, variant, accuracy,
  command, commit, notes`，另记录 `source, run_id, seconds, device, torch_version,
  feature_sha256, log`。
- `results/logs/<run_id>/metadata.json`：环境、可见 GPU、参数、代码哈希、Git 未提交状态及数据检查。
- 同目录保存每组 CLIP / ECALP 日志。ECALP 准确率沿用官方输出的两位小数。
- CSV 采用追加记录；绘图只选择同一次运行、同一数据哈希下成对的 `source=ours` 结果。
- `seconds` 是 Baseline 计算耗时或 ECALP 子进程耗时（含加载），不能与论文端到端每图耗时直接比较。
- 当前分支未提交时，`commit` 是基础提交；必须结合 `source_sha256` 和工作区状态定位实际代码。
- 合成特征测试使用 `--smoke-test`，输出到 `/tmp`，不会混入正式结果。

## 5. 论文对照口径

论文 [Table 1](https://arxiv.org/html/2412.18303v2#S4.T1) 的 RN50 参考值：

| 数据集 | 普通 CLIP | Ensemble | ECALP |
|---|---:|---:|---:|
| DTD | 40.37 | 40.07 | 54.49 |
| EuroSAT | 23.69 | 37.54 | 49.09 |

这些都是 `paper_reference`，不能写成本组测量值。官方预提取文件中的文本原型属于
何种提示词设置，需要结合实测结果与上游说明核实；不能仅凭文件名认定与普通 CLIP 行一致。
本组主对照使用完全相同的文本原型、图像、标签及顺序。

## 6. 小型验证

```bash
python experiments/test_baseline.py
```

该测试验证余弦分类的尺度不变性、多视图处理与数据错误检查。正式跑实验前，
还应使用小型合成数据验证完整脚本、官方子进程和 CSV 记录路径。
