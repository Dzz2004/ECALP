# B：动态图构建消融

本目录提供 PPT 所需的 B 部分素材。

## 推荐使用

1. `figures/graph_structure.png`：ECALP 整体方法与动态图，可用于总 PPT 第 3 页。
2. `figures/accuracy_comparison.png`：DTD 与 EuroSAT 消融结果，可用于总 PPT 第 7 页。

从仓库根目录重新生成两张图：

```bash
python experiments/plot_graph_ablation.py
```

## 构图图说明

- 蓝色节点是类别文本节点；论文中称为 textual prototypes，图中简写为 `Class 1/2`。
- 橙色节点是已经处理过的测试图像，红色节点是当前新到达的测试图像。它们都属于未标注测试节点。
- 蓝线表示测试图像与类别文本之间的连接，橙线表示测试图像之间的连接。
- 绿色节点是 Few-shot 模式下带真实标签的支持图像。绿色虚线直接连接测试图像，对应论文中的测试图像—Few-shot 连接；本次 Zero-shot 实验不包含该节点和连线。
- 实际图通过 top-k 近邻保持稀疏，示意图只画少量代表性边，不表示全连接。文本节点之间、Few-shot 节点之间，以及文本与 Few-shot 节点之间不建立边。

B 的消融仅让橙色图片—图片边不进入最终邻接矩阵。文本边、节点、上下文重加权、传播参数和 KNN 缓存路径保持不变；没有使用未经验证的 `--k_image 0`。

## 运行与验证

```bash
# 两项合成图结构测试
python -m unittest experiments.test_graph_ablation -v

# 正式消融；每次运行自动建立时间戳日志目录
python experiments/run_graph_ablation.py --dataset DTD --features_path ./features/ --clip_model RN
python experiments/run_graph_ablation.py --dataset eurosat --features_path ./features/ --clip_model RN

# 根据 CSV 重新生成两张素材图
python experiments/plot_graph_ablation.py
```

## 实验设置与结果

RN50、Zero-shot，固定 `k_text=3`、`k_image=8`、`gamma=10`、`alpha=1`、`T=3`。两组实验使用相同特征、代码路径和标签传播设置，唯一差异是是否启用测试图片之间的连接。

| 数据集 | 完整 ECALP | 禁用图片—图片边 | 下降 |
|---|---:|---:|---:|
| DTD | 54.43% | 52.42% | 2.01 个百分点 |
| EuroSAT | 49.04% | 40.38% | 8.66 个百分点 |

## 结论

在 RN50、Zero-shot 和相同传播参数下，关闭测试图片之间的连接后，DTD 准确率由 54.43% 降至 52.42%，EuroSAT 由 49.04% 降至 40.38%，分别下降 2.01 和 8.66 个百分点。两组实验仅改变图片—图片边是否进入传播图，说明历史测试图片能够通过视觉相似性为当前样本提供有效的结构信息。EuroSAT 的下降更明显，表明该数据集更依赖测试样本之间的关系。
实验使用 PyTorch 2.6.0 CPU 环境，与 A 的 GPU 完整模型结果存在 0.05 至 0.06 个百分点差异。

完整命令、环境、源码哈希和日志见 `results/graph_ablation.csv` 及其 `log` 字段。
