# DTD 少样本标签传播

实验使用上游 `master` 的 `87dc705` 和官方 RN50 预提取特征。三组均按文件原始顺序处理 DTD 的 1,692 张测试图像。每组启动独立进程，使用 `k_text=3`、`k_image=8`、`k_fewshot=8`、`gamma=10`、`alpha=1` 和 `T=3`。

0-shot 的固定标签来自 47 个类别文本原型。1-shot 与 16-shot 还从官方支持集 `features/fine_grained/RN/few_shots/DTD.pt` 读取每类前 1 或 16 张图像及其真实标签，分别加入 47 或 752 个支持节点。每轮传播后，文本与支持节点恢复固定标签。历史测试节点仅保留预测最高类别的分数，并将其乘以 0.2 供下一张图使用。测试集真实标签在预测后用于计算准确率。按特征逐行比对，支持集和测试集之间没有完全相同的向量。

## 结果

| 每类标注样本数 | 支持节点数 | Top-1 准确率 | 相比 0-shot |
|---:|---:|---:|---:|
| 0 | 0 | 54.43% | — |
| 1 | 47 | 55.73% | +1.30 个百分点 |
| 16 | 752 | 65.43% | +11.00 个百分点 |

在本次固定支持集上，每类加入一张带标签的图像后，准确率从 54.43% 提高到 55.73%。每类增加到 16 张后，准确率达到 65.43%，比 1-shot 再高 9.70 个百分点。较多支持样本为测试图像提供更多可能的同类邻居，同时改变了 Few-shot 边的重加权统计。这三组实验衡量的是增加标注样本带来的整体收益。16-shot 实测与论文 DTD 的 65.37% 相差 0.06 个百分点。1-shot 只评估了官方支持集的首个样本，尚无不同支持集抽样下的波动估计。 [ECALP 论文表 4](https://arxiv.org/html/2412.18303v2#S4.T4)

## 复现

```powershell
python experiments/run_fewshot_results.py --threads 2
python experiments/plot_fewshot_results.py
```

[结果 CSV](fewshot_results.csv)逐行记录命令、运行耗时、日志路径、数据 SHA-256 和 PyTorch 版本。耗时包含子进程启动和特征加载。

[标签传播流程图](label_propagation_flow.png) · [Shot 数对比图](fewshot_accuracy.png)
