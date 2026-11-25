# 分工 C：上下文重加权消融

实验使用官方 RN50 预提取特征，按照文件原始顺序运行 DTD 与 EuroSAT Zero-shot。每个数据集的两组实验共用同一特征文件、`k_text=3`、`k_image=8`、`gamma=10`、`alpha=1` 和传播迭代次数 `T=3`。默认组使用类别文本原型的逐维方差，对测试图像之间的相似度进行重加权。`--no-context-reweight` 组改用原始图像特征的余弦相似度。图像与文本原型之间的边及标签传播设置相同；Zero-shot 实验没有 Few-shot 样本边。

| 数据集 | 原始余弦相似度 | 上下文重加权 | 增益 |
|---|---:|---:|---:|
| DTD | 53.78% | 54.67% | +0.89 个百分点 |
| EuroSAT | 49.44% | 50.01% | +0.57 个百分点 |

两组数据都观察到重加权后的准确率提高，DTD 的增益略大。类别文本原型的方差让不同特征维度在图像间连接中具有不同权重，因此 KNN 邻居和图边强度都可能变化。此处测得的增益仅对应测试图像间的边。`xkc` 分支同时修复了增量 KNN 缓存，四组结果均使用修复后的同一代码。A 在修复前测得的官方 ECALP 主结果与此处默认组的差异，不能归因于重加权开关。本次实验使用 PyTorch 2.5.1+cpu，结果保留两位小数，与 A 的 GPU 环境也有版本及设备差异。

运行命令：

```powershell
python experiments/run_reweight_ablation.py --datasets DTD eurosat --threads 4
python experiments/plot_reweight_ablation.py
```

逐组命令、准确率、Git commit、特征 SHA-256、运行时间及日志路径见 [CSV](reweight_ablation.csv)。本次运行的源文件 SHA-256 和设备信息见 [metadata.json](logs/20261117T025111067670Z/metadata.json)。
