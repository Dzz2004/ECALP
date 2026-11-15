# ECALP 论文复现项目：研究叙事、任务分工与协作规范

> 课程：《图像与视频处理》期末项目  
> 方向：Open-Vocabulary Object Recognition（以开放词汇**图像分类**为实验对象）  
> 主论文：**Efficient and Context-Aware Label Propagation for Zero-/Few-Shot Training-Free Adaptation of Vision-Language Model**（ICLR 2025）  
> 开发仓库：**https://github.com/Dzz2004/ECALP**  
> 项目定位：**核心复现 + 机制消融 + 少量分析**。不重新训练 CLIP，不追求覆盖论文的全部数据集及实验。

## 1. 我们要讲清楚的故事

### 1.1 背景：从 CLIP 到下游适配

CLIP（2021）使用图像编码器和文本编码器将图像、类别名称映射到共享特征空间，通过图文相似度完成 Zero-shot 分类，不需要为下游类别重新训练分类头。然而，原始图文匹配主要独立地评估每张图像，并未充分利用测试样本之间的分布关系，在纹理、遥感、细粒度类别等任务上仍有改进空间。

CoOp（2022）将人工文本 Prompt 替换为可学习的上下文向量，通过少量标注样本适配任务；CoCoOp（2022）引入输入条件化 Prompt，缓解静态 Prompt 在已知类别上适配后损害未见类别泛化的问题。它们说明适配有效，但也引入了额外的训练与推理成本。

ECALP 换了一条路线：**不微调 CLIP，也不重新学习 Prompt，而是在冻结的 CLIP 特征上，利用类别文本、历史测试图像和可选的 Few-shot 标注样本构建图，通过标签传播完成分类。**

### 1.2 核心研究问题

**在不更新 CLIP 参数的前提下，能否利用测试样本之间的关系，以及类别上下文信息，提高 Zero-shot / Few-shot 分类准确率？**

按以下逻辑展开 PPT 和短论文：

1. **问题与动机**：仅使用图文相似度，忽略测试集局部流形结构；现有适配方案还可能需要训练或数据集相关调参。
2. **核心思想**：构建由文本类别原型、Few-shot 样本（可选）、不断到来的测试图像组成的图。
3. **方法模块**：Dynamic Graph Expansion（动态构图）→ Context-Aware Edge Re-weighting（上下文感知边权重加权）→ Iterative / Incremental Label Propagation（迭代、增量标签传播）。
4. **核心实验**：验证 ECALP 相比原始 CLIP 的增益；逐个关闭图关系和重加权；观察少样本标签的帮助。
5. **结论与限制**：讨论收益来自哪些机制，以及图结构依赖样本相似度、顺序推理和额外计算的局限。

**课程项目不声称提出新算法。** 本组的贡献是对近期工作的准确复现、可重复消融和清晰解释。实际观察与原论文不一致时，应记录差异、查明实验条件，不能直接复制论文结果作为本组结果。

### 1.3 项目希望回答的三个问题

- **Q1：ECALP 相比原始 Zero-shot CLIP 提升了多少？**
- **Q2：图像间连接、上下文重加权分别贡献了多少？**
- **Q3：从 Zero-shot 转为 Few-shot 后，准确率如何变化？**

## 2. 统一实验范围与验收标准

| 项目 | 本组约定 |
|---|---|
| CLIP 主干 | **RN50（官方命令使用 `--clip_model RN`）** |
| 数据集 | **DTD 必做；EuroSAT 作为第二个验证集** |
| 特征 | 优先使用作者公布的预提取 `.pt` 特征 |
| 核心评价指标 | Top-1 Accuracy（%） |
| Zero-shot | CLIP Baseline、ECALP、图连接消融、重加权消融 |
| Few-shot | DTD 上 1-shot / 16-shot（至少 16-shot） |
| 不做 | 重训 CLIP、大规模超参搜索、完整 30 场景、强制复现 CoOp/CoCoOp |

论文 Table 1 中，**RN50 / Zero-shot** 的参考值（**不是本组实验结果**）：

| 数据集 | 原论文 CLIP | 原论文 ECALP |
|---|---:|---:|
| DTD | 40.37% | 54.49% |
| EuroSAT | 23.69% | 49.09% |

论文 Table 4 中，DTD 上 ECALP 的 **16-shot** 参考值为 **65.37%**。论文参考数据仅用于复核，不作为本组已完成实验的证据。

**最低完成标准**：DTD 和 EuroSAT 的 Zero-shot 主结果 + DTD 的两项核心消融 + DTD 16-shot 一组结果；所有结果都有对应代码、命令、配置和日志。若时间不足，先保留 DTD 全部核心实验，再缩减 EuroSAT 与其他扩展。

## 3. 参考资料与数据入口

### 3.1 必读资料

| 资料 | 链接 | 需要读的内容 |
|---|---|---|
| **ECALP 主论文（ICLR 2025）** | [arXiv 摘要](https://arxiv.org/abs/2412.18303) · [PDF](https://arxiv.org/pdf/2412.18303) · [HTML 正文](https://arxiv.org/html/2412.18303v2) | §1、§3、§4；Figure 1；Algorithm 1；Table 1、Table 4、Table 5 |
| **本组开发仓库** | https://github.com/Dzz2004/ECALP | `README.md`、`main.py`、`utils/iterative_label_propagation.py` |
| 原作者仓库 | https://github.com/Yushu-Li/ECALP | 运行方法、上游更新、引用信息 |
| **官方预提取特征目录** | [Google Drive：features](https://drive.google.com/drive/folders/10MMCLgnm32XqgS-iPgw0I2Y3y4gzL_47?usp=sharing) | RN50 对应 DTD、EuroSAT 和 DTD Few-shot 特征 |
| 特征提取依赖项目 DMN | https://github.com/YBZh/DMN | **可选**：仅当官方特征缺失、数据格式存在问题时再查 |

### 3.2 背景文献（无需全员精读、无需复现）

- **CLIP** — Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*（ICML 2021）：https://arxiv.org/abs/2103.00020
- **CoOp** — Zhou et al., *Learning to Prompt for Vision-Language Models*（IJCV 2022）：https://arxiv.org/abs/2109.01134
- **CoCoOp** — Zhou et al., *Conditional Prompt Learning for Vision-Language Models*（CVPR 2022）：https://arxiv.org/abs/2203.05557
- **ZLaP** — *Label Propagation for Zero-Shot Classification with Vision-Language Models*（CVPR 2024）：论文 §2.2 的直接相关工作。可由 ECALP 论文参考文献进入。

### 3.3 数据准备约定

官方 README 将特征按 `features/<任务>/<模型>/` 组织。首轮目标文件结构为：

```text
ECALP/
├── main.py
├── utils/
│   └── iterative_label_propagation.py
├── features/                       # 外部下载，不提交 Git
│   └── fine_grained/
│       └── RN/
│           ├── DTD.pt
│           ├── eurosat.pt           # 与下载目录中实际文件名逐一核实
│           └── few_shots/
│               └── DTD.pt
├── experiments/                    # 本组新增：独立实验脚本（建议）
├── results/                        # 本组新增：小体积 CSV、图表、日志（建议）
└── docs/
    └── PROJECT_PLAN.md            # 本文件（建议路径）
```

官方 `main.py` 读取 `image_features`、`text_features`、`labels`；Few-shot 文件读取 `image_features`、`labels`。下载后必须检查文件是否完整、张量形状、设备、标签范围及文件名。**以上示例路径来自源码的路径拼接规则；Google Drive 文件是否能全部正常下载尚未进行实机验证。**

原仓库 `.gitignore` 已忽略 `/features/*` 和 `/data/` 等目录；不要把 `.pt` 特征、原始大数据集、模型权重提交到 Git。小型统计表、命令记录与最终图片可以提交。

### 3.4 官方运行命令（先按原版测试）

```bash
# RN50, DTD, Zero-shot
python main.py --task fine_grained --datasets DTD --clip_model RN --mode ZS

# RN50, EuroSAT, Zero-shot（仓库 README 名称为 eurosat）
python main.py --task fine_grained --datasets eurosat --clip_model RN --mode ZS

# RN50, DTD, 16-shot
python main.py --task fine_grained --datasets DTD --clip_model RN --mode FS --num_shots 16
```

官方推荐单 GPU RTX 3090 24GB，测试环境 PyTorch 1.13.1；这属于推荐配置而不是已验证的最低配置。**先跑官方程序，不先改算法**。不同任务各自启动新进程，避免模块级图状态污染后续实验。

## 4. 四人分工（每人一个研究问题 + 一组实验 + 一页 PPT 素材）

> 成员暂记 **A/B/C/D**，领取任务后替换成姓名。A 承担一次公共环境准备；其余成员先用论文、源码及小型模拟特征独立完成设计，等 A 提供数据后并行跑实验。

### A — 数据、环境与主结果（`feat/data-baseline`）

**负责问题**：原始 CLIP 与 ECALP 完整方法的表现差多少？

**必读**：ECALP §4.1–4.2、Table 1；仓库 `README.md`、`main.py`。

**工作**：
1. 下载 DTD / EuroSAT RN50 特征、核对张量与目录；配置一套共享运行环境。
2. 保持官方实现不变，运行 DTD 和 EuroSAT 的 Zero-shot ECALP。
3. 在同批图像与文本特征上计算原始 CLIP 余弦相似度分类准确率。
4. 整理标准运行命令、数据获取说明和统一结果格式。

**交付**：`docs/SETUP.md`、环境依赖记录、`results/main_results.csv`、主结果对比图（PNG）、一段 200 字左右分析。**验收**：他人用相同数据和命令可重复得到对应的结果；所有后续实验复用其统一环境与数据。

### B — 动态图构建消融（`exp/graph-ablation`）

**负责问题**：利用历史测试图像之间的连接，是否真的改善分类？

**必读**：ECALP §3.3（Eq. 3–6）、Appendix A.2.1–A.2.2；`dynamic_graph_expansion()` 及 KNN / 邻接矩阵构造逻辑。

**工作**：
1. 画出文本节点、测试节点、可选 Few-shot 节点及其连接关系。
2. 基于原实现添加明确的消融开关：**完整图 vs 禁用测试图像—测试图像连接**，保持文本边、重加权、标签传播等设置不变。
3. 在 DTD Zero-shot 上对比结果；如时间允许增加 EuroSAT。

**交付**：独立实验脚本或最小修改、`results/graph_ablation.csv`、一张简化构图图、一张准确率对比图、200 字左右结论。**验收**：两组实验可用明确开关切换，能确认只变化图像间连接机制。

**注意**：不要未经验证直接将 `--k_image 0` 当成合法消融；如果原实现依赖 `topk` 或缓存结构，需要专门处理无图像邻居路径。

### C — 上下文感知边权重消融（`exp/context-reweight`）

**负责问题**：使用类别文本的统计信息重加权相似度，是否比原始特征相似度更有效？

**必读**：ECALP §3.4（Eq. 7–8）、Table 4；源码 `edge_reweighting()`。

**工作**：
1. 理解文本原型在特征维度上的方差如何参与边权计算。
2. 设置 **原始相似度 vs Context-aware Re-weighting**，其余图构建和传播参数保持一致。
3. 在 DTD、EuroSAT Zero-shot 上运行消融，并讨论增益差异。

**交付**：独立实验脚本或最小修改、`results/reweight_ablation.csv`、一张机制示意图、一张对比图、200 字左右分析。**验收**：重加权开关可复现，明确记录实际改动的边类型。

**注意**：Zero-shot 不包含 Few-shot 方差路径；不要将关闭文本重加权错误地表述为关闭所有 Few-shot 机制。

### D — 标签传播与 Few-shot 对比（`exp/label-propagation`）

**负责问题**：少量真实标签能给图传播带来多少额外收益？

**必读**：ECALP §3.2、§3.5（Eq. 1–2、Algorithm 1）、§4.3，Appendix A.2.5；源码 `iterative_label_propagation()` 与 `main.py` 的 FS 模式。

**工作**：
1. 理解迭代传播、已标记文本/样本节点的标签重置，以及历史伪标签。
2. 在 DTD 上对比 Zero-shot、1-shot、16-shot（至少完成 16-shot）；使用作者给出的统一超参数。
3. 如时间允许，额外对比传播迭代次数 `T=1` 和 `T=3`，观察准确率与运行时间的权衡。

**交付**：`results/fewshot_results.csv`、一张标签传播流程图、一张 Shot 数或迭代次数对比图、200 字左右分析。**验收**：Zero-shot/Few-shot 的数据来源与使用标签范围清楚，**不把测试集真实标签送入推理**。

## 5. Git 分支和 PR 协作安排

当前 fork 公开页面显示默认分支为 **`master`**。下面四条实验分支是**计划创建的分支，不表示已存在**。

| 分支 | 负责人 | 主要改动 | 目标合并到 |
|---|---|---|---|
| `master` | 组长维护 | 已验证的稳定版本；仅通过 PR 更新 | — |
| `feat/data-baseline` | A | 数据说明、环境、Baseline、主结果 | `master` |
| `exp/graph-ablation` | B | 图连接消融、结果、说明 | `master` |
| `exp/context-reweight` | C | 重加权消融、结果、说明 | `master` |
| `exp/label-propagation` | D | Few-shot / 迭代实验、结果、说明 | `master` |

**推荐协作流程**：

1. 组长先将此文件放入 `docs/PROJECT_PLAN.md` 并提交到 `master`，确定共同的实验接口和结果文件名。
2. A 最先合入基础环境与 Baseline PR（至少包括数据目录、运行命令和结果记录约定）。B/C/D 不必等 A 读完论文，可先做独立的代码理解和小规模测试。
3. 每人从最新 `master` 创建个人分支，只在个人分支修改，完成后发 Pull Request，附**运行命令、修改说明、结果文件与图表**。
4. B、C 都可能修改 `utils/iterative_label_propagation.py`；尽量把开关与实验入口隔离到 `experiments/`，必要时让较早合并的 PR 先提供通用配置开关，后续分支及时同步 `master`，避免互相覆盖。
5. 每个 PR 至少由另一名组员检查，并确认原始 ECALP 默认路径不受影响。合并前比较与官方 Baseline 是否一致，合并后执行一次最小回归测试。
6. `master` 保持可运行状态。不要提交大数据文件，也不要直接覆盖原论文的实验数值。

```bash
# 克隆本组仓库
git clone https://github.com/Dzz2004/ECALP.git
cd ECALP

# 按实际领取任务创建个人分支；下面以 B 为例
git switch master
git pull origin master
git switch -c exp/graph-ablation

# 完成修改后
git add .
git commit -m "exp: add graph ablation"
git push -u origin exp/graph-ablation
# 随后在 GitHub 建 PR，目标分支选 master
```

原作者仓库可作为只读上游参考：`https://github.com/Yushu-Li/ECALP`。如需同步上游，可由组长统一操作并检验，避免在实验期间不经讨论地改变上游代码版本。

**每个结果最少记录**：`dataset, clip_model, mode, num_shots, variant, accuracy, command, commit, notes`。论文成绩应标注为 `paper_reference`，本组跑出的结果标注为 `ours`，不可混用。建议记录运行设备及 PyTorch 版本。

## 6. 整体推进节奏与汇总

| 阶段 | 目标 | 产物 | 通过条件 |
|---|---|---|---|
| **阶段 1：资料与环境** | 读各自部分，下载特征，跑通官方程序 | 数据路径、命令、统一结果格式 | DTD RN50 ZS 可以完整执行 |
| **阶段 2：复现与消融** | A 主结果；B 图消融；C 重加权；D Few-shot | 各自 CSV、图、实验说明 | 每人至少一组有效对照 |
| **阶段 3：汇报整合** | 同一实验表述、结论和局限 | 课程论文、PPT、可运行仓库 | 复现结果有出处，图表口径一致 |

可由组长按照课程最后期限倒排时间，**无需人为规定每个人都做完整论文的所有实验**。

### PPT 建议（约 8 页、约 10 分钟）

1. 标题、问题与研究目标（A）
2. CLIP → CoOp/CoCoOp → Training-free 图传播的动机（A）
3. ECALP 整体方法与动态图（B）
4. 上下文边权与迭代标签传播（C + D）
5. 实验设置：数据、RN50、指标与比较方法（A）
6. 主结果：CLIP vs ECALP（A）
7. 消融 + Few-shot：为何有效（B + C + D）
8. 结论、局限与工作分工（共同汇总）

### 课程论文建议（不超过 2000 字）

- 背景与研究问题（约 300 字）
- ECALP 方法：三个模块（约 650 字）
- 实验设置与复现结果（约 400 字）
- 消融实验及分析（约 450 字）
- 结论与局限（约 200 字）

最终提交按课程要求包含**论文、代码、PPT**，成员贡献明确并按贡献排序；口头汇报为自愿。项目写作不需要逐字照搬原论文，更应突出**“为什么这样做—我们如何验证—观察到什么”**。

## 7. 最终交付检查表

- [ ] 本文件、参考链接、数据下载说明已提交
- [ ] 数据格式与运行环境已确认；大文件未进入 Git
- [ ] DTD RN50 Zero-shot：CLIP 和 ECALP 均已运行
- [ ] EuroSAT RN50 Zero-shot：CLIP 和 ECALP 均已运行（或明确说明缩减原因）
- [ ] 图连接消融完成，附代码开关和图表
- [ ] 上下文重加权消融完成，附代码开关和图表
- [ ] DTD 16-shot 结果完成；1-shot 为推荐扩展
- [ ] 所有结果有命令、配置、Git commit 和运行日志
- [ ] 四人 PR 合并并复核，`master` 可运行
- [ ] PPT、论文及贡献说明完成

---

*说明：本方案依据 ECALP 论文与官方仓库当前公开内容制定。Google Drive 特征可否顺利下载、实际 GPU 峰值显存和运行时间尚未实机验证；Git 个人分支及实验目录均属于建议方案，尚未在远端自动创建。*
