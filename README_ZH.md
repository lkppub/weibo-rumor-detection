# 中文微博谣言检测

> 面向中文社交媒体的谣言检测 —— 从经典机器学习基线到短文本优化与深度学习参考。**数据仓库与数据挖掘** 课程项目（2026）。

[English / English](README_EN.md)

## 项目简介

本项目系统评估中文社交媒体（新浪微博）上的谣言检测方法：

- 基于 **TF-IDF + 手工构造的文本 / 传播 / 用户特征**，构建 **逻辑回归、朴素贝叶斯、随机森林** 等经典基线；
- 通过**长度分层消融实验**揭示短微博文本是性能瓶颈；
- 针对短文本设计**三种递进式优化策略**：阈值调优 → 特征增强 → 长度感知集成；
- 与 **fine-tuned BERT-base-chinese** 进行对比，作为深度学习参考上限；
- 在三个公开数据集上评估泛化能力，并进行误差模式分析。

## 主要结果

| 维度 | 结果 |
|---|---|
| 跨数据集 F1（随机森林） | CED **0.838** · CHECKED **0.955** · LTCR **0.991** |
| 短文本是瓶颈 | CED 上 F1：0–50 字 **0.594** → 120–180 字 **0.828** |
| 长度感知集成（CED） | 各长度区间 F1 均 ≥ **0.800**（最高相对提升 **+49.0%**） |
| BERT 参考（CED） | 总体 F1 **0.902** |

三种短文本优化将各长度区间的 F1 从最低 0.537 的基线提升到全部 ≥0.800，且跨区间标准差从 0.121 降至 0.034。误差分析表明主要失效模式是**漏检谣言**（80.2% 的误差为假阴性），集中在极短文本、新闻体伪装的谣言以及弱情感表达三类样本。

## 方法流水线

```
原始微博 ──► 清洗（URL/@/话题标签/空白）──► Jieba 分词 ──► TF-IDF（1–2 gram）
                                         └──► 数值特征：长度、标点、SnowNLP 情感分
                         ┌── 仅 CED ──► 传播特征：互动数、唯一用户数、时间跨度
                         └── 仅 CED ──► 用户画像：粉丝数、认证状态、历史发帖
                                   └──► LR / NB / RF  ──► 长度感知阈值集成
                                             └──► BERT-base-chinese（参考）
```

评估协议：80/20 分层划分、固定随机种子 **42**，以谣言类 **F1** 为主指标（同时报告精确率/召回率/准确率）。

## 数据集

三个数据集均为公开的第三方数据，**不随本仓库提供**，请按下方说明下载。

| 数据集 | 说明 | 来源 |
|---|---|---|
| **CED** | 新浪微博短文本，含转发、评论与用户画像（3,387 条） | [thunlp/Chinese_Rumor_Dataset](https://github.com/thunlp/Chinese_Rumor_Dataset) |
| **LTCR** | 长文本中文谣言（新闻风格，标签过滤后 2,247 条） | [Enderfga/DoubleCheck](https://github.com/Enderfga/DoubleCheck) |
| **CHECKED** | 微博上的 COVID-19 中文假新闻（2,104 条） | [cyang03/CHECKED](https://github.com/cyang03/CHECKED) |

## 仓库结构

```
├── code/               # 8 个实验脚本（跨数据集、长度消融、优化、特征分析）
├── figures/            # 结果图表与 CSV 汇总表
├── data/               # 数据集（运行 fetch 脚本自动下载，已 gitignore）
├── scripts/            # fetch_datasets.py —— 一键下载数据集
├── requirements.txt
├── LICENSE             # MIT
└── README_EN.md / README_ZH.md
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载三个公开数据集到 ./data
python scripts/fetch_datasets.py

# 3. 运行实验
python code/cross_dataset_comparison.py   # 跨数据集 F1（LR / NB / RF）
python code/length_ablation.py            # CED 上的长度分层消融
python code/optimization_strategies.py    # 短文本优化（三种策略）
python code/rumor_analysis_advanced.py    # 传播 + 用户特征及重要度分析
```

`scripts/fetch_datasets.py` 会对每个数据集进行浅克隆，仅复制所需子目录，保持仓库干净。

## 图表展示

| 特征画像（谣言 vs 非谣言） | 优化策略对比 | 跨数据集对比 |
|---|---|---|
| [figures/radar chart.png](figures/radar%20chart.png) | [figures/optimization_comparison.png](figures/optimization_comparison.png) | [figures/cross_dataset_comparison.png](figures/cross_dataset_comparison.png) |

更多图表见 [`figures/`](figures/)，包括长度消融、情感密度、特征重要度与多特征融合对比等。

## 说明

- **BERT** 定位为深度学习的**理论上限参考**；本项目的实践重点是经典流水线 + 短文本优化。
- `data/` 目录已 gitignore——数据仅本地下载，永不提交到仓库。

## 许可

MIT —— 详见 [LICENSE](LICENSE)。
