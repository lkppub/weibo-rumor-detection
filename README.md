# Chinese Rumor Detection on Weibo

> Rumor detection for Chinese social media — from classical ML baselines to short-text optimization and a deep-learning reference. Course project for *Data Warehousing and Data Mining* (2026).

[中文版 / Chinese](README_ZH.md)

## Overview

This project systematically evaluates rumor-detection methods on Chinese social-media text (Sina Weibo). It:

- Builds classical baselines — **Logistic Regression, Naive Bayes, Random Forest** — on TF-IDF plus **textual, propagation, and user** features;
- Runs a **length-stratified ablation** to show where short microblogs fail;
- Designs **three incremental optimizations for short text**: threshold tuning → feature enhancement → a length-aware ensemble;
- Compares against a **fine-tuned BERT-base-chinese** model as a deep-learning reference;
- Evaluates generalization across three public datasets and analyzes error patterns.

## Key Results

| Aspect | Result |
|---|---|
| Cross-dataset F1 (Random Forest) | CED **0.838** · CHECKED **0.955** · LTCR **0.991** |
| Short text is the bottleneck | CED F1: 0–50 chars **0.594** → 120–180 chars **0.828** |
| Length-aware ensemble (CED) | every interval F1 ≥ **0.800** (best relative gain **+49.0%**) |
| BERT reference (CED) | overall F1 **0.902** |

The three short-text optimizations raise per-interval F1 to at least 0.800 across all length bins (from a baseline that dipped to 0.537), and the across-interval standard deviation drops from 0.121 to 0.034. Error analysis shows the dominant failure mode is **missed rumors** (80.2% of errors are false negatives), concentrated in very short posts, news-disguised rumors, and low-emotion claims.

## Methods

```
raw posts ──► clean (URLs/@/#/whitespace) ──► Jieba tokenization ──► TF-IDF (1–2 gram)
                                          └──► numeric features: length, punctuation, SnowNLP sentiment
                          ┌── CED only ──► propagation: interactions, unique users, time span
                          └── CED only ──► user profile: followers, verification, posting history
                                    └──► LR / NB / RF  ──► length-aware threshold ensemble
                                              └──► BERT-base-chinese (reference)
```

Evaluation protocol: stratified 80/20 split, fixed seed **42**, rumor-class **F1** as the primary metric (precision/recall/accuracy also reported).

## Datasets

All datasets are public and third-party. They are **not** bundled in this repository — see [Getting Started](#getting-started) to download.

| Dataset | Description | Source |
|---|---|---|
| **CED** | Sina Weibo short text with reposts, comments, and user profiles (3,387 posts) | [thunlp/Chinese_Rumor_Dataset](https://github.com/thunlp/Chinese_Rumor_Dataset) |
| **LTCR** | Long-text Chinese rumors in news style (2,247 posts after label filtering) | [Enderfga/DoubleCheck](https://github.com/Enderfga/DoubleCheck) |
| **CHECKED** | COVID-19 Chinese fake news on Weibo (2,104 posts) | [cyang03/CHECKED](https://github.com/cyang03/CHECKED) |

## Repository Layout

```
├── code/               # 8 experiment scripts (cross-dataset, length ablation, optimization, feature analysis)
├── figures/            # result charts and CSV summary tables
├── data/               # datasets (downloaded via scripts/fetch_datasets.py, gitignored)
├── scripts/            # fetch_datasets.py — one-command dataset download
├── requirements.txt
├── LICENSE             # MIT
└── README_EN.md / README_ZH.md
```

## Getting Started

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. download the three public datasets into ./data
python scripts/fetch_datasets.py

# 3. run experiments
python code/cross_dataset_comparison.py   # cross-dataset F1 (LR / NB / RF)
python code/length_ablation.py            # length-stratified ablation on CED
python code/optimization_strategies.py    # short-text optimization (3 strategies)
python code/rumor_analysis_advanced.py    # propagation + user features & importance
```

`scripts/fetch_datasets.py` shallow-clones each dataset and copies only the needed subtree, so the repository stays clean.

## Results Gallery

| Feature profile (rumor vs. non-rumor) | Optimization comparison | Cross-dataset comparison |
|---|---|---|
| [figures/radar chart.png](figures/radar%20chart.png) | [figures/optimization_comparison.png](figures/optimization_comparison.png) | [figures/cross_dataset_comparison.png](figures/cross_dataset_comparison.png) |

More charts live in [`figures/`](figures/), including length ablation, sentiment density, feature importance, and the feature-fusion comparison.

## Notes

- **BERT** is positioned as a deep-learning **upper-bound reference**; the practical focus of this project is the classical pipeline plus short-text optimization.
- The `data/` directory is gitignored — data is downloaded locally, never committed.
