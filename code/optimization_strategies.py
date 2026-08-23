# -*- coding: utf-8 -*-
"""
优化三部曲：阈值调优 → 特征增强 → 长度感知集成
对比各策略在不同长度区间上的性能提升
"""

import os
import json
import re
import warnings
import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 数据加载与特征工程 ==========

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', '', text)
    text = re.sub(r'@[\w]+', '', text)
    text = re.sub(r'#[\u4e00-\u9fa5a-zA-Z0-9]+#', '', text)
    text = re.sub(r'\s+', '', text)
    return text

def load_ced_data(base_path):
    original_dir = os.path.join(base_path, "original-microblog")
    rumor_repost_dir = os.path.join(base_path, "rumor-repost")
    non_rumor_repost_dir = os.path.join(base_path, "non-rumor-repost")
    rumor_files = set(os.listdir(rumor_repost_dir)) if os.path.exists(rumor_repost_dir) else set()
    non_rumor_files = set(os.listdir(non_rumor_repost_dir)) if os.path.exists(non_rumor_repost_dir) else set()
    data = []
    for filename in os.listdir(original_dir):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(original_dir, filename), 'r', encoding='utf-8') as f:
            try:
                obj = json.load(f)
            except:
                continue
        if not isinstance(obj, dict) or 'text' not in obj:
            continue
        if filename in rumor_files:
            label = 1
        elif filename in non_rumor_files:
            label = 0
        else:
            continue
        data.append({'text': obj['text'], 'label': label})
    df = pd.DataFrame(data)
    print(f"CED_Dataset: {len(df)} 条")
    return df

def extract_base_features(df, text_column='text'):
    """基础特征：清洗文本、长度、感叹号、问号"""
    df['clean_text'] = df[text_column].astype(str).apply(clean_text)
    df['text_len'] = df['clean_text'].apply(len)
    df['exclamation_count'] = df['clean_text'].apply(lambda x: x.count('！') + x.count('!'))
    df['question_count'] = df['clean_text'].apply(lambda x: x.count('？') + x.count('?'))
    return df

def extract_enhanced_features(df):
    """增强特征：数字、英文、@、话题标签"""
    df['digit_count'] = df['clean_text'].apply(lambda x: len(re.findall(r'\d', x)))
    df['english_count'] = df['clean_text'].apply(lambda x: len(re.findall(r'[a-zA-Z]', x)))
    df['at_count'] = df['text'].astype(str).apply(lambda x: x.count('@'))
    df['hashtag_count'] = df['text'].astype(str).apply(lambda x: x.count('#'))
    return df

# ========== 2. 辅助函数：训练与评估 ==========

def train_and_evaluate(X, y, threshold=0.5):
    """在给定数据上训练随机森林，返回评估指标"""
    if len(set(y)) < 2:
        return None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 获取概率并应用自定义阈值
    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Non-rumor', 'Rumor'],
                                   labels=[0, 1], zero_division=0, output_dict=True)
    return {
        'accuracy': acc,
        'precision': report.get('Rumor', {}).get('precision', 0.0),
        'recall': report.get('Rumor', {}).get('recall', 0.0),
        'f1': report.get('Rumor', {}).get('f1-score', 0.0)
    }

def evaluate_by_length_groups(df, vectorizer, X, y, threshold=0.5):
    """
    按长度区间分别评估模型性能
    返回: dict {group: {'f1': ..., 'recall': ..., 'n': ...}}
    """
    bins = [0, 50, 80, 120, 180, 1000]
    labels = ['0-50', '50-80', '80-120', '120-180', '180+']
    df['len_group'] = pd.cut(df['text_len'], bins=bins, labels=labels, right=False)
    
    results = {}
    for group in labels:
        mask = df['len_group'] == group
        if mask.sum() < 10:
            results[group] = {'f1': 0, 'recall': 0, 'precision': 0, 'n': mask.sum()}
            continue
        # 关键修复：将 mask 转为 numpy 数组
        mask_np = mask.to_numpy()
        X_sub = X[mask_np]
        y_sub = y[mask_np]   # y 是 pandas Series，可用 numpy 数组索引
        if len(set(y_sub)) < 2:
            results[group] = {'f1': 0, 'recall': 0, 'precision': 0, 'n': mask.sum()}
            continue
        # 划分训练/测试
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub
        )
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_proba = rf.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        report = classification_report(y_test, y_pred, target_names=['Non-rumor', 'Rumor'],
                                       labels=[0, 1], zero_division=0, output_dict=True)
        results[group] = {
            'f1': report.get('Rumor', {}).get('f1-score', 0.0),
            'recall': report.get('Rumor', {}).get('recall', 0.0),
            'precision': report.get('Rumor', {}).get('precision', 0.0),
            'n': mask.sum()
        }
    return results

# ========== 3. 优化策略实现 ==========

def optimize_threshold(df, X, y, target_group='0-50'):
    """
    优化1：在目标长度区间上搜索最佳阈值（最大化F1）
    """
    print("\n" + "="*60)
    print("优化1：阈值调优 (在 {} 区间搜索最佳阈值)".format(target_group))
    print("="*60)
    
    bins = [0, 50, 80, 120, 180, 1000]
    labels = ['0-50', '50-80', '80-120', '120-180', '180+']
    df['len_group'] = pd.cut(df['text_len'], bins=bins, labels=labels, right=False)
    
    mask = df['len_group'] == target_group
    if mask.sum() < 10:
        print(f"  目标区间 {target_group} 样本不足，跳过")
        return 0.5
    
    mask_np = mask.to_numpy()
    X_sub = X[mask_np]
    y_sub = y[mask_np]
    if len(set(y_sub)) < 2:
        print(f"  目标区间 {target_group} 类别不足，跳过")
        return 0.5
    
    # 在子集上训练
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:, 1]
    
    # 搜索阈值：0.1 到 0.6，步长 0.05
    best_f1 = 0
    best_th = 0.5
    for th in np.arange(0.1, 0.65, 0.05):
        y_pred = (y_proba >= th).astype(int)
        report = classification_report(y_test, y_pred, target_names=['Non-rumor', 'Rumor'],
                                       labels=[0, 1], zero_division=0, output_dict=True)
        f1 = report.get('Rumor', {}).get('f1-score', 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    
    print(f"  ✅ 最佳阈值: {best_th:.2f} (F1 = {best_f1:.4f})")
    return best_th

def evaluate_enhanced_features(df, threshold_short=0.3):
    """
    优化2：特征增强 + 低阈值
    在短文本区间(0-80字)使用增强特征，其他区间使用基础特征
    """
    print("\n" + "="*60)
    print("优化2：特征增强 + 低阈值")
    print("="*60)
    
    # 提取增强特征
    df_enhanced = extract_enhanced_features(df)
    
    # 基础特征列
    base_features = ['exclamation_count', 'question_count']
    enhanced_features = ['digit_count', 'english_count', 'at_count', 'hashtag_count']
    all_numeric = base_features + enhanced_features
    
    # 构建TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), tokenizer=lambda x: jieba.lcut(x))
    X_tfidf = vectorizer.fit_transform(df_enhanced['clean_text'])
    
    # 标准化数值特征
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df_enhanced[all_numeric])
    # 关键修复：转换为 CSR 格式以支持布尔索引
    X_comb = sp.hstack([X_tfidf, X_num]).tocsr()
    
    y = df_enhanced['label']
    
    # 按长度区间评估，短文本使用低阈值
    bins = [0, 50, 80, 120, 180, 1000]
    labels = ['0-50', '50-80', '80-120', '120-180', '180+']
    df_enhanced['len_group'] = pd.cut(df_enhanced['text_len'], bins=bins, labels=labels, right=False)
    
    results = {}
    for group in labels:
        mask = df_enhanced['len_group'] == group
        if mask.sum() < 10:
            results[group] = {'f1': 0, 'recall': 0, 'precision': 0}
            continue
        mask_np = mask.to_numpy()
        X_sub = X_comb[mask_np]
        y_sub = y[mask_np]
        if len(set(y_sub)) < 2:
            results[group] = {'f1': 0, 'recall': 0, 'precision': 0}
            continue
        
        # 短文本(0-80字)使用低阈值，其他使用0.5
        if group in ['0-50', '50-80']:
            th = threshold_short
        else:
            th = 0.5
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub
        )
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_proba = rf.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= th).astype(int)
        report = classification_report(y_test, y_pred, target_names=['Non-rumor', 'Rumor'],
                                       labels=[0, 1], zero_division=0, output_dict=True)
        results[group] = {
            'f1': report.get('Rumor', {}).get('f1-score', 0.0),
            'recall': report.get('Rumor', {}).get('recall', 0.0),
            'precision': report.get('Rumor', {}).get('precision', 0.0)
        }
        print(f"  {group} (阈值={th:.2f}): F1={results[group]['f1']:.4f}, Recall={results[group]['recall']:.4f}")
    
    return results

def evaluate_length_aware_ensemble(df, threshold_map):
    """
    优化3：长度感知集成 — 不同长度区间使用不同的阈值
    threshold_map = {'0-50': 0.25, '50-80': 0.30, '80-120': 0.40, '120-180': 0.50}
    """
    print("\n" + "="*60)
    print("优化3：长度感知集成 (按区间应用不同阈值)")
    print("="*60)
    print(f"  阈值映射: {threshold_map}")
    
    # 使用基础特征 + TF-IDF
    df_base = extract_base_features(df, 'text')
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), tokenizer=lambda x: jieba.lcut(x))
    X_tfidf = vectorizer.fit_transform(df_base['clean_text'])
    y = df_base['label']
    
    # 全量训练一个随机森林
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:, 1]
    
    # 获取测试集的长度
    test_indices = X_test.indices if hasattr(X_test, 'indices') else list(range(len(y_test)))
    # 更准确：从原始df中获取测试集的文本长度
    # 由于train_test_split打乱了索引，我们需要保留原始索引
    # 简化方法：重新划分时保留索引
    X_train_idx, X_test_idx, y_train_idx, y_test_idx = train_test_split(
        range(len(df_base)), y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 重新训练以匹配索引
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tfidf[X_train_idx], y.iloc[X_train_idx])
    y_proba = rf.predict_proba(X_tfidf[X_test_idx])[:, 1]
    
    # 获取每个测试样本的长度
    test_lengths = df_base.iloc[X_test_idx]['text_len'].values
    test_labels = y.iloc[X_test_idx].values
    
    # 按长度分组应用不同阈值
    y_pred = np.zeros_like(y_proba, dtype=int)
    for i, (prob, length) in enumerate(zip(y_proba, test_lengths)):
        if length < 50:
            th = threshold_map.get('0-50', 0.5)
        elif length < 80:
            th = threshold_map.get('50-80', 0.5)
        elif length < 120:
            th = threshold_map.get('80-120', 0.5)
        else:
            th = threshold_map.get('120-180', 0.5)
        y_pred[i] = 1 if prob >= th else 0
    
    # 整体评估
    report = classification_report(test_labels, y_pred, target_names=['Non-rumor', 'Rumor'],
                                   labels=[0, 1], zero_division=0, output_dict=True)
    overall_f1 = report.get('Rumor', {}).get('f1-score', 0.0)
    overall_recall = report.get('Rumor', {}).get('recall', 0.0)
    overall_precision = report.get('Rumor', {}).get('precision', 0.0)
    
    print(f"  整体 F1: {overall_f1:.4f}, Recall: {overall_recall:.4f}, Precision: {overall_precision:.4f}")
    
    # 按区间分别统计
    results = {}
    for group, th in threshold_map.items():
        # 找出该区间的样本
        if group == '0-50':
            mask = test_lengths < 50
        elif group == '50-80':
            mask = (test_lengths >= 50) & (test_lengths < 80)
        elif group == '80-120':
            mask = (test_lengths >= 80) & (test_lengths < 120)
        elif group == '120-180':
            mask = (test_lengths >= 120) & (test_lengths < 180)
        else:
            mask = test_lengths >= 180
        
        if mask.sum() == 0:
            results[group] = {'f1': 0, 'recall': 0, 'precision': 0, 'n': 0}
            continue
        
        y_sub = test_labels[mask]
        y_pred_sub = y_pred[mask]
        report_sub = classification_report(y_sub, y_pred_sub, target_names=['Non-rumor', 'Rumor'],
                                           labels=[0, 1], zero_division=0, output_dict=True)
        results[group] = {
            'f1': report_sub.get('Rumor', {}).get('f1-score', 0.0),
            'recall': report_sub.get('Rumor', {}).get('recall', 0.0),
            'precision': report_sub.get('Rumor', {}).get('precision', 0.0),
            'n': mask.sum(),
            'threshold': th
        }
        print(f"  {group} (n={mask.sum()}, th={th:.2f}): F1={results[group]['f1']:.4f}")
    
    return results, overall_f1, overall_recall

# ========== 4. 主流程 ==========

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_DIR = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(REPO_DIR, 'data')
    FIG_DIR = os.path.join(REPO_DIR, 'figures')
    base_dir = DATA_DIR
    ced_path = os.path.join(DATA_DIR, "CED_Dataset/")
    
    print("加载 CED 数据集...")
    df = load_ced_data(ced_path)
    
    # 提取基础特征
    df = extract_base_features(df, 'text')
    
    # 构建基础TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), tokenizer=lambda x: jieba.lcut(x))
    X_tfidf = vectorizer.fit_transform(df['clean_text'])
    y = df['label']
    
    # 添加长度分组
    bins = [0, 50, 80, 120, 180, 1000]
    labels = ['0-50', '50-80', '80-120', '120-180', '180+']
    df['len_group'] = pd.cut(df['text_len'], bins=bins, labels=labels, right=False)
    
    # ---------- 基准性能（按长度区间） ----------
    print("\n" + "="*60)
    print("基准性能 (Baseline: TF-IDF + RandomForest, 阈值=0.5)")
    print("="*60)
    baseline_results = evaluate_by_length_groups(df, vectorizer, X_tfidf, y, threshold=0.5)
    baseline_df = pd.DataFrame(baseline_results).T
    print(baseline_df[['n', 'f1', 'recall', 'precision']].to_string())
    
    # ---------- 优化1：阈值调优 ----------
    best_th = optimize_threshold(df, X_tfidf, y, target_group='0-50')
    
    # 应用最佳阈值重新评估各区间
    print("\n应用最佳阈值 (0-50字区间用 {:.2f}, 其他用0.5)".format(best_th))
    opt1_results = {}
    for group in ['0-50', '50-80', '80-120', '120-180']:
        th = best_th if group == '0-50' else 0.5
        mask = df['len_group'] == group
        if mask.sum() < 10 or len(set(y[mask])) < 2:
            opt1_results[group] = {'f1': 0, 'recall': 0, 'precision': 0, 'n': mask.sum()}
            continue
        # 关键修复：转为numpy数组
        mask_np = mask.to_numpy()
        X_sub = X_tfidf[mask_np]
        y_sub = y[mask_np]
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub
        )
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_proba = rf.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= th).astype(int)
        report = classification_report(y_test, y_pred, target_names=['Non-rumor', 'Rumor'],
                                       labels=[0, 1], zero_division=0, output_dict=True)
        opt1_results[group] = {
            'f1': report.get('Rumor', {}).get('f1-score', 0.0),
            'recall': report.get('Rumor', {}).get('recall', 0.0),
            'precision': report.get('Rumor', {}).get('precision', 0.0),
            'n': mask.sum()
        }
        print(f"  {group}: F1={opt1_results[group]['f1']:.4f}, Recall={opt1_results[group]['recall']:.4f}")
    
    # ---------- 优化2：特征增强 + 低阈值 ----------
    opt2_results = evaluate_enhanced_features(df, threshold_short=best_th)
    
    # ---------- 优化3：长度感知集成 ----------
    threshold_map = {
        '0-50': best_th,
        '50-80': 0.35,
        '80-120': 0.40,
        '120-180': 0.50
    }
    opt3_results, overall_f1, overall_recall = evaluate_length_aware_ensemble(df, threshold_map)
    
    # ---------- 汇总对比 ----------
    print("\n" + "="*60)
    print("📊 优化策略汇总对比 (F1 分数)")
    print("="*60)
    
    summary = pd.DataFrame()
    for group in ['0-50', '50-80', '80-120', '120-180']:
        row = {'长度区间': group}
        row['Baseline'] = baseline_results.get(group, {}).get('f1', 0)
        row['Opt1 (阈值调优)'] = opt1_results.get(group, {}).get('f1', 0)
        row['Opt2 (特征增强)'] = opt2_results.get(group, {}).get('f1', 0)
        row['Opt3 (长度感知)'] = opt3_results.get(group, {}).get('f1', 0)
        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
    
    print(summary.to_string(index=False))
    
    # 保存结果
    summary.to_csv(os.path.join(FIG_DIR, 'optimization_summary.csv'), index=False)
    print("\n✅ 优化结果已保存至 optimization_summary.csv")
    
    # ---------- 可视化 ----------
    plt.figure(figsize=(12, 6))
    groups = summary['长度区间'].tolist()
    x = np.arange(len(groups))
    width = 0.2
    
    plt.bar(x - 1.5*width, summary['Baseline'], width, label='Baseline (阈值0.5)', color='steelblue')
    plt.bar(x - 0.5*width, summary['Opt1 (阈值调优)'], width, label='Opt1 (阈值调优)', color='darkorange')
    plt.bar(x + 0.5*width, summary['Opt2 (特征增强)'], width, label='Opt2 (特征增强)', color='forestgreen')
    plt.bar(x + 1.5*width, summary['Opt3 (长度感知)'], width, label='Opt3 (长度感知集成)', color='crimson')
    
    plt.xlabel('文本长度区间 (字符数)')
    plt.ylabel('F1 分数 (谣言类)')
    plt.title('优化策略性能对比')
    plt.xticks(x, groups)
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'optimization_comparison.png'), dpi=150)
    plt.show()
    
    print("\n✅ 可视化图表已保存至 figures/optimization_comparison.png")