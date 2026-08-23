# -*- coding: utf-8 -*-
"""
方向2：文本长度分层实验 (Length Ablation Experiment)
验证文本长度对谣言检测性能的影响
"""

import os
import json
import re
import warnings
import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

# 路径配置（基于脚本所在目录，便于在仓库任意位置运行）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(BASE_DIR), 'figures')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 复用之前的加载和清洗函数 ==========
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

def extract_text_features(df, text_column='text'):
    df['clean_text'] = df[text_column].astype(str).apply(clean_text)
    df['text_len'] = df['clean_text'].apply(len)
    return df

# ========== 长度分层实验 ==========
def length_ablation_experiment(df, dataset_name="CED", max_features=5000):
    """
    按文本长度分层，评估随机森林在各层上的性能
    """
    print(f"\n{'='*60}")
    print(f"长度分层实验: {dataset_name}")
    print(f"{'='*60}")
    
    # 1. 提取特征（只需要文本和长度）
    df = extract_text_features(df, 'text')
    
    # 2. 定义长度区间（根据CED分布手动调整：0-50, 50-80, 80-120, 120-180, 180+）
    # CED平均约100字，中位数约117
    bins = [0, 50, 80, 120, 180, 1000]
    labels = ['0-50', '50-80', '80-120', '120-180', '180+']
    df['len_group'] = pd.cut(df['text_len'], bins=bins, labels=labels, right=False)
    
    # 统计各区间样本数
    group_counts = df['len_group'].value_counts().sort_index()
    print("\n📊 各长度区间样本分布：")
    for group in labels:
        count = group_counts.get(group, 0)
        rumor_count = len(df[(df['len_group'] == group) & (df['label'] == 1)])
        non_count = len(df[(df['len_group'] == group) & (df['label'] == 0)])
        print(f"  {group}字: {count} 条 (谣言: {rumor_count}, 非谣言: {non_count})")
    
    # 3. 逐层训练评估（跳过样本过少的层）
    results = []
    overall_f1 = None
    
    for group in labels:
        subset = df[df['len_group'] == group].copy()
        if len(subset) < 30:
            print(f"\n⚠️ {group} 样本不足30条，跳过")
            continue
        
        # 准备数据
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), 
                                     tokenizer=lambda x: jieba.lcut(x))
        X = vectorizer.fit_transform(subset['clean_text'])
        y = subset['label']
        
        # 若某一类样本太少，跳过
        if len(set(y)) < 2:
            print(f"\n⚠️ {group} 只有单一类别，跳过")
            continue
        
        # 划分训练/测试（80/20，分层）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练随机森林
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # 计算指标
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, 
                                       target_names=['Non-rumor', 'Rumor'],
                                       labels=[0, 1],
                                       zero_division=0, 
                                       output_dict=True)
        f1_rumor = report.get('Rumor', {}).get('f1-score', 0.0)
        precision_rumor = report.get('Rumor', {}).get('precision', 0.0)
        recall_rumor = report.get('Rumor', {}).get('recall', 0.0)
        
        # 保存整体F1作为对比基准（第一次遇到时记录）
        if overall_f1 is None:
            # 用全量数据训练一个整体F1
            X_all = vectorizer.fit_transform(df['clean_text'])
            y_all = df['label']
            X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
            )
            rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_all.fit(X_train_all, y_train_all)
            y_pred_all = rf_all.predict(X_test_all)
            report_all = classification_report(y_test_all, y_pred_all, 
                                               target_names=['Non-rumor', 'Rumor'],
                                               labels=[0, 1],
                                               zero_division=0, 
                                               output_dict=True)
            overall_f1 = report_all.get('Rumor', {}).get('f1-score', 0.0)
            print(f"\n📌 全量数据基线 F1 (谣言类): {overall_f1:.4f}")
        
        results.append({
            'Length Group': group,
            'Sample Count': len(subset),
            'Accuracy': round(acc, 4),
            'Precision (Rumor)': round(precision_rumor, 4),
            'Recall (Rumor)': round(recall_rumor, 4),
            'F1 (Rumor)': round(f1_rumor, 4)
        })
        print(f"  {group}字: Acc={acc:.4f}, F1={f1_rumor:.4f} (样本数: {len(subset)})")
    
    # 4. 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 5. 可视化：F1 vs 长度
    if len(results_df) > 0:
        plt.figure(figsize=(10, 6))
        # 柱状图或折线图
        groups = results_df['Length Group'].tolist()
        f1_scores = results_df['F1 (Rumor)'].tolist()
        sample_counts = results_df['Sample Count'].tolist()
        
        # 绘制折线图
        plt.plot(groups, f1_scores, marker='o', linestyle='-', linewidth=2, markersize=8, color='darkorange', label='F1 Score')
        
        # 添加整体基线
        if overall_f1:
            plt.axhline(y=overall_f1, color='steelblue', linestyle='--', linewidth=1.5, label=f'Overall F1 = {overall_f1:.3f}')
        
        # 在数据点上标注样本数
        for i, (g, f, cnt) in enumerate(zip(groups, f1_scores, sample_counts)):
            plt.annotate(f'n={cnt}', (g, f), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
        
        plt.xlabel('文本长度区间 (字符数)', fontsize=12)
        plt.ylabel('F1 分数 (谣言类)', fontsize=12)
        plt.title(f'文本长度对谣言检测性能的影响 ({dataset_name} 数据集)', fontsize=14)
        plt.ylim(0.5, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'length_ablation_f1.png'), dpi=150)
        plt.show()
        
        # 额外：柱状图显示准确率和F1对比
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(groups))
        width = 0.35
        ax.bar(x - width/2, results_df['Accuracy'], width, label='Accuracy', color='steelblue')
        ax.bar(x + width/2, results_df['F1 (Rumor)'], width, label='F1 (Rumor)', color='darkorange')
        ax.set_xlabel('文本长度区间 (字符数)')
        ax.set_ylabel('分数')
        ax.set_title('不同文本长度区间下的模型性能对比')
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'length_ablation_bar.png'), dpi=150)
        plt.show()
    
    # 保存结果
    results_df.to_csv(os.path.join(FIG_DIR, 'length_ablation_results.csv'), index=False)
    print(f"\n✅ 结果已保存至 length_ablation_results.csv 和 figures/ 目录")
    return results_df

if __name__ == "__main__":
    base_dir = os.path.join(BASE_DIR, '..', 'data')
    ced_path = os.path.join(base_dir, "CED_Dataset/")
    
    print("加载 CED 数据集...")
    df = load_ced_data(ced_path)
    
    # 运行长度分层实验
    results = length_ablation_experiment(df, dataset_name="CED")
    
    print("\n" + "="*60)
    print("长度分层实验结果汇总：")
    print("="*60)
    print(results.to_string(index=False))