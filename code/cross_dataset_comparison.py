# -*- coding: utf-8 -*-
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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', '', text)
    text = re.sub(r'@[\w]+', '', text)
    text = re.sub(r'#[\u4e00-\u9fa5a-zA-Z0-9]+#', '', text)
    text = re.sub(r'\s+', '', text)
    return text

def extract_text_features(df, text_column='text'):
    df['clean_text'] = df[text_column].astype(str).apply(clean_text)
    df['text_len'] = df['clean_text'].apply(len)
    df['exclamation_count'] = df['clean_text'].apply(lambda x: x.count('！') + x.count('!'))
    df['question_count'] = df['clean_text'].apply(lambda x: x.count('？') + x.count('?'))
    try:
        from snownlp import SnowNLP
        df['sentiment'] = df['clean_text'].apply(lambda x: SnowNLP(x).sentiments if len(x)>0 else 0.5)
    except:
        df['sentiment'] = 0.5
    return df

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
    print(f"CED_Dataset: {len(df)} samples (Rumor/Fake: {sum(df.label==1)}, Non-rumor/Real: {sum(df.label==0)})")
    return df

def load_ltcr_data(path):
    csv_path = os.path.join(path, "LTCR.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'text' in df.columns and 'label' in df.columns:
            df['label'] = df['label'].astype(int)
            df = df[df['label'].isin([0, 1])]
            df = df[['text', 'label']].copy()
            print(f"LTCR: {len(df)} samples (Rumor/Fake: {sum(df.label==1)}, Non-rumor/Real: {sum(df.label==0)})")
            return df
        else:
            raise ValueError("LTCR.csv missing 'text' or 'label' column")
    else:
        raise FileNotFoundError(f"File not found: {csv_path}")

def load_checked_data(path):
    data = []
    fake_dir = os.path.join(path, "fake_news")
    real_dir = os.path.join(path, "real_news")
    if os.path.exists(fake_dir):
        for filename in os.listdir(fake_dir):
            if filename.endswith('.json'):
                with open(os.path.join(fake_dir, filename), 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                if 'text' in obj:
                    label = 1 if obj.get('label', 'real') == 'fake' else 0
                    data.append({'text': obj['text'], 'label': label})
    if os.path.exists(real_dir):
        for filename in os.listdir(real_dir):
            if filename.endswith('.json'):
                with open(os.path.join(real_dir, filename), 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                if 'text' in obj:
                    label = 1 if obj.get('label', 'real') == 'fake' else 0
                    data.append({'text': obj['text'], 'label': label})
    if data:
        df = pd.DataFrame(data)
        print(f"CHECKED: {len(df)} samples (Rumor/Fake: {sum(df.label==1)}, Non-rumor/Real: {sum(df.label==0)})")
        return df
    else:
        raise ValueError("CHECKED dataset loading failed: no valid JSON files found")

def evaluate_on_dataset(df, dataset_name, max_features=5000):
    print(f"\n{'='*60}")
    print(f"Evaluating Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    df = extract_text_features(df, 'text')
    
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), 
                                 tokenizer=lambda x: jieba.lcut(x))
    X_tfidf = vectorizer.fit_transform(df['clean_text'])
    y = df['label']
    
    unique = set(y.unique())
    if not unique.issubset({0, 1}):
        print(f"Warning: labels contain non-0/1 values {unique}, mapping values >0 to 1")
        y = y.apply(lambda x: 1 if x > 0 else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if name == 'Random Forest':
            analyze_errors(df, y_test, y_pred, dataset_name, model_name="Random Forest")
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, 
                                       target_names=['Non-rumor/Real', 'Rumor/Fake'],
                                       labels=[0, 1],
                                       zero_division=0, 
                                       output_dict=True)
        p = report.get('Rumor/Fake', {}).get('precision', 0.0)
        r = report.get('Rumor/Fake', {}).get('recall', 0.0)
        f1 = report.get('Rumor/Fake', {}).get('f1-score', 0.0)
        results.append({
            'Dataset': dataset_name,
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision (Rumor)': round(p, 4),
            'Recall (Rumor)': round(r, 4),
            'F1 (Rumor)': round(f1, 4)
        })
        print(f"{name:25s} Acc: {acc:.4f}, P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
    
    return pd.DataFrame(results)

def analyze_errors(df, y_test, y_pred, dataset_name, model_name="Random Forest"):
    print("\n" + "="*70)
    print(f"Error Analysis Report: {dataset_name} (Model: {model_name})")
    print("="*70)
    
    y_test_reset = y_test.reset_index(drop=True)
    y_pred_series = pd.Series(y_pred)
    
    test_indices = y_test.index
    temp_df = df.loc[test_indices].reset_index(drop=True)
    
    errors = y_test_reset[y_test_reset != y_pred_series].index
    correct = y_test_reset[y_test_reset == y_pred_series].index
    
    if len(errors) == 0:
        print("Perfect! No errors.")
        return
    
    error_df = temp_df.loc[errors]
    correct_df = temp_df.loc[correct]
    
    print(f"\nStatistics:")
    print(f"  Total test samples: {len(y_test_reset)}")
    print(f"  Error samples: {len(errors)} ({len(errors)/len(y_test_reset)*100:.2f}%)")
    
    print(f"\nFeature Comparison (Error vs Correct):")
    features = ['text_len', 'exclamation_count', 'question_count', 'sentiment']
    for feat in features:
        if feat in df.columns:
            err_mean = error_df[feat].mean() if len(error_df) > 0 else 0
            corr_mean = correct_df[feat].mean() if len(correct_df) > 0 else 0
            print(f"  {feat}:  {err_mean:.2f} (Error) vs {corr_mean:.2f} (Correct)")
    
    false_negatives = y_test_reset[(y_test_reset == 1) & (y_pred_series == 0)].index
    false_positives = y_test_reset[(y_test_reset == 0) & (y_pred_series == 1)].index
    
    print(f"\nError Type Breakdown:")
    print(f"  False Negatives (Missed rumors): {len(false_negatives)}")
    print(f"  False Positives (Misclassified as rumor): {len(false_positives)}")
    
    if len(false_negatives) > 0:
        print("\nTypical False Negative Cases (Rumor classified as normal):")
        for i, idx in enumerate(list(false_negatives)[:3]):
            text = temp_df.loc[idx, 'clean_text']
            display_text = text[:150] + "..." if len(text) > 150 else text
            print(f"  Case {i+1}: {display_text}")
    
    if len(false_positives) > 0:
        print("\nTypical False Positive Cases (Normal classified as rumor):")
        for i, idx in enumerate(list(false_positives)[:3]):
            text = temp_df.loc[idx, 'clean_text']
            display_text = text[:150] + "..." if len(text) > 150 else text
            print(f"  Case {i+1}: {display_text}")
    
    if len(false_negatives) > 0:
        print("\nHigh-frequency words in missed rumors (Top 10):")
        from collections import Counter
        word_list = []
        for idx in false_negatives[:50]:
            text = temp_df.loc[idx, 'clean_text']
            words = jieba.lcut(text)
            word_list.extend([w for w in words if len(w) > 1])
        counter = Counter(word_list).most_common(10)
        for word, cnt in counter:
            print(f"    {word}: {cnt}")

def plot_cross_dataset_comparison(results_df, save_path='cross_dataset_comparison.png'):
    """
    Generate a grouped bar chart comparing model performance across datasets.
    All labels and annotations are in English.
    """
    # Pivot the data for easier plotting
    datasets = results_df['Dataset'].unique()
    models = results_df['Model'].unique()
    
    # Set up colors for different datasets
    dataset_colors = {
        'CED (Weibo Short)': '#2E86AB',
        'LTCR (Long Text)': '#A23B72', 
        'CHECKED (COVID-19)': '#F18F01'
    }
    
    # Create figure with two subplots: one for F1 and one for Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Metric 1: F1 Score
    ax1 = axes[0]
    x = np.arange(len(models))
    width = 0.25
    
    for i, dataset in enumerate(datasets):
        subset = results_df[results_df['Dataset'] == dataset]
        f1_values = [subset[subset['Model'] == m]['F1 (Rumor)'].values[0] if len(subset[subset['Model'] == m]) > 0 else 0 for m in models]
        ax1.bar(x + i * width, f1_values, width, label=dataset, color=dataset_colors.get(dataset, '#888888'))
    
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('F1 Score (Rumor Class)', fontsize=12)
    ax1.set_title('Cross-Dataset Performance Comparison (F1 Score)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels on bars
    for i, dataset in enumerate(datasets):
        subset = results_df[results_df['Dataset'] == dataset]
        for j, model in enumerate(models):
            val = subset[subset['Model'] == model]['F1 (Rumor)'].values[0] if len(subset[subset['Model'] == model]) > 0 else 0
            if val > 0:
                ax1.text(j + i * width, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Metric 2: Accuracy
    ax2 = axes[1]
    for i, dataset in enumerate(datasets):
        subset = results_df[results_df['Dataset'] == dataset]
        acc_values = [subset[subset['Model'] == m]['Accuracy'].values[0] if len(subset[subset['Model'] == m]) > 0 else 0 for m in models]
        ax2.bar(x + i * width, acc_values, width, label=dataset, color=dataset_colors.get(dataset, '#888888'))
    
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Cross-Dataset Performance Comparison (Accuracy)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(models, rotation=15, ha='right')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels on bars
    for i, dataset in enumerate(datasets):
        subset = results_df[results_df['Dataset'] == dataset]
        for j, model in enumerate(models):
            val = subset[subset['Model'] == model]['Accuracy'].values[0] if len(subset[subset['Model'] == model]) > 0 else 0
            if val > 0:
                ax2.text(j + i * width, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Cross-dataset comparison chart saved as: {save_path}")
    plt.show()
    
    # Also generate a separate heatmap-style comparison table using seaborn
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    
    # Create a pivot table for F1 scores
    pivot_f1 = results_df.pivot(index='Dataset', columns='Model', values='F1 (Rumor)')
    pivot_acc = results_df.pivot(index='Dataset', columns='Model', values='Accuracy')
    
    # Plot heatmap for F1
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='YlOrRd', 
                linewidths=0.5, ax=ax3, cbar_kws={'label': 'F1 Score'})
    ax3.set_title('Cross-Dataset F1 Score Comparison (Heatmap)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Model', fontsize=12)
    ax3.set_ylabel('Dataset', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(save_path), 'cross_dataset_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Cross-dataset heatmap saved as: cross_dataset_heatmap.png")
    plt.show()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_DIR = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(REPO_DIR, "data")
    FIG_DIR = os.path.join(REPO_DIR, "figures")
    ced_path = os.path.join(DATA_DIR, "CED_Dataset/")
    ltcr_path = os.path.join(DATA_DIR, "LTCR/data")
    checked_path = os.path.join(DATA_DIR, "CHECKED/dataset")
    
    print("Loading datasets...")
    ced_df = load_ced_data(ced_path) if os.path.exists(ced_path) else None
    ltcr_df = load_ltcr_data(ltcr_path) if os.path.exists(ltcr_path) else None
    checked_df = load_checked_data(checked_path) if os.path.exists(checked_path) else None
    
    datasets = []
    if ced_df is not None:
        datasets.append(('CED (Weibo Short)', ced_df))
    if ltcr_df is not None:
        datasets.append(('LTCR (Long Text)', ltcr_df))
    if checked_df is not None:
        datasets.append(('CHECKED (COVID-19)', checked_df))
    
    if not datasets:
        print("No datasets loaded. Please check paths and file formats.")
        exit()
    
    all_results = []
    for name, df in datasets:
        result = evaluate_on_dataset(df, name)
        all_results.append(result)
    
    final_df = pd.concat(all_results, ignore_index=True)
    print("\n" + "="*60)
    print("Cross-Dataset Performance Comparison")
    print("="*60)
    print(final_df.to_string(index=False))
    
    final_df.to_csv(os.path.join(FIG_DIR, 'cross_dataset_comparison.csv'), index=False)
    print("\nResults saved to: cross_dataset_comparison.csv")
    
    # Generate cross-dataset comparison charts
    plot_cross_dataset_comparison(final_df, save_path=os.path.join(FIG_DIR, 'cross_dataset_comparison.png'))