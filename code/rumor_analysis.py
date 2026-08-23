# -*- coding: utf-8 -*-
import os
import json
import re
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 设置中文显示（如果系统没有中文字体，可以注释掉）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data_from_original_microblog(base_path):
    original_dir = os.path.join(base_path, "original-microblog")
    rumor_repost_dir = os.path.join(base_path, "rumor-repost")
    non_rumor_repost_dir = os.path.join(base_path, "non-rumor-repost")
    
    rumor_files = set(os.listdir(rumor_repost_dir)) if os.path.exists(rumor_repost_dir) else set()
    non_rumor_files = set(os.listdir(non_rumor_repost_dir)) if os.path.exists(non_rumor_repost_dir) else set()
    
    data = []
    for filename in os.listdir(original_dir):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(original_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                obj = json.load(f)
            except:
                print(f"Read failed: {filename}")
                continue
        if not isinstance(obj, dict) or 'text' not in obj:
            continue
        
        if filename in rumor_files:
            label = 1
        elif filename in non_rumor_files:
            label = 0
        else:
            print(f"Skipped {filename}: no corresponding repost file")
            continue
        
        obj['label'] = label
        data.append(obj)
    
    print(f"Successfully loaded {len(data)} posts")
    if len(data) > 0:
        rumor_cnt = sum(1 for d in data if d['label']==1)
        non_cnt = sum(1 for d in data if d['label']==0)
        print(f"Rumors: {rumor_cnt}, Non-rumors: {non_cnt}")
    return pd.DataFrame(data)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_DIR, "figures")
data_dir = os.path.join(REPO_DIR, "data", "CED_Dataset/")
df = load_data_from_original_microblog(data_dir)

if len(df) == 0:
    print("Error: No data loaded. Check directory structure.")
    exit()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', '', text)
    text = re.sub(r'@[\w]+', '', text)
    text = re.sub(r'#[\u4e00-\u9fa5a-zA-Z0-9]+#', '', text)
    text = re.sub(r'\s+', '', text)
    return text

df['clean_text'] = df['text'].astype(str).apply(clean_text)
df['text_len'] = df['clean_text'].apply(len)
df['word_count'] = df['clean_text'].apply(lambda x: len(jieba.lcut(x)))
df['exclamation_count'] = df['clean_text'].apply(lambda x: x.count('！') + x.count('!'))
df['question_count'] = df['clean_text'].apply(lambda x: x.count('？') + x.count('?'))

try:
    from snownlp import SnowNLP
    df['sentiment'] = df['clean_text'].apply(lambda x: SnowNLP(x).sentiments if len(x)>0 else 0.5)
    print("Sentiment computed using SnowNLP")
except ImportError:
    df['sentiment'] = 0.5
    print("SnowNLP not installed, sentiment set to 0.5")

if 'comments' in df.columns:
    df['comment_count'] = df['comments'].fillna(0)
else:
    df['comment_count'] = 0
if 'reposts' in df.columns:
    df['repost_count'] = df['reposts'].fillna(0)
else:
    df['repost_count'] = 0

os.makedirs("figures", exist_ok=True)

# Figure 1: Boxplot of text length
plt.figure(figsize=(8,6))
sns.boxplot(x='label', y='text_len', data=df, palette='Set2')
plt.xticks([0,1], ['Non-rumor', 'Rumor'])
plt.ylabel('Text length (characters)')
plt.title('Comparison of Text Length: Rumor vs Non-rumor')
plt.savefig(os.path.join(FIG_DIR, 'boxplot_text_len.png'), dpi=150, bbox_inches='tight')
plt.show()

# Figure 2: Density plot of sentiment
plt.figure(figsize=(8,6))
sns.kdeplot(data=df, x='sentiment', hue='label', fill=True, common_norm=False, palette='Set1')
plt.xlabel('Sentiment score (0 negative ~ 1 positive)')
plt.ylabel('Density')
plt.title('Sentiment Distribution: Rumor vs Non-rumor')
plt.savefig(os.path.join(FIG_DIR, 'density_sentiment.png'), dpi=150, bbox_inches='tight')
plt.show()

# Figure 3: Word clouds
def generate_wordcloud(texts, title, filename):
    text_all = ' '.join(texts)
    font_path = 'C:/Windows/Fonts/simhei.ttf'
    if not os.path.exists(font_path):
        font_path = 'C:/Windows/Fonts/msyh.ttc'
    wc = WordCloud(font_path=font_path, width=800, height=400,
                   background_color='white', max_words=100).generate(text_all)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=150, bbox_inches='tight')
    plt.show()

rumor_texts = df[df.label==1]['clean_text'].tolist()
non_rumor_texts = df[df.label==0]['clean_text'].tolist()
if rumor_texts:
    generate_wordcloud(rumor_texts, 'Word Cloud - Rumors', 'wordcloud_rumor.png')
if non_rumor_texts:
    generate_wordcloud(non_rumor_texts, 'Word Cloud - Non-rumors', 'wordcloud_non_rumor.png')

# ========== 4. Classification ==========
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), tokenizer=lambda x: jieba.lcut(x))
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Non-rumor','Rumor'], output_dict=True)
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision (Rumor)': report['Rumor']['precision'],
        'Recall (Rumor)': report['Rumor']['recall'],
        'F1 (Rumor)': report['Rumor']['f1-score']
    })
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-rumor','Rumor']))

results_df = pd.DataFrame(results)
print("\nModel Performance Summary:")
print(results_df)

summary = df.groupby('label').agg({
    'text_len': ['mean', 'std', 'median'],
    'exclamation_count': 'mean',
    'question_count': 'mean',
    'sentiment': 'mean',
    'repost_count': 'mean'
}).round(2)
summary.columns = ['Mean Length', 'Std Length', 'Median Length', 'Mean Exclamation', 'Mean Question', 'Mean Sentiment', 'Mean Reposts']
summary.index = ['Non-rumor', 'Rumor']
summary.to_csv(os.path.join(FIG_DIR, 'statistics_summary.csv'))
print("\nStatistics summary saved to figures/statistics_summary.csv")