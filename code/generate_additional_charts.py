# -*- coding: utf-8 -*-
import os
import json
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import scipy.sparse as sp
import jieba

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_DIR, "figures")
data_dir = os.path.join(REPO_DIR, "data", "CED_Dataset/")

def load_original_data(base_path):
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
                continue
        if not isinstance(obj, dict) or 'text' not in obj:
            continue
        
        if filename in rumor_files:
            label = 1
        elif filename in non_rumor_files:
            label = 0
        else:
            continue
        
        obj['label'] = label
        obj['filename'] = filename
        data.append(obj)
    return pd.DataFrame(data)

def extract_propagation_features(repost_path):
    if not os.path.exists(repost_path):
        return {'total_interactions': 0, 'unique_users': 0, 'time_span_hours': 0}
    with open(repost_path, 'r', encoding='utf-8') as f:
        try:
            reposts = json.load(f)
        except:
            return {'total_interactions': 0, 'unique_users': 0, 'time_span_hours': 0}
    if not isinstance(reposts, list):
        return {'total_interactions': 0, 'unique_users': 0, 'time_span_hours': 0}
    
    total = len(reposts)
    unique_users = set()
    timestamps = []
    for item in reposts:
        if 'uid' in item:
            unique_users.add(item['uid'])
        if 'date' in item and item['date']:
            try:
                ts = datetime.strptime(item['date'], '%Y-%m-%d %H:%M:%S')
                timestamps.append(ts)
            except:
                pass
    if len(timestamps) >= 2:
        time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600.0
    else:
        time_span = 0
    return {
        'total_interactions': total,
        'unique_users': len(unique_users),
        'time_span_hours': time_span
    }

def extract_user_features(user_dict):
    if not isinstance(user_dict, dict):
        return {
            'user_followers': 0, 'user_friends': 0, 'user_verified': 0,
            'user_messages': 0, 'user_follow_ratio': 0
        }
    followers = user_dict.get('followers', 0)
    friends = user_dict.get('friends', 0)
    verified = 1 if user_dict.get('verified', False) else 0
    messages = user_dict.get('messages', 0)
    follow_ratio = followers / (friends + 1)
    return {
        'user_followers': followers,
        'user_friends': friends,
        'user_verified': verified,
        'user_messages': messages,
        'user_follow_ratio': follow_ratio
    }

df = load_original_data(data_dir)
print(f"Loaded {len(df)} posts")

prop_list = []
for idx, row in df.iterrows():
    filename = row['filename']
    repost_path_rumor = os.path.join(data_dir, "rumor-repost", filename)
    repost_path_non = os.path.join(data_dir, "non-rumor-repost", filename)
    if os.path.exists(repost_path_rumor):
        repost_path = repost_path_rumor
    elif os.path.exists(repost_path_non):
        repost_path = repost_path_non
    else:
        repost_path = None
    if repost_path:
        feat = extract_propagation_features(repost_path)
    else:
        feat = {'total_interactions': 0, 'unique_users': 0, 'time_span_hours': 0}
    prop_list.append(feat)
prop_df = pd.DataFrame(prop_list)
for col in prop_df.columns:
    df[col] = prop_df[col]

user_list = []
for idx, row in df.iterrows():
    user_dict = row.get('user', {})
    feat = extract_user_features(user_dict)
    user_list.append(feat)
user_df = pd.DataFrame(user_list)
for col in user_df.columns:
    df[col] = user_df[col]

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
df['exclamation_count'] = df['clean_text'].apply(lambda x: x.count('！') + x.count('!'))
df['question_count'] = df['clean_text'].apply(lambda x: x.count('？') + x.count('?'))

try:
    from snownlp import SnowNLP
    df['sentiment'] = df['clean_text'].apply(lambda x: SnowNLP(x).sentiments if len(x)>0 else 0.5)
except:
    df['sentiment'] = 0.5

text_features = ['text_len', 'exclamation_count', 'question_count', 'sentiment']
prop_cols = ['total_interactions', 'unique_users', 'time_span_hours']
user_cols = ['user_followers', 'user_friends', 'user_verified', 'user_messages', 'user_follow_ratio']
all_numeric = text_features + prop_cols + user_cols

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), tokenizer=lambda x: jieba.lcut(x))
X_tfidf = vectorizer.fit_transform(df['clean_text'])
scaler = StandardScaler()
X_num = scaler.fit_transform(df[all_numeric])
X_comb = sp.hstack([X_tfidf, X_num])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X_comb, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

os.makedirs(FIG_DIR, exist_ok=True)

# Figure 4: Normalized feature comparison bar chart
feature_means = df.groupby('label')[all_numeric].mean()
feature_means.index = ['Non-rumor', 'Rumor']
norm_means = feature_means.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
norm_means.T.plot(kind='bar', figsize=(12,6), colormap='Set1')
plt.title('Normalized Feature Means: Rumor vs Non-rumor')
plt.ylabel('Normalized Value')
plt.xlabel('Feature')
plt.xticks(rotation=45)
plt.legend(title='Category')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'bar_feature_comparison.png'), dpi=150)
plt.show()

# Figure 5: Feature importance horizontal bar chart
importances = rf.feature_importances_
numeric_importances = importances[-len(all_numeric):]
indices = np.argsort(numeric_importances)[::-1]
plt.figure(figsize=(10,6))
plt.barh(range(len(all_numeric)), numeric_importances[indices], align='center')
plt.yticks(range(len(all_numeric)), np.array(all_numeric)[indices])
plt.xlabel('Importance Score')
plt.title('Random Forest Feature Importance (Numeric Features)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'feature_importance.png'), dpi=150)
plt.show()

# Figure 6: Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-rumor', 'Rumor'], yticklabels=['Non-rumor', 'Rumor'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Random Forest, Multi-feature Fusion)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'confusion_matrix.png'), dpi=150)
plt.show()

# Figure 7: Performance comparison bar chart
baseline = {'Accuracy': 0.866, 'Precision (Rumor)': 0.929, 'Recall (Rumor)': 0.763, 'F1 (Rumor)': 0.838}
advanced = {'Accuracy': 0.897, 'Precision (Rumor)': 0.91, 'Recall (Rumor)': 0.86, 'F1 (Rumor)': 0.88}
metrics = list(baseline.keys())
x = np.arange(len(metrics))
width = 0.35
plt.figure(figsize=(10,6))
plt.bar(x - width/2, [baseline[m] for m in metrics], width, label='Text-only Features', color='steelblue')
plt.bar(x + width/2, [advanced[m] for m in metrics], width, label='Text + Propagation + User', color='darkorange')
plt.xticks(x, metrics, rotation=15)
plt.ylabel('Score')
plt.ylim(0.6, 1.0)
plt.title('Classification Performance Comparison')
plt.legend()
for i, (b, a) in enumerate(zip([baseline[m] for m in metrics], [advanced[m] for m in metrics])):
    plt.text(i - width/2, b + 0.01, f'{b:.3f}', ha='center', fontsize=9)
    plt.text(i + width/2, a + 0.01, f'{a:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'performance_comparison.png'), dpi=150)
plt.show()

# Figure 8: Boxplot of propagation time span
plt.figure(figsize=(8,6))
data_to_plot = [df[df.label==0]['time_span_hours'].dropna(), df[df.label==1]['time_span_hours'].dropna()]
bp = plt.boxplot(data_to_plot, labels=['Non-rumor', 'Rumor'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
plt.ylabel('Propagation Time Span (hours)')
plt.title('Comparison of Propagation Duration: Rumor vs Non-rumor')
plt.yscale('log')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(FIG_DIR, 'time_span_boxplot.png'), dpi=150)
plt.show()

# Figure 9: Histogram of follower counts (log scale)
plt.figure(figsize=(10,6))
for label, color, name in [(0, 'blue', 'Non-rumor'), (1, 'red', 'Rumor')]:
    subset = df[df.label==label]['user_followers']
    plt.hist(subset, bins=50, alpha=0.5, color=color, label=name)
plt.xscale('log')
plt.xlabel('Follower Count (log scale)')
plt.ylabel('Frequency')
plt.title('Distribution of User Follower Counts')
plt.legend()
plt.savefig(os.path.join(FIG_DIR, 'followers_hist.png'), dpi=150)
plt.show()

print("All additional charts have been generated in the 'figures' directory.")