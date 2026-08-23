# -*- coding: utf-8 -*-
import os
import json
import re
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
    
    print(f"Successfully loaded {len(data)} original posts")
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

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_DIR, "figures")
data_dir = os.path.join(REPO_DIR, "data", "CED_Dataset/")
df = load_original_data(data_dir)
if len(df) == 0:
    print("No data loaded, exiting")
    exit()

prop_features = []
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
    prop_features.append(feat)
prop_df = pd.DataFrame(prop_features)
for col in prop_df.columns:
    df[col] = prop_df[col].values

user_features = []
for idx, row in df.iterrows():
    user_dict = row.get('user', {})
    feat = extract_user_features(user_dict)
    user_features.append(feat)
user_df = pd.DataFrame(user_features)
for col in user_df.columns:
    df[col] = user_df[col].values

print("Propagation and user features added.")

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
    print("Sentiment computed using SnowNLP")
except ImportError:
    df['sentiment'] = 0.5

text_features = ['text_len', 'exclamation_count', 'question_count', 'sentiment']
prop_features_cols = ['total_interactions', 'unique_users', 'time_span_hours']
user_features_cols = ['user_followers', 'user_friends', 'user_verified', 'user_messages', 'user_follow_ratio']
all_numeric_features = text_features + prop_features_cols + user_features_cols

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), tokenizer=lambda x: jieba.lcut(x))
X_tfidf = vectorizer.fit_transform(df['clean_text'])
scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[all_numeric_features])
X_combined = sp.hstack([X_tfidf, X_numeric])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\n=== Random Forest with Propagation + User Features ===")
print(classification_report(y_test, y_pred, target_names=['Non-rumor', 'Rumor']))
acc_advanced = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc_advanced:.4f}")

baseline_acc = 0.8658
print(f"\nPerformance comparison:")
print(f"Text-only Random Forest accuracy: {baseline_acc:.4f}")
print(f"Fused features accuracy: {acc_advanced:.4f}")
print(f"Improvement: {(acc_advanced - baseline_acc)*100:.2f}%")

print("\nTop 10 numeric feature importances:")
feature_names = all_numeric_features
importances = rf.feature_importances_
numeric_importances = importances[-len(all_numeric_features):]
sorted_idx = np.argsort(numeric_importances)[::-1]
for i in sorted_idx[:10]:
    print(f"{feature_names[i]}: {numeric_importances[i]:.4f}")

summary_advanced = df.groupby('label').agg({
    'text_len': 'mean',
    'exclamation_count': 'mean',
    'question_count': 'mean',
    'sentiment': 'mean',
    'total_interactions': 'mean',
    'unique_users': 'mean',
    'time_span_hours': 'mean',
    'user_followers': 'mean',
    'user_verified': 'mean',
    'user_messages': 'mean'
}).round(2)
summary_advanced.index = ['Non-rumor', 'Rumor']
print("\nAdvanced feature statistics:")
print(summary_advanced)
summary_advanced.to_csv(os.path.join(FIG_DIR, 'statistics_summary_advanced.csv'))