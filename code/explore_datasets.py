import os
import json
import pandas as pd

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

print("="*50)
print("1. 查看 LTCR.csv 的前几行（如果有）")
print("="*50)
ltcr_csv = os.path.join(base_dir, "LTCR/data/LTCR.csv")
if os.path.exists(ltcr_csv):
    df = pd.read_csv(ltcr_csv, nrows=3)
    print("列名:", df.columns.tolist())
    print(df)
else:
    print("LTCR.csv 不存在，查看 data 子文件夹下的文件")
    data_sub = os.path.join(base_dir, "LTCR/data/data")
    if os.path.exists(data_sub):
        print("data/data 目录下的文件:", os.listdir(data_sub))
        # 尝试读取 dev.txt
        dev_path = os.path.join(data_sub, "dev.txt")
        if os.path.exists(dev_path):
            with open(dev_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"dev.txt 前3行:")
                for i, line in enumerate(lines[:3]):
                    print(f"  第{i+1}行: {line[:200]}")

print("\n" + "="*50)
print("2. 查看 CHECKED 中的一个 JSON 文件结构")
print("="*50)
fake_dir = os.path.join(base_dir, "CHECKED/dataset/fake_news")
if os.path.exists(fake_dir):
    json_files = [f for f in os.listdir(fake_dir) if f.endswith('.json')]
    if json_files:
        sample_file = os.path.join(fake_dir, json_files[0])
        with open(sample_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"样例文件: {json_files[0]}")
        print("顶层键:", list(data.keys()))
        # 如果数据是嵌套的，展开看看
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"  {key} 的子键: {list(value.keys())[:5]}")
            elif isinstance(value, list):
                print(f"  {key} 的长度: {len(value)}")
            else:
                print(f"  {key}: {str(value)[:100]}")