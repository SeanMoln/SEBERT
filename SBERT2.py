import os
import csv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import pandas as pd
import pandas as pd
from datasets import Dataset
# 加载模型和分词器
model = SentenceTransformer("all-mpnet-base-v2")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

# 设置滑动视窗大小和步长
window_size = 382  # 需比模型输入最大序列长度 - 2
step_size = 191  # 一般设置为 window_size 的一半

# 設定滑動視窗大小和步長
window_size = 382    #需比模型輸入最大序列長度-2 
step_size = 191  # 一般設定為 window_size 的一半
total_similarity = 0
window_count = 0 # 紀錄步數
max_similarity = -1  # 初始化為一個非常小的值
best_window = None

def sliding_window_tokenize(text, tokenizer, window_size, step_size):
    tokens = tokenizer.tokenize(text)
    num_tokens = len(tokens)
    windows = []
    for i in range(0, num_tokens, step_size):
        window_tokens = tokens[i:i + window_size]
        if len(window_tokens) < window_size:  # 確定視窗內的tokens數量足夠
            break
        windows.append(tokenizer.convert_tokens_to_string(window_tokens))
    return windows
def read_csv_to_long_text(csv_file):
    print(f"Reading CSV file: {csv_file}")
    
    df = pd.read_csv(csv_file, delimiter='\t', header=None, names=['audio_file', 'transcription'])
 
    
    print("DataFrame read successfully:")
    print(df.head(5))  # 打印前5行数据

    # 确保数据类型一致
    df['audio_file'] = df['audio_file'].astype(str)
    df['transcription'] = df['transcription'].astype(str)

    # 将所有转录文本合并成一个长文本
    long_text = ' '.join(df['transcription'].tolist())
    return long_text

# 讀取文本1


text1_path = r"C:\Users\User\Downloads\SBERT\demo\gpt-4o\traditional_chinese.csv"
text2_path = r"C:\Users\User\Downloads\SBERT\demo\transformer-Helsinki-NLP\traditional_chinese.csv"
text1 = read_csv_to_long_text(text1_path)
emb1 = model.encode(text1)

# 讀取文本2並進行滑動視窗切分


text2 = read_csv_to_long_text(text2_path)
windows2 = sliding_window_tokenize(text2, tokenizer, window_size, step_size)

for i, window2 in enumerate(tqdm(windows2, desc="Processing text2 windows")):
    emb2 = model.encode(window2)
    cos_sim = util.cos_sim(emb1, emb2).item()

    # 更新總和與視窗統計
    total_similarity += cos_sim
    window_count += 1

    # 檢查是否為最高相似度的視窗
    if cos_sim > max_similarity:
        max_similarity = cos_sim
        best_window = window2

# 計算平均相似度
if window_count > 0:
    average_similarity = total_similarity / window_count
else:
    average_similarity = 0  # 處理無視窗的情況

print("Max Cosine-Similarity:", max_similarity)
print("Best matching window from text2:", best_window)
print("Average Cosine-Similarity:", average_similarity)