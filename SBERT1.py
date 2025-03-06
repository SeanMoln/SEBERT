import os
import json
import csv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
import pandas as pd
from datasets import Dataset

def process_file(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        # 逐行讀取檔案
        data = []
        for line in file:
            line = line.strip()  # 去除行尾的換行符號或空格
            # 確保行不為空
            data.append(line)
    return data

def file_to_dataset(csv_file):
    print(csv_file)
    df = pd.read_csv(csv_file, delimiter='\t', header=None, names=['audio_file', 'transcription'])
    print("df OK")
    # 將 DataFrame 轉換為字典
    df['audio_file'] = df['audio_file'].astype(str)
    df['transcription'] = df['transcription'].astype(str)
    # df['translate'] = df['translate'].astype(str)
    # 將 DataFrame 轉換為字典
    dataset_dict = {
        'audio_file': df['audio_file'].tolist(),
        'transcription': df['transcription'].tolist(),
        # 'translate': df['translate'].tolist()
    }
    print(df.head(5))
    print("dict OK")
    # 將字典轉換為 Huggingface Dataset
    processed_dataset = Dataset.from_dict(dataset_dict)
    print("processed_dataset OK")
    return processed_dataset

def set(text1_path, text2_path, output_file):
    model = SentenceTransformer("all-mpnet-base-v2")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    # tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    file = file_to_dataset(text1_path)
    print(file)
    file2 = process_file(text2_path)
    file1 = file['transcription']

    print(file1)
    file2 = file2
    # print(file2)
    temp = []
    sum_list = []

    with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        writer.writerow(["文本一", "文本二", "tokens1", "tokens2", "cos_sim"])

        # 計算文本的向量表示（embedding）並將結果寫入 CSV
        for text_1, text_2 in zip(file1, file2):
            print(text_1)
            print("__")
            print(text_2)

            # 分詞
            tokens1 = tokenizer.tokenize(text_1)
            tokens2 = tokenizer.tokenize(text_2)
            print("Number of tokens1:", len(tokens1))
            print("Number of tokens2:", len(tokens2))

            # 計算向量表示（embedding）
            emb1 = model.encode(text_1)
            emb2 = model.encode(text_2)

            # 計算餘弦相似度
            cos_sim = util.cos_sim(emb1, emb2).item()

            # 儲存結果
            temp.append([text_1, text_2, tokens1, tokens2, cos_sim])
            sum_list.append(cos_sim)

            # 寫入到 CSV
            writer.writerow([text_1, text_2, tokens1, tokens2, cos_sim])

    # 計算平均餘弦相似度
    average_cosine_similarity = sum(sum_list) / len(sum_list) if sum_list else 0
    print("Average Cosine-Similarity:", average_cosine_similarity)

a = r"translated_file.tsv"
b = r"nejm.test.zh"
c = "Data.csv"
set(a, b, c)