import csv
from transformers import pipeline

# 設定 LLaMA 翻譯模型（假設模型已經微調）
translator = pipeline("translation", model="huggingface/llama3-model-name")  # 替換為實際的模型名稱

# 讀取 TSV 檔案，並用 LLaMA 翻譯每一行文本
def translate_text_with_llama(text, target_language="zh"):
    """用 LLaMA 模型翻譯文本"""
    translated_text = translator(text, src_lang="en", tgt_lang=target_language)[0]['translation_text']
    return translated_text

# 讀取 TSV 檔案並翻譯
def translate_tsv(input_file, output_file, target_language="zh-cn"):
    with open(input_file, mode='r', encoding='utf-8') as infile, \
         open(output_file, mode='w', encoding='utf-8', newline='') as outfile:

        # 讀取 TSV 檔案
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')

        # 遍歷每一行並翻譯
        temp = 0
        for row in reader:
            if temp < 30:
                temp += 1
            # 假設需要翻譯的是第一欄
            if row:  # 確保行不為空
                original_text = row[0]
                translated_text = translate_text_with_llama(original_text, target_language)

                # 將翻譯後的文本寫入到新的檔案中
                row.append(translated_text)  # 可以選擇在原有欄位後新增翻譯結果
                writer.writerow(row)
                print(f"Translated: {original_text} -> {translated_text}")
            else:
                break

# 設定檔案路徑
input_file = 'SFC.tsv'  # 替換為你的 TSV 檔案
output_file = 'translated_file_llama.tsv'  # 翻譯後的輸出檔案

# 翻譯 TSV 檔案
translate_tsv(input_file, output_file, target_language="zh-cn")
