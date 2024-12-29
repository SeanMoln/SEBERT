import csv

# 開啟並讀取 TSV 檔案
def read_tsv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        # 使用 csv.reader 並設置分隔符為制表符（tab）
        reader = csv.reader(file, delimiter='\t')
        
        # 迭代讀取每一行並打印
        for row in reader:
            print(row)

# 指定你的 TSV 檔案路徑
file_path = 'HK-DOJ.tsv'

# 調用函數來讀取檔案
read_tsv(file_path)
