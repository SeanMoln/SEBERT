
# SEBERT

SEBERT 是一個用於語音識別、翻譯和文本相似度計算的工具集。此資料夾包含多個 Python 腳本，用於不同的任務。

## 目錄

- [安裝](#安裝)
- [使用說明](#使用說明)
  - [data_test.py](#data_testpy)
  - [SBERT1.py](#sbert1py)
  - [SBERT2.py](#sbert2py)
  - [speaker-dataset.py](#speaker-datasetpy)


## 安裝

請確保已安裝以下依賴項：

- Python 3.7+
- pandas
- torchaudio
- transformers
- sentence-transformers
- google-cloud-speech
- google-cloud-translate
- ibm-watson
- boto3
- librosa
- noisereduce
- tqdm
- pydub
- jiwer

可以使用以下命令安裝依賴項：


```
pip install -r requirements.txt
```
## 使用說明

### data_test.py

此腳本用於語音識別和翻譯，並計算語音識別的錯誤率。

#### 使用方法

```bash
python data_test.py
```

#### 輸入

- 音頻文件夾路徑

#### 輸出

- 語音識別結果
- 翻譯結果
- 錯誤率（WER、MER、WIL、WIP、CER）

### SBERT1.py

此腳本將文本文件轉換為 Huggingface Dataset，並計算文本對之間的餘弦相似度。

#### 使用方法

```bash
python SBERT1.py
```

#### 輸入

- 兩個文本文件路徑
- 輸出文件路徑

#### 輸出

- 包含文本對及其餘弦相似度的 CSV 文件

### SBERT2.py

此腳本使用滑動視窗技術計算兩個長文本之間的最大和平均餘弦相似度。

#### 使用方法

```bash
python SBERT2.py
```

#### 輸入

- 兩個 CSV 文件路徑

#### 輸出

- 最大餘弦相似度
- 最佳匹配視窗
- 平均餘弦相似度

### speaker-dataset.py

此腳本加載和處理 LibriSpeech 數據集，並將其轉換為 PyTorch Dataset。

#### 使用方法

```bash
python speaker-dataset.py
```

#### 輸入

- LibriSpeech 數據集文件夾路徑

#### 輸出

- PyTorch Dataset，包含音頻波形、採樣率和轉錄文本

