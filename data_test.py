import os
import random
import soundfile as sf
import re
import pandas as pd
import time  # 用於記錄延遲時間
import librosa
import noisereduce as nr
from google.cloud import speech
from google.oauth2 import service_account
from google.cloud import translate_v2 as translate
from jiwer import wer, mer, wil, wip, cer
from tqdm import tqdm  # 用於顯示進度條
from pydub import AudioSegment
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from audio_separator.separator import Separator
from pydub.utils import which
import boto3
import wave
import requests
import json
# 顯式指定 ffmpeg 的路徑
 # 或者直接給出完整路徑
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import azure.cognitiveservices.speech as speechsdk
def convert_flac_to_wav(input_flac_path, output_wav_path):
    # 加載 FLAC 文件
    audio = AudioSegment.from_file(input_flac_path, format="flac")
    
    # 導出為 WAV 格式
    audio.export(output_wav_path, format="wav")
# 參數設置
RATE = 16000
CHUNK = int(RATE / 10)



AudioSegment.ffmpeg = which("ffmpeg") 
# 手動指定 Google Cloud 憑證文件
def google(data_folder):
    credentials = service_account.Credentials.from_service_account_file("speech-to-text-api-438608-5accaa91d5f2.json")

    # 初始化客戶端
    client = speech.SpeechClient(credentials=credentials)
    translate_client = translate.Client(credentials=credentials)

    # 配置語音識別參數
    RATE = 16000
    CHUNK = int(RATE / 10)
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-us",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=False
    )

    # 加載對照文本
    def load_transcriptions(folder_path):
        transcriptions = []
        # 遍歷文件夾尋找以 '.trans.txt' 結尾的文件
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.trans.txt'):
                    with open(os.path.join(root, file), 'r') as f:
                        for line in f:
                            parts = line.strip().split(maxsplit=1)
                            if len(parts) == 2:
                                audio_file = os.path.join(root, parts[0] + ".flac")  # 對應的音頻文件
                                transcript = parts[1]  # 對照文本
                                transcriptions.append((audio_file, transcript))
        return transcriptions

    # 音頻生成器
    def audio_file_generator(file_path, chunk_size=CHUNK):
        with sf.SoundFile(file_path, mode='r') as audio_file:
            while True:
                data = audio_file.read(chunk_size, dtype='int16')
                if len(data) == 0:
                    break
                yield data

    # 處理單個音頻文件
    def process_audio_file(file_path, transcript):
        print(f"處理音頻文件: {file_path}")

        audio_generator = audio_file_generator(file_path)
        requests = (speech.StreamingRecognizeRequest(audio_content=chunk.tobytes()) for chunk in audio_generator)

        # 記錄單個文件的開始時間
        start_time = time.time()

        recognized_text = ""
        try:
            responses = client.streaming_recognize(streaming_config, requests)
            for response in responses:
                for result in response.results:
                    if result.alternatives:
                        recognized_text = result.alternatives[0].transcript
                        break  # 獲取第一個識別結果

            # 翻譯
            translation = translate_client.translate(recognized_text, target_language='en')
            translation_T = translate_client.translate(recognized_text, target_language='zh-TW')
            print(f"對照文本：{transcript}")
            print(f"識別結果：{recognized_text}")
            print(f"翻譯（英文）：{translation['translatedText']}")
            print(f"翻譯（繁體中文）：{translation_T['translatedText']}")

            # 計算評估指標
            error_rate_wer = wer(transcript.lower(), recognized_text.lower())
            error_rate_mer = mer(transcript.lower(), recognized_text.lower())
            error_rate_wil = wil(transcript.lower(), recognized_text.lower())
            error_rate_wip = wip(transcript.lower(), recognized_text.lower())
            error_rate_cer = cer(transcript.lower(), recognized_text.lower())

            print(f"Word Error Rate (WER): {error_rate_wer:.2%}")
            print(f"Match Error Rate (MER): {error_rate_mer:.2%}")
            print(f"Word Information Lost (WIL): {error_rate_wil:.2%}")
            print(f"Word Information Preserved (WIP): {error_rate_wip:.2%}")
            print(f"Character Error Rate (CER): {error_rate_cer:.2%}")

        except Exception as e:
            print(f"處理文件 {file_path} 時出錯: {e}")
            error_rate_wer = None
            error_rate_mer = None
            error_rate_wil = None
            error_rate_wip = None
            error_rate_cer = None

        # 記錄處理時間
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"處理延遲時間: {elapsed_time:.2f} 秒")

        return error_rate_wer, error_rate_mer, error_rate_wil, error_rate_wip, error_rate_cer, elapsed_time

    # 遍歷文件夾並處理所有音頻文件
    def process_audio_folder(folder_path, sample_size=50):
        transcriptions = load_transcriptions(folder_path)

        # 隨機抽樣50個音頻文件
        sampled_transcriptions = random.sample(transcriptions, sample_size)

        total_wer = 0
        total_mer = 0
        total_wil = 0
        total_wip = 0
        total_cer = 0
        total_files = 0
        total_time = 0

        # 使用 tqdm 顯示進度條
        for audio_file, transcript in tqdm(sampled_transcriptions, desc="處理音頻文件", unit="個文件"):
            error_rate_wer, error_rate_mer, error_rate_wil, error_rate_wip, error_rate_cer, elapsed_time = process_audio_file(audio_file, transcript)

            # 累計各項錯誤率和延遲時間
            if error_rate_wer is not None:
                total_wer += error_rate_wer
                total_mer += error_rate_mer
                total_wil += error_rate_wil
                total_wip += error_rate_wip
                total_cer += error_rate_cer
                total_files += 1
            total_time += elapsed_time

        # 計算各項平均值
        wer_time = (total_time / total_files) if total_files > 0 else None
        average_wer = (total_wer / total_files) if total_files > 0 else None
        average_mer = (total_mer / total_files) if total_files > 0 else None
        average_wil = (total_wil / total_files) if total_files > 0 else None
        average_wip = (total_wip / total_files) if total_files > 0 else None
        average_cer = (total_cer / total_files) if total_files > 0 else None

        return average_wer, average_mer, average_wil, average_wip, average_cer, wer_time

    # 執行處理並計算結果
    average_wer, average_mer, average_wil, average_wip, average_cer, wer_time = process_audio_folder(data_folder, sample_size=50)

    # 輸出整體結果
    if average_wer is not None:
        print(f"整體平均 Word Error Rate (WER): {average_wer:.2%}")
        print(f"整體平均 Match Error Rate (MER): {average_mer:.2%}")
        print(f"整體平均 Word Information Lost (WIL): {average_wil:.2%}")
        print(f"整體平均 Word Information Preserved (WIP): {average_wip:.2%}")
        print(f"整體平均 Character Error Rate (CER): {average_cer:.2%}")
    else:
        print("無有效文件計算平均值。")
    print(f"平均處理時間: {wer_time:.2f} 秒")

def azure(data_folder):

    # 參數設置
    RATE = 16000
    CHUNK = int(RATE / 10)

    # Azure 語音服務憑證設置
    AZURE_KEY = '2HLw0CKOToNdjnNWQCjj0BhLLMw176tRPVjt6GH48IP0HDdK99rsJQQJ99AKACxCCsyXJ3w3AAAYACOGYlod'  # 替換為你自己的 Azure API Key
    AZURE_REGION = 'japanwest'  # 替換為你自己的 Azure 區域

    # 初始化 Azure 客戶端
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
    speech_config.speech_recognition_language = "en-US"

    # 加載對照文本
    def load_transcriptions(folder_path):
        transcriptions = []
        # 遍歷文件夾尋找以 '.trans.txt' 結尾的文件
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.trans.txt'):
                    with open(os.path.join(root, file), 'r') as f:
                        for line in f:
                            parts = line.strip().split(maxsplit=1)
                            if len(parts) == 2:
                                audio_file = os.path.join(root, parts[0] + ".flac")  # 對應的音頻文件
                                transcript = parts[1]  # 對照文本
                                transcriptions.append((audio_file, transcript))
        return transcriptions

    # 處理單個音頻文件
    def process_audio_file(file_path, transcript):
        print(f"處理音頻文件: {file_path}")

        # 如果文件是 FLAC 格式，先轉換為 WAV 格式
        if file_path.endswith(".flac"):
            wav_file_path = file_path.replace(".flac", ".wav")
            convert_flac_to_wav(file_path, wav_file_path)
            audio_file_path = wav_file_path
        else:
            audio_file_path = file_path  # 非 FLAC 格式直接使用原文件

        # 配置語音識別器
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        # 語音識別流程
        try:
            start_time = time.time()
            result = recognizer.recognize_once()  # 使用 recognize_once() 進行單次識別
            recognized_text = result.text.strip()  # 去除多餘空格

            # 計算各種錯誤率
            error_rate_wer = wer(transcript.lower(), recognized_text.lower())
            error_rate_mer = mer(transcript.lower(), recognized_text.lower())
            error_rate_wil = wil(transcript.lower(), recognized_text.lower())
            error_rate_wip = wip(transcript.lower(), recognized_text.lower())
            error_rate_cer = cer(transcript.lower(), recognized_text.lower())

            # 輸出結果
            print(f"對照文本：{transcript}")
            print(f"識別結果：{recognized_text}")
            print(f"Word Error Rate (WER): {error_rate_wer:.2%}")
            print(f"Match Error Rate (MER): {error_rate_mer:.2%}")
            print(f"Word Information Lost (WIL): {error_rate_wil:.2%}")
            print(f"Word Information Preserved (WIP): {error_rate_wip:.2%}")
            print(f"Character Error Rate (CER): {error_rate_cer:.2%}")

            # 記錄處理時間
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"處理延遲時間: {elapsed_time:.2f} 秒")

            return {
                "wer": error_rate_wer,
                "mer": error_rate_mer,
                "wil": error_rate_wil,
                "wip": error_rate_wip,
                "cer": error_rate_cer,
                "elapsed_time": elapsed_time,
            }

        except Exception as e:
            print(f"處理文件 {file_path} 時出錯: {e}")
            return {
                "wer": None,
                "mer": None,
                "wil": None,
                "wip": None,
                "cer": None,
                "elapsed_time": None,
            }

    # 遍歷文件夾並處理所有音頻文件
    def process_audio_folder(folder_path, sample_size=50):
        transcriptions = load_transcriptions(folder_path)

        # 隨機抽樣50個音頻文件
        sampled_transcriptions = random.sample(transcriptions, sample_size)

        total_time = 0
        total_files = 0

        total_wer = 0
        total_mer = 0
        total_wil = 0
        total_wip = 0
        total_cer = 0

        # 使用 tqdm 顯示進度條
        for audio_file, transcript in tqdm(sampled_transcriptions, desc="處理音頻文件", unit="個文件"):
            result = process_audio_file(audio_file, transcript)

            if result["elapsed_time"] is not None:
                total_time += result["elapsed_time"]
                total_files += 1

                total_wer += result["wer"]
                total_mer += result["mer"]
                total_wil += result["wil"]
                total_wip += result["wip"]
                total_cer += result["cer"]

        # 計算平均處理時間
        average_time = (total_time / total_files) if total_files > 0 else None

        # 計算五個錯誤率指標的平均值
        average_wer = (total_wer / total_files) if total_files > 0 else None
        average_mer = (total_mer / total_files) if total_files > 0 else None
        average_wil = (total_wil / total_files) if total_files > 0 else None
        average_wip = (total_wip / total_files) if total_files > 0 else None
        average_cer = (total_cer / total_files) if total_files > 0 else None

        return {
            "average_time": average_time,
            "average_wer": average_wer,
            "average_mer": average_mer,
            "average_wil": average_wil,
            "average_wip": average_wip,
            "average_cer": average_cer,
        }

    # 執行處理並計算結果
    result = process_audio_folder(data_folder, sample_size=50)

    # 輸出整體結果
    if result["average_time"] is not None:
        print(f"整體平均處理時間: {result['average_time']:.2f} 秒")
    else:
        print("無有效文件計算平均處理時間。")

    if result["average_wer"] is not None:
        print(f"整體平均 Word Error Rate (WER): {result['average_wer']:.2%}")
    else:
        print("無有效文件計算平均 WER。")

    if result["average_mer"] is not None:
        print(f"整體平均 Match Error Rate (MER): {result['average_mer']:.2%}")
    else:
        print("無有效文件計算平均 MER。")

    if result["average_wil"] is not None:
        print(f"整體平均 Word Information Lost (WIL): {result['average_wil']:.2%}")
    else:
        print("無有效文件計算平均 WIL。")

    if result["average_wip"] is not None:
        print(f"整體平均 Word Information Preserved (WIP): {result['average_wip']:.2%}")
    else:
        print("無有效文件計算平均 WIP。")

    if result["average_cer"] is not None:
        print(f"整體平均 Character Error Rate (CER): {result['average_cer']:.2%}")
    else:
        print("無有效文件計算平均 CER。")



def clean_audio(file_path, output_path):
    # 加載音頻文件
    st_time = time.time()
    audio, sr = librosa.load(file_path, sr=None)
    # 使用音頻的前0.5秒作為噪音樣本（假設前0.5秒無語音）
    noise_sample = audio[:int(0.5 * sr)]
    # 降噪處理
    cleaned_audio = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample)
    # 保存清潔後的音頻
    sf.write(output_path, cleaned_audio, sr)
    # print(f"清潔後的音頻已保存到: {output_path}")
   
    ed_time = time.time()
    separation_time = ed_time - st_time
    print(f"語音清潔（分離）所需時間：{separation_time:.2f}秒")
    return separation_time

def amazon_transcribe(data_folder):
    # AWS Transcribe 配置信息
    AWS_REGION = 'us-west-2'  # 替換為你的 AWS 區域
    S3_BUCKET_NAME = 'moln9110'  # 替換為你的 S3 桶名稱
    transcribe = boto3.client('transcribe', region_name=AWS_REGION)

    # 加載對照文本
    def load_transcriptions(folder_path):
        transcriptions = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.trans.txt'):
                    with open(os.path.join(root, file), 'r') as f:
                        for line in f:
                            parts = line.strip().split(maxsplit=1)
                            if len(parts) == 2:
                                audio_file = os.path.join(root, parts[0] + ".flac")
                                transcript = parts[1]
                                transcriptions.append((audio_file, transcript))
        return transcriptions

    # 如果文件是 FLAC 格式，先轉換為 WAV 格式
    def convert_flac_to_wav(input_flac_path, output_wav_path):
        audio = AudioSegment.from_file(input_flac_path, format="flac")
        audio.export(output_wav_path, format="wav")

    # 處理單個音頻文件
    def process_audio_file(file_path, transcript):
        print(f"處理音頻文件: {file_path}")

        # 如果文件是 FLAC 格式，先轉換為 WAV 格式
        if file_path.endswith(".flac"):
            wav_file_path = file_path.replace(".flac", ".wav")
            convert_flac_to_wav(file_path, wav_file_path)
            audio_file_path = wav_file_path
        else:
            audio_file_path = file_path

        # 上傳音頻文件到 S3
        s3_client = boto3.client('s3')
        s3_key = os.path.basename(audio_file_path)
        s3_client.upload_file(audio_file_path, S3_BUCKET_NAME, s3_key)
        s3_url = f"s3://{S3_BUCKET_NAME}/{s3_key}"

        # 使用 Amazon Transcribe 進行語音識別
        try:
            start_time = time.time()
            
            # 開始 Transcribe 任務
            job_name = f"transcribe-job-{int(time.time())}"
            transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                LanguageCode='en-US',
                Media={'MediaFileUri': s3_url},
                MediaFormat='wav',
                OutputBucketName=S3_BUCKET_NAME
            )

            # 等待任務完成
            while True:
                status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
                job_status = status['TranscriptionJob']['TranscriptionJobStatus']
                if job_status in ['COMPLETED', 'FAILED']:
                    break
                time.sleep(5)

            # 獲取結果
            if job_status == 'COMPLETED':
                result_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
                try:
                    # 請求轉錄結果並打印原始響應
                    response = requests.get(result_url)
                    if response.status_code == 200:
                        result_json = response.json()
                        recognized_text = result_json['results']['transcripts'][0]['transcript']
                        print(f"識別文本: {recognized_text}")

                        # 計算 WER
                        error_rate = wer(transcript.lower(), recognized_text.lower())
                        print(f"Word Error Rate (WER): {error_rate:.2%}")
                    else:
                        print(f"請求失敗，狀態碼: {response.status_code}，響應內容: {response.text}")
                        recognized_text = ""
                        error_rate = None
                except Exception as e:
                    print(f"獲取轉錄結果時出錯: {e}")
                    recognized_text = ""
                    error_rate = None
            else:
                print(f"Transcription job failed for {file_path}")
                recognized_text = ""
                error_rate = None

            # 記錄處理時間
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"處理延遲時間: {elapsed_time:.2f} 秒")
            
        except Exception as e:
            print(f"處理文件 {file_path} 時出錯: {e}")
            recognized_text = ""
            error_rate = None
            elapsed_time = 0

        return error_rate, elapsed_time

    # 遍歷文件夾並處理所有音頻文件
    def process_audio_folder(folder_path, sample_size=50):
        transcriptions = load_transcriptions(folder_path)

        # 隨機抽樣50個音頻文件
        sampled_transcriptions = random.sample(transcriptions, min(sample_size, len(transcriptions)))

        total_wer = 0
        total_files = 0
        total_time = 0

        # 使用 tqdm 顯示進度條
        for audio_file, transcript in tqdm(sampled_transcriptions, desc="處理音頻文件", unit="個文件"):
            error_rate, elapsed_time = process_audio_file(audio_file, transcript)

            # 累計 WER 和延遲時間
            if error_rate is not None:
                total_wer += error_rate
                total_files += 1
            total_time += elapsed_time

        # 計算平均 WER
        wer_time = (total_time / total_files) if total_files > 0 else None
        average_wer = (total_wer / total_files) if total_files > 0 else None
        return average_wer, wer_time

    # 執行處理並計算結果
    average_wer, wer_time = process_audio_folder(data_folder, sample_size=50)

    # 輸出整體結果
    if average_wer is not None:
        print(f"整體平均 Word Error Rate (WER): {average_wer:.2%}")
    else:
        print("無有效文件計算平均 WER。")
    print(f"總處理時間: {wer_time:.2f} 秒")

def save_results_to_excel(results, output_file="results.xlsx"):
    rows = []
    for result in results:
        if "error" in result:
            rows.append({
                "file_path": result["file_path"],
                "transcript": None,
                "recognized_text": None,
                "separation_time": None,
                "elapsed_time": None,
                "WER": None,
                "MER": None,
                "WIL": None,
                "WIP": None,
                "CER": None,
                "error": result["error"]
            })
        else:
            rows.append({
                "file_path": result["file_path"],
                "transcript": result["transcript"],
                "recognized_text": result["recognized_text"],
                "separation_time": result["separation_time"],
                "elapsed_time": result["elapsed_time"],
                "WER": result["error_rates"]["WER"],
                "MER": result["error_rates"]["MER"],
                "WIL": result["error_rates"]["WIL"],
                "WIP": result["error_rates"]["WIP"],
                "CER": result["error_rates"]["CER"],
                "error": None
            })

    df = pd.DataFrame(rows)
    df.to_excel(output_file, index=False)
    print(f"結果已保存到 {output_file}")
def remove_symbols(text):
    # 去除符號，只保留字母和數字
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)
def whisper_tiny(data_folder):
    # 初始化模型和處理器
    results = []  # 用來存儲處理結果的陣列
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    
    # 處理單個音頻文件
    def process_audio_file(file_path, transcript):
        separation_times = 0
        print(f"處理音頻文件: {file_path}")
        
        # 記錄開始時間
        start_time = time.time()

        try:
            # 加載音頻文件
            audio, rate = sf.read(file_path)
            if rate != 16000:
                raise ValueError("Whisper 模型需要音頻採樣率為 16kHz。")
            
            
          
            
            separation_time =clean_audio(file_path, "test.wav")
            # 輸出清潔所需的時間
           

            # 準備輸入特徵
            input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

            # 推理
            predicted_ids = model.generate(input_features)
            recognized_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # 輸出識別結果和對照文本
            print(f"對照文本：{transcript}")
            print(f"識別結果：{recognized_text}")

            # 去除符號
            transcript_clean = remove_symbols(transcript.lower())
            recognized_text_clean = remove_symbols(recognized_text.lower())

            # 計算各種錯誤率
            error_rate_wer = wer(transcript_clean, recognized_text_clean)
            error_rate_mer = mer(transcript_clean, recognized_text_clean)
            error_rate_wil = wil(transcript_clean, recognized_text_clean)
            error_rate_wip = wip(transcript_clean, recognized_text_clean)
            error_rate_cer = cer(transcript_clean, recognized_text_clean)

            # 輸出各種錯誤率
            print(f"Word Error Rate (WER): {error_rate_wer:.2%}")
            print(f"Match Error Rate (MER): {error_rate_mer:.2%}")
            print(f"Word Information Lost (WIL): {error_rate_wil:.2%}")
            print(f"Word Information Preserved (WIP): {error_rate_wip:.2%}")
            print(f"Character Error Rate (CER): {error_rate_cer:.2%}")

        except Exception as e:
            print(f"處理文件 {file_path} 時出錯: {e}")
            error_rate_wer = error_rate_mer = error_rate_wil = error_rate_wip = error_rate_cer = None

        # 記錄處理時間
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"處理延遲時間: {elapsed_time:.2f} 秒")
        results.append({
        "file_path": file_path,
        "transcript": transcript,
        "recognized_text": recognized_text,
        "separation_time": separation_time,
        "elapsed_time": elapsed_time,
        "error_rates": {
            "WER": error_rate_wer,
            "MER": error_rate_mer,
            "WIL": error_rate_wil,
            "WIP": error_rate_wip,
            "CER": error_rate_cer,
        }
    })
        return {
            "wer": error_rate_wer,
            "mer": error_rate_mer,
            "wil": error_rate_wil,
            "wip": error_rate_wip,
            "cer": error_rate_cer,
            "elapsed_time": elapsed_time,"separation_time":separation_time,
        }

    # 遍歷文件夾並處理所有音頻文件
    def process_audio_folder(folder_path, sample_size=50):
        # 加載對照文本
        def load_transcriptions(folder_path):
            transcriptions = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.trans.txt'):
                        with open(os.path.join(root, file), 'r') as f:
                            for line in f:
                                parts = line.strip().split(maxsplit=1)
                                if len(parts) == 2:
                                    audio_file = os.path.join(root, parts[0] + ".flac")  # 對應的音頻文件
                                    transcript = parts[1]  # 對照文本
                                    transcriptions.append((audio_file, transcript))
            return transcriptions

        transcriptions = load_transcriptions(folder_path)

        # 隨機抽樣音頻文件
        sampled_transcriptions = random.sample(transcriptions, sample_size)

        total_wer = total_mer = total_wil = total_wip = total_cer = 0
        total_files = 0
        total_time = 0
        total_separatio =0
        # 使用 tqdm 顯示進度條
        for audio_file, transcript in tqdm(sampled_transcriptions, desc="處理音頻文件", unit="個文件"):
            result = process_audio_file(audio_file, transcript)

            if result["wer"] is not None:
                total_wer += result["wer"]
                total_mer += result["mer"]
                total_wil += result["wil"]
                total_wip += result["wip"]
                total_cer += result["cer"]
                total_files += 1
            total_time += result["elapsed_time"]
            total_separatio +=result["separation_time"]

        # 計算平均值
        avg_wer = (total_wer / total_files) if total_files > 0 else None
        avg_mer = (total_mer / total_files) if total_files > 0 else None
        avg_wil = (total_wil / total_files) if total_files > 0 else None
        avg_wip = (total_wip / total_files) if total_files > 0 else None
        avg_cer = (total_cer / total_files) if total_files > 0 else None
        avg_time = (total_time / total_files) if total_files > 0 else None
        avg_st  = (total_separatio / total_files) if total_files > 0 else None


        return avg_wer, avg_mer, avg_wil, avg_wip, avg_cer, avg_time,avg_st

    # 執行處理並輸出整體結果
    avg_wer, avg_mer, avg_wil, avg_wip, avg_cer, avg_time,avg_st = process_audio_folder(data_folder, sample_size=50)

    if avg_wer is not None:
        print(f"整體平均 Word Error Rate (WER): {avg_wer:.2%}")
        print(f"整體平均 Match Error Rate (MER): {avg_mer:.2%}")
        print(f"整體平均 Word Information Lost (WIL): {avg_wil:.2%}")
        print(f"整體平均 Word Information Preserved (WIP): {avg_wip:.2%}")
        print(f"整體平均 Character Error Rate (CER): {avg_cer:.2%}")
        
    else:
        print("無有效文件計算平均 WER。")
    print(f"平均處理時間: {avg_time:.2f} 秒")
    save_results_to_excel(results)
    print(f"平均清潔時間: {avg_st:.2f} 秒")
# 調用示例

# amazon_transcribe(data_folder)



data_folder = "test-other/LibriSpeech/test-other"  
data_folder2 = "dev-clean/LibriSpeech/dev-clean" 
whisper_tiny(data_folder)
# google(data_folder)
# amazon_transcribe(data_folder)

# whisper_tiny(data_folder2)
print("_______")
# whisper_tiny(data_folder)