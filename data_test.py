import os
import random
import soundfile as sf
import re
import pandas as pd
import time  # 用于记录延迟时间
import librosa
import noisereduce as nr
from google.cloud import speech
from google.oauth2 import service_account
from google.cloud import translate_v2 as translate
from jiwer import wer, mer, wil, wip, cer
from tqdm import tqdm  # 用于显示进度条
from pydub import AudioSegment
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from audio_separator.separator import Separator
from pydub.utils import which
import boto3
import wave
import requests
import json
# 显式指定 ffmpeg 的路径
 # 或者直接给出完整路径
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import azure.cognitiveservices.speech as speechsdk
def convert_flac_to_wav(input_flac_path, output_wav_path):
    # 加载 FLAC 文件
    audio = AudioSegment.from_file(input_flac_path, format="flac")
    
    # 导出为 WAV 格式
    audio.export(output_wav_path, format="wav")
# 参数设置
RATE = 16000
CHUNK = int(RATE / 10)



AudioSegment.ffmpeg = which("ffmpeg") 
# 手动指定 Google Cloud 凭证文件
def google(data_folder):
    credentials = service_account.Credentials.from_service_account_file("speech-to-text-api-438608-5accaa91d5f2.json")

    # 初始化客户端
    client = speech.SpeechClient(credentials=credentials)
    translate_client = translate.Client(credentials=credentials)

    # 配置语音识别参数
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

    # 加载对照文本
    def load_transcriptions(folder_path):
        transcriptions = []
        # 遍历文件夹寻找以 '.trans.txt' 结尾的文件
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.trans.txt'):
                    with open(os.path.join(root, file), 'r') as f:
                        for line in f:
                            parts = line.strip().split(maxsplit=1)
                            if len(parts) == 2:
                                audio_file = os.path.join(root, parts[0] + ".flac")  # 对应的音频文件
                                transcript = parts[1]  # 对照文本
                                transcriptions.append((audio_file, transcript))
        return transcriptions

    # 音频生成器
    def audio_file_generator(file_path, chunk_size=CHUNK):
        with sf.SoundFile(file_path, mode='r') as audio_file:
            while True:
                data = audio_file.read(chunk_size, dtype='int16')
                if len(data) == 0:
                    break
                yield data

    # 处理单个音频文件
    def process_audio_file(file_path, transcript):
        print(f"处理音频文件: {file_path}")

        audio_generator = audio_file_generator(file_path)
        requests = (speech.StreamingRecognizeRequest(audio_content=chunk.tobytes()) for chunk in audio_generator)

        # 记录单个文件的开始时间
        start_time = time.time()

        recognized_text = ""
        try:
            responses = client.streaming_recognize(streaming_config, requests)
            for response in responses:
                for result in response.results:
                    if result.alternatives:
                        recognized_text = result.alternatives[0].transcript
                        break  # 获取第一个识别结果

            # 翻译
            translation = translate_client.translate(recognized_text, target_language='en')
            translation_T = translate_client.translate(recognized_text, target_language='zh-TW')
            print(f"對照文本：{transcript}")
            print(f"识别结果：{recognized_text}")
            print(f"翻译（英文）：{translation['translatedText']}")
            print(f"翻译（繁体中文）：{translation_T['translatedText']}")

            # 计算评估指标
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
            print(f"处理文件 {file_path} 时出错: {e}")
            error_rate_wer = None
            error_rate_mer = None
            error_rate_wil = None
            error_rate_wip = None
            error_rate_cer = None

        # 记录处理时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"处理延迟时间: {elapsed_time:.2f} 秒")

        return error_rate_wer, error_rate_mer, error_rate_wil, error_rate_wip, error_rate_cer, elapsed_time

    # 遍历文件夹并处理所有音频文件
    def process_audio_folder(folder_path, sample_size=50):
        transcriptions = load_transcriptions(folder_path)

        # 随机抽样50个音频文件
        sampled_transcriptions = random.sample(transcriptions, sample_size)

        total_wer = 0
        total_mer = 0
        total_wil = 0
        total_wip = 0
        total_cer = 0
        total_files = 0
        total_time = 0

        # 使用 tqdm 显示进度条
        for audio_file, transcript in tqdm(sampled_transcriptions, desc="处理音频文件", unit="个文件"):
            error_rate_wer, error_rate_mer, error_rate_wil, error_rate_wip, error_rate_cer, elapsed_time = process_audio_file(audio_file, transcript)

            # 累计各项错误率和延迟时间
            if error_rate_wer is not None:
                total_wer += error_rate_wer
                total_mer += error_rate_mer
                total_wil += error_rate_wil
                total_wip += error_rate_wip
                total_cer += error_rate_cer
                total_files += 1
            total_time += elapsed_time

        # 计算各项平均值
        wer_time = (total_time / total_files) if total_files > 0 else None
        average_wer = (total_wer / total_files) if total_files > 0 else None
        average_mer = (total_mer / total_files) if total_files > 0 else None
        average_wil = (total_wil / total_files) if total_files > 0 else None
        average_wip = (total_wip / total_files) if total_files > 0 else None
        average_cer = (total_cer / total_files) if total_files > 0 else None

        return average_wer, average_mer, average_wil, average_wip, average_cer, wer_time

    # 执行处理并计算结果
    average_wer, average_mer, average_wil, average_wip, average_cer, wer_time = process_audio_folder(data_folder, sample_size=50)

    # 输出整体结果
    if average_wer is not None:
        print(f"整体平均 Word Error Rate (WER): {average_wer:.2%}")
        print(f"整体平均 Match Error Rate (MER): {average_mer:.2%}")
        print(f"整体平均 Word Information Lost (WIL): {average_wil:.2%}")
        print(f"整体平均 Word Information Preserved (WIP): {average_wip:.2%}")
        print(f"整体平均 Character Error Rate (CER): {average_cer:.2%}")
    else:
        print("无有效文件计算平均值。")
    print(f"平均处理时间: {wer_time:.2f} 秒")

def azure(data_folder):

    # 参数设置
    RATE = 16000
    CHUNK = int(RATE / 10)

    # Azure 语音服务凭证设置
    AZURE_KEY = '2HLw0CKOToNdjnNWQCjj0BhLLMw176tRPVjt6GH48IP0HDdK99rsJQQJ99AKACxCCsyXJ3w3AAAYACOGYlod'  # 替换为你自己的 Azure API Key
    AZURE_REGION = 'japanwest'  # 替换为你自己的 Azure 区域

    # 初始化 Azure 客户端
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
    speech_config.speech_recognition_language = "en-US"

    # 加载对照文本
    def load_transcriptions(folder_path):
        transcriptions = []
        # 遍历文件夹寻找以 '.trans.txt' 结尾的文件
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.trans.txt'):
                    with open(os.path.join(root, file), 'r') as f:
                        for line in f:
                            parts = line.strip().split(maxsplit=1)
                            if len(parts) == 2:
                                audio_file = os.path.join(root, parts[0] + ".flac")  # 对应的音频文件
                                transcript = parts[1]  # 对照文本
                                transcriptions.append((audio_file, transcript))
        return transcriptions

    # 处理单个音频文件
    def process_audio_file(file_path, transcript):
        print(f"处理音频文件: {file_path}")

        # 如果文件是 FLAC 格式，先转换为 WAV 格式
        if file_path.endswith(".flac"):
            wav_file_path = file_path.replace(".flac", ".wav")
            convert_flac_to_wav(file_path, wav_file_path)
            audio_file_path = wav_file_path
        else:
            audio_file_path = file_path  # 非 FLAC 格式直接使用原文件

        # 配置语音识别器
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        # 语音识别流程
        try:
            start_time = time.time()
            result = recognizer.recognize_once()  # 使用 recognize_once() 进行单次识别
            recognized_text = result.text.strip()  # 去除多余空格

            # 计算各种错误率
            error_rate_wer = wer(transcript.lower(), recognized_text.lower())
            error_rate_mer = mer(transcript.lower(), recognized_text.lower())
            error_rate_wil = wil(transcript.lower(), recognized_text.lower())
            error_rate_wip = wip(transcript.lower(), recognized_text.lower())
            error_rate_cer = cer(transcript.lower(), recognized_text.lower())

            # 输出结果
            print(f"对照文本：{transcript}")
            print(f"识别结果：{recognized_text}")
            print(f"Word Error Rate (WER): {error_rate_wer:.2%}")
            print(f"Match Error Rate (MER): {error_rate_mer:.2%}")
            print(f"Word Information Lost (WIL): {error_rate_wil:.2%}")
            print(f"Word Information Preserved (WIP): {error_rate_wip:.2%}")
            print(f"Character Error Rate (CER): {error_rate_cer:.2%}")

            # 记录处理时间
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"处理延迟时间: {elapsed_time:.2f} 秒")

            return {
                "wer": error_rate_wer,
                "mer": error_rate_mer,
                "wil": error_rate_wil,
                "wip": error_rate_wip,
                "cer": error_rate_cer,
                "elapsed_time": elapsed_time,
            }

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return {
                "wer": None,
                "mer": None,
                "wil": None,
                "wip": None,
                "cer": None,
                "elapsed_time": None,
            }

    # 遍历文件夹并处理所有音频文件
    def process_audio_folder(folder_path, sample_size=50):
        transcriptions = load_transcriptions(folder_path)

        # 随机抽样50个音频文件
        sampled_transcriptions = random.sample(transcriptions, sample_size)

        total_time = 0
        total_files = 0

        total_wer = 0
        total_mer = 0
        total_wil = 0
        total_wip = 0
        total_cer = 0

        # 使用 tqdm 显示进度条
        for audio_file, transcript in tqdm(sampled_transcriptions, desc="处理音频文件", unit="个文件"):
            result = process_audio_file(audio_file, transcript)

            if result["elapsed_time"] is not None:
                total_time += result["elapsed_time"]
                total_files += 1

                total_wer += result["wer"]
                total_mer += result["mer"]
                total_wil += result["wil"]
                total_wip += result["wip"]
                total_cer += result["cer"]

        # 计算平均处理时间
        average_time = (total_time / total_files) if total_files > 0 else None

        # 计算五个错误率指标的平均值
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

    # 执行处理并计算结果
    result = process_audio_folder(data_folder, sample_size=50)

    # 输出整体结果
    if result["average_time"] is not None:
        print(f"整体平均处理时间: {result['average_time']:.2f} 秒")
    else:
        print("无有效文件计算平均处理时间。")

    if result["average_wer"] is not None:
        print(f"整体平均 Word Error Rate (WER): {result['average_wer']:.2%}")
    else:
        print("无有效文件计算平均 WER。")

    if result["average_mer"] is not None:
        print(f"整体平均 Match Error Rate (MER): {result['average_mer']:.2%}")
    else:
        print("无有效文件计算平均 MER。")

    if result["average_wil"] is not None:
        print(f"整体平均 Word Information Lost (WIL): {result['average_wil']:.2%}")
    else:
        print("无有效文件计算平均 WIL。")

    if result["average_wip"] is not None:
        print(f"整体平均 Word Information Preserved (WIP): {result['average_wip']:.2%}")
    else:
        print("无有效文件计算平均 WIP。")

    if result["average_cer"] is not None:
        print(f"整体平均 Character Error Rate (CER): {result['average_cer']:.2%}")
    else:
        print("无有效文件计算平均 CER。")



def clean_audio(file_path, output_path):
    # 加载音频文件
    st_time = time.time()
    audio, sr = librosa.load(file_path, sr=None)
    # 使用音频的前0.5秒作为噪音样本（假设前0.5秒无语音）
    noise_sample = audio[:int(0.5 * sr)]
    # 降噪处理
    cleaned_audio = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample)
    # 保存清洁后的音频
    sf.write(output_path, cleaned_audio, sr)
    # print(f"清洁后的音频已保存到: {output_path}")
   
    ed_time = time.time()
    separation_time = ed_time - st_time
    print(f"语音清洁（分离）所需时间：{separation_time:.2f}秒")
    return separation_time

def amazon_transcribe(data_folder):
    # AWS Transcribe 配置信息
    AWS_REGION = 'us-west-2'  # 替换为你的 AWS 区域
    S3_BUCKET_NAME = 'moln9110'  # 替换为你的 S3 桶名称
    transcribe = boto3.client('transcribe', region_name=AWS_REGION)

    # 加载对照文本
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

    # 如果文件是 FLAC 格式，先转换为 WAV 格式
    def convert_flac_to_wav(input_flac_path, output_wav_path):
        audio = AudioSegment.from_file(input_flac_path, format="flac")
        audio.export(output_wav_path, format="wav")

    # 处理单个音频文件
    def process_audio_file(file_path, transcript):
        print(f"处理音频文件: {file_path}")

        # 如果文件是 FLAC 格式，先转换为 WAV 格式
        if file_path.endswith(".flac"):
            wav_file_path = file_path.replace(".flac", ".wav")
            convert_flac_to_wav(file_path, wav_file_path)
            audio_file_path = wav_file_path
        else:
            audio_file_path = file_path

        # 上传音频文件到 S3
        s3_client = boto3.client('s3')
        s3_key = os.path.basename(audio_file_path)
        s3_client.upload_file(audio_file_path, S3_BUCKET_NAME, s3_key)
        s3_url = f"s3://{S3_BUCKET_NAME}/{s3_key}"

        # 使用 Amazon Transcribe 进行语音识别
        try:
            start_time = time.time()
            
            # 开始 Transcribe 任务
            job_name = f"transcribe-job-{int(time.time())}"
            transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                LanguageCode='en-US',
                Media={'MediaFileUri': s3_url},
                MediaFormat='wav',
                OutputBucketName=S3_BUCKET_NAME
            )

            # 等待任务完成
            while True:
                status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
                job_status = status['TranscriptionJob']['TranscriptionJobStatus']
                if job_status in ['COMPLETED', 'FAILED']:
                    break
                time.sleep(5)

            # 获取结果
            if job_status == 'COMPLETED':
                result_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
                try:
                    # 请求转录结果并打印原始响应
                    response = requests.get(result_url)
                    if response.status_code == 200:
                        result_json = response.json()
                        recognized_text = result_json['results']['transcripts'][0]['transcript']
                        print(f"识别文本: {recognized_text}")

                        # 计算 WER
                        error_rate = wer(transcript.lower(), recognized_text.lower())
                        print(f"Word Error Rate (WER): {error_rate:.2%}")
                    else:
                        print(f"请求失败，状态码: {response.status_code}，响应内容: {response.text}")
                        recognized_text = ""
                        error_rate = None
                except Exception as e:
                    print(f"获取转录结果时出错: {e}")
                    recognized_text = ""
                    error_rate = None
            else:
                print(f"Transcription job failed for {file_path}")
                recognized_text = ""
                error_rate = None

            # 记录处理时间
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"处理延迟时间: {elapsed_time:.2f} 秒")
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            recognized_text = ""
            error_rate = None
            elapsed_time = 0

        return error_rate, elapsed_time

    # 遍历文件夹并处理所有音频文件
    def process_audio_folder(folder_path, sample_size=50):
        transcriptions = load_transcriptions(folder_path)

        # 随机抽样50个音频文件
        sampled_transcriptions = random.sample(transcriptions, min(sample_size, len(transcriptions)))

        total_wer = 0
        total_files = 0
        total_time = 0

        # 使用 tqdm 显示进度条
        for audio_file, transcript in tqdm(sampled_transcriptions, desc="处理音频文件", unit="个文件"):
            error_rate, elapsed_time = process_audio_file(audio_file, transcript)

            # 累计 WER 和延迟时间
            if error_rate is not None:
                total_wer += error_rate
                total_files += 1
            total_time += elapsed_time

        # 计算平均 WER
        wer_time = (total_time / total_files) if total_files > 0 else None
        average_wer = (total_wer / total_files) if total_files > 0 else None
        return average_wer, wer_time

    # 执行处理并计算结果
    average_wer, wer_time = process_audio_folder(data_folder, sample_size=50)

    # 输出整体结果
    if average_wer is not None:
        print(f"整体平均 Word Error Rate (WER): {average_wer:.2%}")
    else:
        print("无有效文件计算平均 WER。")
    print(f"总处理时间: {wer_time:.2f} 秒")

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
    print(f"结果已保存到 {output_file}")
def remove_symbols(text):
    # 去除符号，只保留字母和数字
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)
def whisper_tiny(data_folder):
    # 初始化模型和处理器
    results = []  # 用來存儲處理結果的陣列
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    
    # 处理单个音频文件
    def process_audio_file(file_path, transcript):
        separation_times = 0
        print(f"处理音频文件: {file_path}")
        
        # 记录开始时间
        start_time = time.time()

        try:
            # 加载音频文件
            audio, rate = sf.read(file_path)
            if rate != 16000:
                raise ValueError("Whisper 模型需要音频采样率为 16kHz。")
            
            
          
            
            separation_time =clean_audio(file_path, "test.wav")
            # 输出清洁所需的时间
           

            # 准备输入特征
            input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features

            # 推理
            predicted_ids = model.generate(input_features)
            recognized_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # 输出识别结果和对照文本
            print(f"对照文本：{transcript}")
            print(f"识别结果：{recognized_text}")

            # 去除符号
            transcript_clean = remove_symbols(transcript.lower())
            recognized_text_clean = remove_symbols(recognized_text.lower())

            # 计算各种错误率
            error_rate_wer = wer(transcript_clean, recognized_text_clean)
            error_rate_mer = mer(transcript_clean, recognized_text_clean)
            error_rate_wil = wil(transcript_clean, recognized_text_clean)
            error_rate_wip = wip(transcript_clean, recognized_text_clean)
            error_rate_cer = cer(transcript_clean, recognized_text_clean)

            # 输出各种错误率
            print(f"Word Error Rate (WER): {error_rate_wer:.2%}")
            print(f"Match Error Rate (MER): {error_rate_mer:.2%}")
            print(f"Word Information Lost (WIL): {error_rate_wil:.2%}")
            print(f"Word Information Preserved (WIP): {error_rate_wip:.2%}")
            print(f"Character Error Rate (CER): {error_rate_cer:.2%}")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            error_rate_wer = error_rate_mer = error_rate_wil = error_rate_wip = error_rate_cer = None

        # 记录处理时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"处理延迟时间: {elapsed_time:.2f} 秒")
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

    # 遍历文件夹并处理所有音频文件
    def process_audio_folder(folder_path, sample_size=50):
        # 加载对照文本
        def load_transcriptions(folder_path):
            transcriptions = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.trans.txt'):
                        with open(os.path.join(root, file), 'r') as f:
                            for line in f:
                                parts = line.strip().split(maxsplit=1)
                                if len(parts) == 2:
                                    audio_file = os.path.join(root, parts[0] + ".flac")  # 对应的音频文件
                                    transcript = parts[1]  # 对照文本
                                    transcriptions.append((audio_file, transcript))
            return transcriptions

        transcriptions = load_transcriptions(folder_path)

        # 随机抽样音频文件
        sampled_transcriptions = random.sample(transcriptions, sample_size)

        total_wer = total_mer = total_wil = total_wip = total_cer = 0
        total_files = 0
        total_time = 0
        total_separatio =0
        # 使用 tqdm 显示进度条
        for audio_file, transcript in tqdm(sampled_transcriptions, desc="处理音频文件", unit="个文件"):
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

        # 计算平均值
        avg_wer = (total_wer / total_files) if total_files > 0 else None
        avg_mer = (total_mer / total_files) if total_files > 0 else None
        avg_wil = (total_wil / total_files) if total_files > 0 else None
        avg_wip = (total_wip / total_files) if total_files > 0 else None
        avg_cer = (total_cer / total_files) if total_files > 0 else None
        avg_time = (total_time / total_files) if total_files > 0 else None
        avg_st  = (total_separatio / total_files) if total_files > 0 else None


        return avg_wer, avg_mer, avg_wil, avg_wip, avg_cer, avg_time,avg_st

    # 执行处理并输出整体结果
    avg_wer, avg_mer, avg_wil, avg_wip, avg_cer, avg_time,avg_st = process_audio_folder(data_folder, sample_size=50)

    if avg_wer is not None:
        print(f"整体平均 Word Error Rate (WER): {avg_wer:.2%}")
        print(f"整体平均 Match Error Rate (MER): {avg_mer:.2%}")
        print(f"整体平均 Word Information Lost (WIL): {avg_wil:.2%}")
        print(f"整体平均 Word Information Preserved (WIP): {avg_wip:.2%}")
        print(f"整体平均 Character Error Rate (CER): {avg_cer:.2%}")
        
    else:
        print("无有效文件计算平均 WER。")
    print(f"平均处理时间: {avg_time:.2f} 秒")
    save_results_to_excel(results)
    print(f"平均清潔时间: {avg_st:.2f} 秒")
# 调用示例

# amazon_transcribe(data_folder)



data_folder = "test-other/LibriSpeech/test-other"  
data_folder2 = "dev-clean/LibriSpeech/dev-clean" 
whisper_tiny(data_folder)
# google(data_folder)
# amazon_transcribe(data_folder)

# whisper_tiny(data_folder2)
print("_______")
# whisper_tiny(data_folder)