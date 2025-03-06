import azure.cognitiveservices.speech as speechsdk
import time
import openai

# 參數設置

# API KEY 設置
openai.api_key = ""#
speech_key = ""  
service_region = "japanwest"  


#azure-Speech模型參數設置
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_recognition_language = "zh-TW"  # 使用繁体中文识别

# 音頻輸入設置
audio_config = speechsdk.AudioConfig(use_default_microphone=True)
recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# 主函數
def recognized_handler(evt):
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:

        
        recognition_time = time.time()
        response =  openai.chat.completions.create(
            
                    model="gpt-3.5-turbo", 
                    messages=[
                        {"role": "system", "content": "你是專業的翻譯師."},
                        {"role": "user", "content": f"將下面輸入語言同時翻譯成中文及英文 只需要輸出翻譯後的 中文 下一行英文:\n\n{evt.result.text}"}
                    ]
                ) 
        chinese_translation = response.choices[0].message.content
        translation_time = time.time()
        print(chinese_translation)
        total_delay = translation_time - recognition_time
        print(f"翻譯延遲: {total_delay:.2f} 秒")  
    elif evt.result.reason == speechsdk.ResultReason.NoMatch:
        print("未識別語音")
    elif evt.result.reason == speechsdk.ResultReason.Canceled:
        cancellation = evt.result.cancellation_details
        print(f"識別取消: {cancellation.reason}")
        if cancellation.reason == speechsdk.CancellationReason.Error:
            print(f"error: {cancellation.error_details}")


 
recognizer.recognized.connect(recognized_handler)
print("请开始说话...")

# 連續識別
recognizer.start_continuous_recognition()

try:
   
    while True:
        pass
except KeyboardInterrupt:
    print("停止監聽...")
finally:
    recognizer.stop_continuous_recognition()