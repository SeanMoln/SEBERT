import azure.cognitiveservices.speech as speechsdk
import time
import openai

# 设置语音服务的参数
openai.api_key = ""
speech_key = ""  # 替换为你的 API 密钥
service_region = "japanwest"  


# 创建语音配置
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_recognition_language = "zh-TW"  # 使用繁体中文识别

# 配置音频输入
audio_config = speechsdk.AudioConfig(use_default_microphone=True)
recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# 定义识别事件的回调函数
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
        print(f"语音识别到翻译输出的延迟: {total_delay:.2f} 秒")  # 实时输出识别结果
    elif evt.result.reason == speechsdk.ResultReason.NoMatch:
        print("未识别到语音")
    elif evt.result.reason == speechsdk.ResultReason.Canceled:
        cancellation = evt.result.cancellation_details
        print(f"识别取消: {cancellation.reason}")
        if cancellation.reason == speechsdk.CancellationReason.Error:
            print(f"错误详情: {cancellation.error_details}")

# 连接事件处理器
 
recognizer.recognized.connect(recognized_handler)
print("请开始说话...")

# 开启连续识别
recognizer.start_continuous_recognition()

try:
    # 持续运行程序，等待识别事件
    while True:
        pass
except KeyboardInterrupt:
    print("停止监听...")
finally:
    recognizer.stop_continuous_recognition()