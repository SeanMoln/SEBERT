import speech_recognition as sr
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

recognizer = sr.Recognizer()

def listen_and_recognize():
  
    with sr.Microphone() as source:
        # logging.info("調整麥克風以適應環境噪音...")
        recognizer.adjust_for_ambient_noise(source) 
        # logging.info("已調整，開始監聽...")

        try:
            
            # logging.info("偵測語音中...")
            audio = recognizer.listen(source)  
            # logging.info("偵測到語音，開始處理...")

       
            text = recognizer.recognize_google(audio, language="zh-TW")  
            # logging.info(f"轉換的文字為：{text}")
            return text
        except sr.UnknownValueError:
            logging.warning("")
        except sr.RequestError as e:
            # logging.error(f"無法請求結果；錯誤信息：{e}")
            logging.error("")
        return ""


def continuous_listen_and_recognize():
    try:
        while True:
            text = listen_and_recognize()
            if text:
                logging.info(f"輸出文字：{text}")
    except KeyboardInterrupt:
        logging.info("停止語音偵測程序")


continuous_listen_and_recognize()
