import os
import torchaudio
from torch.utils.data import Dataset

def load_transcriptions(root_dir):
    transcriptions = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.trans.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    for line in f:
                        parts = line.strip().split(maxsplit=1)
                        audio_file = os.path.join(root, parts[0] + ".flac")
                        transcript = parts[1]
                        transcriptions.append((audio_file, transcript))
    return transcriptions

# 加載 dev-clean 的數據
data_dir = "dev-clean/LibriSpeech/dev-clean"
data = load_transcriptions(data_dir)

# 查看數據
for audio_file, transcript in data[:5]:
    print(f"Audio: {audio_file}, Transcript: {transcript}")

print("----------")
# 讀取一個音頻文件
waveform, sample_rate = torchaudio.load(data[0][0])
print(f"Waveform Shape: {waveform.shape}, Sample Rate: {sample_rate}")


class LibriSpeechDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_file, transcript = self.data[idx]
        waveform, sample_rate = torchaudio.load(audio_file)
        return waveform, sample_rate, transcript

# 初始化數據集
dataset = LibriSpeechDataset(data)

# 查看數據集中的一個樣本
waveform, sample_rate, transcript = dataset[0]
print(f"Waveform: {waveform.shape}, Sample Rate: {sample_rate}, Transcript: {transcript}")
