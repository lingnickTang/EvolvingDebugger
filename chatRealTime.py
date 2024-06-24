import pyaudio
import wave
import dashscope
from dashscope.audio.asr import Recognition
from _KEYS import DASHSCOPE_API_KEY
import numpy as np

def calculate_energy(data):
    """
    计算音频数据的能量。
    
    参数:
    data (bytes): 从音频流中读取的原始音频数据。
    
    返回:
    float: 音频数据的能量。
    """
    # 将音频数据转换为整数数组（假设每个样本是一个有符号的16位整数）
    samples = np.frombuffer(data, dtype=np.int16)

    # 计算能量（能量的计算通常是对采样值进行平方，然后求和）
    energy = np.sum(samples**2)

    # 可以选择归一化能量，例如除以采样点的数量
    normalized_energy = energy / len(samples)

    return normalized_energy

# 初始化DashScope API
dashscope.api_key = DASHSCOPE_API_KEY

# 初始化PyAudio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

# 添加静默检测的阈值和计数器
SILENCE_THRESHOLD = 50  # 音频帧能量的静默阈值
SILENCE_COUNT_MAX = 30  # 连续静默帧的最大数量
silence_count = 0  # 当前连续静默帧的数量

p = pyaudio.PyAudio()

# 打开麦克风流
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

recognition = Recognition(model='paraformer-realtime-v1',
                                  format='wav',
                                  sample_rate=RATE,
                                  callback=None)

print("* recording")

try:
    while True: # 多次queries
        # 清空录音帧
        frames = []
        input("输入任意键开始录音：")
        # 开始录制
        while True: # 判断一个query的停止
            data = stream.read(CHUNK)
            frames.append(data)
            energy = calculate_energy(data)  # 假设这是一个计算音频帧能量的函数
            # 判断是否为静默帧
            if energy < SILENCE_THRESHOLD:
                silence_count += 1
            else:
                silence_count = 0

        # 如果达到最大静默帧数，认为用户已结束讲话
            if silence_count >= SILENCE_COUNT_MAX:
                print("Detected end of speech.")
                break

        # 将录制的数据写入WAV文件
        wf = wave.open('output.wav', 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        # 使用DashScope进行语音识别
        result = recognition.call('output.wav')

        # 输出识别结果
        for sentence in result.get_sentence():
            print(sentence['text'])
except KeyboardInterrupt:
    pass

print("* done recording")

# 关闭资源
stream.stop_stream()
stream.close()
p.terminate()