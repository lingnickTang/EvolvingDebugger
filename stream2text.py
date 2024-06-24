# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html

import pyaudio
import dashscope
from dashscope.audio.asr import (Recognition, RecognitionCallback,
                                 RecognitionResult)
from _KEYS import DASHSCOPE_API_KEY
import numpy as np

dashscope.api_key=DASHSCOPE_API_KEY

mic = None
stream = None
query = None
# 添加静默检测的阈值和计数器
SILENCE_THRESHOLD = 50  # 音频帧能量的静默阈值
SILENCE_COUNT_MAX = 10  # 连续静默帧的最大数量
silence_count = 0  # 当前连续静默帧的数量

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

# 假设你已经从音频流中读取了数据
# data = stream.read(6400, exception_on_overflow=False)

# 调用函数计算能量
# energy = calculate_energy(data)

# 根据能量值做出判断，例如是否低于某个阈值

class Callback(RecognitionCallback):
    def on_open(self) -> None:
        global mic
        global stream
        print('RecognitionCallback open.')
        mic = pyaudio.PyAudio()
        stream = mic.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=16000,
                          input=True)

    def on_close(self) -> None:
        global mic
        global stream
        print('RecognitionCallback close.')
        stream.stop_stream()
        stream.close()
        mic.terminate()
        stream = None
        mic = None

    def on_event(self, result: RecognitionResult) -> None:
        print('RecognitionCallback sentence: {}'.format(result.get_sentence()))

callback = Callback()
recognition = Recognition(model='paraformer-realtime-v1',
                          format='pcm',
                          sample_rate=16000,
                          callback=callback)
recognition.start()

while True:
    input("输入任意键开始录音：")
    audio_data_list = []
    while True:
        if stream:
            data = stream.read(3200, exception_on_overflow = False)
            audio_data_list.append(data)
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
        else:
            break
    combined_data = b''.join(audio_data_list)
    recognition.send_audio_frame(combined_data)  

recognition.stop()