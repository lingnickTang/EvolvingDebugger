# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html

import requests
from http import HTTPStatus

import dashscope
from dashscope.audio.asr import Recognition
from _KEYS import DASHSCOPE_API_KEY

dashscope.api_key = DASHSCOPE_API_KEY

r = requests.get(
    'https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/paraformer/hello_world_female2.wav'
)
with open('asr_example.wav', 'wb') as f:
    f.write(r.content)

recognition = Recognition(model='paraformer-realtime-v1',
                          format='wav',
                          sample_rate=16000,
                          callback=None)
result = recognition.call('asr_example.wav')
print(result.get_sentence()[0]['text'])