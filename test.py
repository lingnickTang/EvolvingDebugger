from openai import OpenAI
from _KEYS import QEWN_KEY

client = OpenAI(
    api_key= QEWN_KEY, # 替换成真实DashScope的API_KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务endpoint
)

completion = client.chat.completions.create(
    model="qwen-long",
    messages=[
        {
            'role': 'system',
            'content': 'You are a helpful assistant.'
        },
        {
            'role': 'user',
            'content': 'Please introduce yourself in brief.'
        }
    ],
    #stream=True
)
print(completion.choices[0].message.content) # directly achieve all the content 
# streaming processing
# for chunk in completion:
#     if chunk.choices[0].delta.content is not None:
#         print(chunk.choices[0])
