from openai import OpenAI

# Configured by environment variables
client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=3600)

messages = [
    {
        "role": "user",
        "content": [
            # {
            #     "type": "image_url",
            #     "image_url": {
            #         "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/RealWorld/RealWorld-04.png"
            #     }
            # },
            {
                "type": "text",
                "text": "What is the side effect of drug aspirin?"
            }
        ]
    }
]

chat_response = client.chat.completions.create(
    model="Qwen3.8-27B",
    messages=messages,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }, 
)
print("Chat response:", chat_response)
