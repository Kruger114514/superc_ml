"""
测试 DeepSeek API 连接
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("找不到 DEEPSEEK_API_KEY, 检查 .env 文件")

# DeepSeek 
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个材料科学助手"},
        {"role": "user", "content": "用一句话告诉我 MgB2 的超导临界温度是多少"}
    ],
    temperature=0.3,
)

print("=" * 50)
print("DeepSeek 回复:")
print(response.choices[0].message.content)
print("=" * 50)
print(f"消耗 token: {response.usage.total_tokens}")