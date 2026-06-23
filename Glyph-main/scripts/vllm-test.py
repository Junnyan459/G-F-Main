import zai
from zai import ZhipuAiClient
import base64

client = ZhipuAiClient(api_key="4d86190163a341bba69de7cb7505efc7.fpjipPDXxDcoTDnB")  # 替换为你的完整API Key

# 对本地图片进行 Base64 编码
with open("/root/code/Glyph-main/scripts/output_images/test_001/page_001.png", "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

response = client.chat.completions.create(
    model="glm-4.1v-thinking-flashx",  # 请确认模型名称，官方示例为 "glm-4.1v-thinking-flashx"[reference:5]
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Based on the story in the figures, what is the ending of wolf?"},
                {"type": "image_url", "image_url": {"url": img_base64}}
            ]
        }
    ]
)
print(response.choices[0].message.content)