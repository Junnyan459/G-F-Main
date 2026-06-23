import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "/root/code/Glyph-main/assets/Little_Red_Riding_Hood.png"
            },
            {
                "type": "text",
                "text": "Who pretended to be Little Red Riding Hood's grandmother"
            }
        ],
    }
]
processor = AutoProcessor.from_pretrained("/root/autodl-tmp/Glyph")
model = AutoModelForImageTextToText.from_pretrained(
     pretrained_model_name_or_path="/root/autodl-tmp/Glyph", 
    torch_dtype=torch.bfloat16,
    device_map="auto",
    #local_files_only=True ,
)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=8192)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
print(output_text)