from PIL import Image
import os

def check_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                try:
                    with Image.open(path) as img:
                        img.verify()  # 验证文件完整性
                except Exception as e:
                    print(f"损坏图片: {path}, 错误: {e}")

# 运行检查，替换为你的图片目录
check_images("/root/autodl-tmp/Data/ruler/rendered_images/126000")