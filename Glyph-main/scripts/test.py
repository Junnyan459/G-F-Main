from word2png_function import text_to_images

CONFIG_EN_PATH = '../config/config_en.json'
OUTPUT_DIR = './output_images'
INPUT_FILE = './input.txt'

# Read text from file
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

# Convert text to images
images = text_to_images(
    text=text,
    output_dir=OUTPUT_DIR,
    config_path=CONFIG_EN_PATH,
    unique_id='test_001'
)

print(f"\nGenerated {len(images)} image(s):")
for img_path in images:
    print(f"  {img_path}")