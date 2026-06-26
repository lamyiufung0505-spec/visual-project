import os
import easyocr
from PIL import Image
import matplotlib.pyplot as plt

# =========================
# 路径（你的数据集）
# =========================
img_path = "./dataset/object_detection/data"

# =========================
# 初始化OCR模型
# =========================
print("加载OCR模型...")
reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)

# =========================
# 获取图片
# =========================
image_files = [
    f for f in os.listdir(img_path)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

print("图片数量:", len(image_files))

# =========================
# OCR识别（只跑前5张用于演示）
# =========================
for img_name in image_files[:5]:

    path = os.path.join(img_path, img_name)

    print("\n==============================")
    print("图片:", img_name)

    result = reader.readtext(path)

    if len(result) == 0:
        print("未识别到文字")
        continue

    # 输出识别结果
    for bbox, text, prob in result:
        print(f"{text}  (置信度: {prob:.2f})")

    # 可视化（生成图片用于PPT）
    img = Image.open(path)

    plt.figure()
    plt.imshow(img)
    plt.title(img_name)
    plt.axis("off")

print("\nOCR完成")
plt.show()
