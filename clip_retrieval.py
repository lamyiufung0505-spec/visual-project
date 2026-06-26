import os
import torch
import open_clip
from PIL import Image
import numpy as np
from tqdm import tqdm

# =========================
# 路径
# =========================
base_path = "./dataset/image_retrieval/base"
query_path = "./dataset/image_retrieval/query"

# =========================
# 设备
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备:", device)

# =========================
# CLIP模型
# =========================
print("加载CLIP模型...")

model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

model = model.to(device)
model.eval()

# =========================
# 提取特征函数
# =========================
def get_image_feature(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print("❌ 读取失败:", img_path)
        return None

    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.encode_image(image)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.cpu().numpy()

# =========================
# 递归读取所有图片（关键修复）
# =========================
image_suffix = (".jpg", ".jpeg", ".png", ".bmp")

def load_images(root_path):
    files = []
    paths = []

    for root, dirs, fs in os.walk(root_path):
        for f in fs:
            if f.lower().endswith(image_suffix):
                files.append(f)
                paths.append(os.path.join(root, f))

    return files, paths

base_files, base_paths = load_images(base_path)
query_files, query_paths = load_images(query_path)

print("Base图片数量:", len(base_paths))
print("Query图片数量:", len(query_paths))

# =========================
# 提取 base 特征
# =========================
print("\n开始提取Base特征...")

base_features = []
base_names = []

for path, name in tqdm(zip(base_paths, base_files), total=len(base_paths)):

    feat = get_image_feature(path)

    if feat is not None:
        base_features.append(feat)
        base_names.append(name)

# 防止空
if len(base_features) == 0:
    raise ValueError("❌ base特征为空，请检查图片是否损坏")

base_features = np.vstack(base_features)

print("Base特征提取完成:", len(base_features))

# =========================
# 检索
# =========================
print("\n开始检索...\n")

for q_path, q_name in list(zip(query_paths, query_files))[:5]:

    q_feat = get_image_feature(q_path)

    if q_feat is None:
        continue

    sims = (base_features @ q_feat.T).squeeze()

    topk = np.argsort(-sims)[:5]

    print("=" * 60)
    print("Query:", q_name)

    for rank, idx in enumerate(topk, 1):
        print(f"Top{rank}: {base_names[idx]} | score={sims[idx]:.4f}")

print("\n✅ 完成！")