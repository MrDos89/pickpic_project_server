import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog

# 모델 준비
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features[0].cpu().numpy()

# 파일 선택창 띄우기
root = tk.Tk()
root.withdraw()  # 메인 윈도우 숨기기
query_image_path = filedialog.askopenfilename(title="기준이 될 이미지를 선택하세요", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

if not query_image_path:
    print("이미지를 선택하지 않았습니다. 프로그램을 종료합니다.")
    exit()

# 비교할 이미지가 들어있는 폴더
image_folder = "./"
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# 기준 이미지 임베딩
query_embedding = get_clip_embedding(query_image_path)

# 각 이미지와의 유사도 계산
results = []
for img_path in image_files:
    emb = get_clip_embedding(img_path)
    cosine_similarity = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
    results.append((img_path, cosine_similarity))

# 유사도 순으로 정렬
results.sort(key=lambda x: x[1], reverse=True)

# 결과 출력
for img_path, sim in results:
    print(f"{img_path}: 유사도 {sim:.4f}")

# 예시: 유사도가 0.8 이상인 이미지만 출력
print("\n유사도가 0.8 이상인 이미지:")
for img_path, sim in results:
    if sim >= 0.8:
        print(f"{img_path}: {sim:.4f}")
