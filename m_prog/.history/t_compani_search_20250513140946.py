import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageTk
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
query_image_path = filedialog.askopenfilename(
    title="기준이 될 이미지를 선택하세요",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)
root.destroy()  # 파일 선택 후 Tk 윈도우 완전히 닫기

if not query_image_path:
    print("이미지를 선택하지 않았습니다. 프로그램을 종료합니다.")
    exit()

# 비교할 이미지가 들어있는 폴더
image_folder = "./img"
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

#!!!!!!!!!!!!!!!!!!!! 
# 유사도가 0.7 이상인 이미지 중 가장 유사도가 높은 이미지 하나만 출력
similar_images = [(img_path, sim) for img_path, sim in results if sim >= 0.7]

if similar_images:
    best_img_path, best_sim = max(similar_images, key=lambda x: x[1])
else:
    best_img_path, best_sim = max(results, key=lambda x: x[1])

# --- 여기서부터 창 여러 개 띄우기 ---
popup = tk.Tk()
popup.title("기준 이미지 vs 유사한 이미지들")

# 안내 문구 결정
if similar_images:
    msg = "이 이미지와 유사한 이미지들입니다!"
else:
    msg = "일치하는 이미지는 없지만 유사한 이미지를 보여줍니다."

# 안내 문구 라벨 추가
msg_label = tk.Label(popup, text=msg, font=("Arial", 16, "bold"), fg="blue")
msg_label.grid(row=0, column=0, columnspan=max(2, len(similar_images)), pady=(10, 0))

# 기준 이미지
query_img = Image.open(query_image_path).resize((256, 256))
query_img_tk = ImageTk.PhotoImage(query_img)
label1 = tk.Label(popup, image=query_img_tk, text="기준 이미지", compound="top")
label1.image = query_img_tk
label1.grid(row=1, column=0, padx=20, pady=20)

img_refs = [query_img_tk]  # 이미지 참조 유지

# 유사한 이미지들 (0.8 이상)
if similar_images:
    for idx, (img_path, sim) in enumerate(similar_images):
        img = Image.open(img_path).resize((256, 256))
        img_tk = ImageTk.PhotoImage(img)
        label = tk.Label(
            popup,
            image=img_tk,
            text=f"{os.path.basename(img_path)}\n유사도: {sim:.4f}",
            compound="top"
        )
        label.image = img_tk
        img_refs.append(img_tk)
        label.grid(row=1, column=idx + 1, padx=20, pady=20)
else:
    # 0.8 이상이 없으면 가장 유사한 이미지 하나만
    img = Image.open(best_img_path).resize((256, 256))
    img_tk = ImageTk.PhotoImage(img)
    label = tk.Label(
        popup,
        image=img_tk,
        text=f"{os.path.basename(best_img_path)}\n유사도: {best_sim:.4f}",
        compound="top"
    )
    label.image = img_tk
    img_refs.append(img_tk)
    label.grid(row=1, column=1, padx=20, pady=20)

popup.mainloop()