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

# 결과 출력
for img_path, sim in results:
    print(f"{img_path}: 유사도 {sim:.4f}")

# 유사도가 0.8 이상인 이미지 중 가장 유사도가 높은 이미지 하나만 출력
# similar_images = [(img_path, sim) for img_path, sim in results if sim >= 0.8]
# 
# if similar_images:
#     best_img_path, best_sim = max(similar_images, key=lambda x: x[1])
#     print(f"\n가장 유사도가 높은 이미지: {os.path.basename(best_img_path)} (유사도: {best_sim:.4f})")
# else:
#     # print("\n유사도가 0.8 이상인 이미지가 없습니다.")
#     best_img_path, best_sim = max(results, key=lambda x: x[1])
#     print(f"\n0.8 이상은 없지만, 가장 유사한 이미지: {os.path.basename(best_img_path)} (유사도: {best_sim:.4f})")

# --- 여기서부터 창 두 개 띄우기 ---
popup = tk.Tk()  # 이제 새로 Tk() 생성해도 안전
popup.title("기준 이미지 vs 가장 유사한 이미지")

# 안내 문구 결정
if best_sim >= 0.8:
    msg = "이 이미지랑 일치해요!"
else:
    msg = "일치하는 이미지는 없지만 유사한 이미지입니다."

# 안내 문구 라벨 추가
msg_label = tk.Label(popup, text=msg, font=("Arial", 16, "bold"), fg="blue")
msg_label.grid(row=0, column=0, columnspan=2, pady=(10, 0))

# 기준 이미지
query_img = Image.open(query_image_path).resize((256, 256))
query_img_tk = ImageTk.PhotoImage(query_img)
label1 = tk.Label(popup, image=query_img_tk, text="기준 이미지", compound="top")
label1.image = query_img_tk
label1.grid(row=1, column=0, padx=20, pady=20)

# 가장 유사한 이미지
best_img = Image.open(best_img_path).resize((256, 256))
best_img_tk = ImageTk.PhotoImage(best_img)
label2 = tk.Label(popup, image=best_img_tk, text=f"유사 이미지\n{os.path.basename(best_img_path)}\n유사도: {best_sim:.4f}", compound="top")
label2.image = best_img_tk
label2.grid(row=1, column=1, padx=20, pady=20)

popup.mainloop()