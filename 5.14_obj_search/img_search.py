import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageTk
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
from ultralytics import YOLO  # YOLOv8 import

#ㅇㅇ
# 모델 준비
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
yolo_model = YOLO('model/yolov8n.pt')  # 가장 가벼운 YOLOv8 모델 사용

def detect_objects(image_path, conf=0.3):
    """
    이미지에서 객체 검출 (YOLOv8)
    return: [(xmin, ymin, xmax, ymax, class_id, conf), ...]
    """
    results = yolo_model(image_path, conf=conf)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
    class_ids = results[0].boxes.cls.cpu().numpy()  # (N,)
    confs = results[0].boxes.conf.cpu().numpy()  # (N,)
    return [(*boxes[i], int(class_ids[i]), confs[i]) for i in range(len(boxes))]

def crop_object(image: Image.Image, box):
    """
    box: (xmin, ymin, xmax, ymax)
    return: cropped PIL Image
    """
    xmin, ymin, xmax, ymax = map(int, box[:4])
    return image.crop((xmin, ymin, xmax, ymax))

def get_clip_embedding_from_pil(pil_img):
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features[0].cpu().numpy()

# 비교할 이미지가 들어있는 폴더
image_folder = "./img"
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# 각 이미지에서 객체 검출 및 crop, 임베딩 추출
all_image_objects = []  # [(img_path, [ (crop_img, emb, box, class_id, conf), ... ]), ...]
for img_path in image_files:
    img = Image.open(img_path).convert("RGB")
    objects = detect_objects(img_path)
    obj_crops = []
    for box in objects:
        crop = crop_object(img, box)
        emb = get_clip_embedding_from_pil(crop)
        obj_crops.append((crop, emb, box[:4], box[4], box[5]))
    all_image_objects.append((img_path, obj_crops))

# --- 이하: 기준 이미지에서 객체 검출/선택, 유사도 비교, UI 등은 2차로 추가 예정 ---

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

# 유사도 기준값 입력 받기 (기본값 0.75)
try:
    similarity_threshold = float(input("유사도 임계값을 입력하세요 (0~1, 예: 0.75, 엔터시 기본값 0.75): ") or 0.75)
    if not (0.0 <= similarity_threshold <= 1.0):
        print("0~1 사이의 값을 입력해야 합니다. 기본값 0.75로 진행합니다.")
        similarity_threshold = 0.75
except ValueError:
    print("잘못된 입력입니다. 기본값 0.75로 진행합니다.")
    similarity_threshold = 0.75

# 기준 이미지 임베딩
query_embedding = get_clip_embedding_from_pil(Image.open(query_image_path).convert("RGB"))

# 각 이미지와의 유사도 계산
results = []
for img_path, obj_crops in all_image_objects:
    for crop, emb, box, class_id, conf in obj_crops:
        cosine_similarity = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
        results.append((img_path, cosine_similarity))

# 유사도 순으로 정렬
results.sort(key=lambda x: x[1], reverse=True)

# 유사도가 유사도 기준 이상인 이미지 중 가장 유사도가 높은 이미지 하나만 출력
# (동일 이미지가 여러 객체로 중복될 수 있으니 set으로 중복 제거)
similar_images = [(img_path, sim) for img_path, sim in results if sim >= similarity_threshold]
unique_similar = {}
for img_path, sim in similar_images:
    if img_path not in unique_similar or sim > unique_similar[img_path]:
        unique_similar[img_path] = sim
similar_images = list(unique_similar.items())  # [(img_path, sim), ...]

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
    msg = "일치하는 이미지가 없습니다."

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

# 유사한 이미지들 (원본 전체)
IMAGES_PER_ROW = 4  # 한 줄에 최대 4장

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
        row = 1 + (idx // IMAGES_PER_ROW)
        col = (idx % IMAGES_PER_ROW) + 1  # +1: 0번은 기준 이미지
        label.grid(row=row, column=col, padx=20, pady=20)

popup.mainloop()