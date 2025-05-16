# 필요 라이브러리 자동 설치
try:
    import os
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image, ImageTk
    import torch
    import numpy as np
    import tkinter as tk
    from tkinter import filedialog, simpledialog
    from ultralytics import YOLO  # YOLOv8 import
    import mediapipe as mp
    from deepface import DeepFace
    import cv2
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    import os
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image, ImageTk
    import torch
    import numpy as np
    import tkinter as tk
    from tkinter import filedialog, simpledialog
    from ultralytics import YOLO  # YOLOv8 import
    import mediapipe as mp
    from deepface import DeepFace
    import cv2

# GPU 사용가능 여부 확인. tensorflow GPU가 안되서 넣은것. !! 지워도 무관.
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 모델 준비
device = 0 if torch.cuda.is_available() else 'cpu'
# !! 모델 경로 자동 생성 및 yolo 모델 다운.
yolo_model = YOLO('yolov8n.pt')
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Yolo 객체 탐지
def detect_objects(image_path, conf=0.3):
    """
    이미지에서 객체 검출 (YOLOv8)
    return: [(xmin, ymin, xmax, ymax, class_id, conf), ...]
    """
    results = yolo_model.predict(image_path, conf=conf, device=device, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    return [(*boxes[i], int(class_ids[i]), confs[i]) for i in range(len(boxes))]

# 객체 영역 추출
def crop_object(image: Image.Image, box):
    """
    box: (xmin, ymin, xmax, ymax)
    return: cropped PIL Image
    """
    xmin, ymin, xmax, ymax = map(int, box[:4])
    return image.crop((xmin, ymin, xmax, ymax))

# CLIP 이미지 임베딩 추출
def get_clip_embedding_from_pil(pil_img):
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features[0].cpu().numpy()

# 얼굴 검출
def detect_faces(image_path):
    """
    이미지에서 얼굴 검출 (MediaPipe 사용)
    return: [(xmin, ymin, xmax, ymax, face_embedding), ...]
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    results = face_detection.process(image_rgb)
    faces = []

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            xmin = int(bbox.xmin * width)
            ymin = int(bbox.ymin * height)
            xmax = int((bbox.xmin + bbox.width) * width)
            ymax = int((bbox.ymin + bbox.height) * height)

            # 얼굴 영역 추출
            face_img = image[ymin:ymax, xmin:xmax]
            if face_img.size == 0:
                continue

            try:
                embedding = DeepFace.represent(
                    face_img,
                    model_name="Facenet",
                    enforce_detection=False,
                    detector_backend="skip",
                    device_name="cuda" if torch.cuda.is_available() else "cpu"
                )
                if embedding:
                    faces.append((xmin, ymin, xmax, ymax, embedding[0]['embedding']))
            except:
                continue

    return faces

def compare_faces(face1_embedding, face2_embedding):
    """
    두 얼굴 임베딩 간의 유사도 계산
    """
    return np.dot(face1_embedding, face2_embedding) / (np.linalg.norm(face1_embedding) * np.linalg.norm(face2_embedding))

# 기준 이미지 선택
query_image_path = input("기준이 될 이미지 경로를 입력하세요 (상대경로나 절대경로 가능): ").strip()

if not query_image_path or not os.path.isfile(query_image_path):
    print("유효한 이미지 경로를 입력하지 않았습니다. 프로그램을 종료합니다.")
    exit()

# !! 기준 이미지 경로에서 폴더명과 파일명 추출
query_folder = os.path.basename(os.path.dirname(query_image_path))  # 폴더 이름
query_filename = os.path.basename(query_image_path)  # 파일 이름

print(f"기준 이미지 폴더: {query_folder}")
print(f"기준 이미지 이름: {query_filename}")

# !! 비교할 이미지 경로 설정
image_folder = os.path.dirname(query_image_path)
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# 각 이미지에서 객체 검출 및 crop, 임베딩 추출
all_image_objects = []
for img_path in image_files:
    img = Image.open(img_path).convert("RGB")
    objects = detect_objects(img_path)
    obj_crops = []
    for box in objects:
        crop = crop_object(img, box)
        emb = get_clip_embedding_from_pil(crop)
        obj_crops.append((crop, emb, box[:4], box[4], box[5]))
    all_image_objects.append((img_path, obj_crops))

# 기준 이미지에서 얼굴과 객체 검출
query_faces = detect_faces(query_image_path)
query_embedding = get_clip_embedding_from_pil(Image.open(query_image_path).convert("RGB"))

# 각 이미지와의 유사도 계산
results = []

# 얼굴 유사도 계산
if query_faces:
    query_face_embedding = query_faces[0][4]
    for img_path in image_files:
        faces = detect_faces(img_path)
        for face in faces:
            face_embedding = face[4]
            similarity = compare_faces(query_face_embedding, face_embedding)
            results.append((img_path, similarity, "face"))

# 객체(전체이미지) 유사도 계산
for img_path, obj_crops in all_image_objects:
    for crop, emb, box, class_id, conf in obj_crops:
        cosine_similarity = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
        results.append((img_path, cosine_similarity, "object"))

# 유사도 순으로 정렬
results.sort(key=lambda x: x[1], reverse=True)

# !! 유사도 0.6 이상. 7.5 ~ 6.5 추천
similar_results = [(img_path, sim, type_) for img_path, sim, type_ in results if sim >= 0.6]

# 중복 제거 (같은 이미지에 대해 가장 높은 유사도를 가진 결과만 유지)
unique_similar = {}
for img_path, sim, type_ in similar_results:
    if img_path not in unique_similar or sim > unique_similar[img_path][0]:
        unique_similar[img_path] = (sim, type_)

similar_results = [(img_path, sim, type_) for img_path, (sim, type_) in unique_similar.items()]

if results:
    best_img_path, best_sim, best_type = max(results, key=lambda x: x[1])
else:
    print("비교할 결과가 없습니다. (results가 비어있음)")
    exit()

# --- 여기서부터 창 여러 개 띄우기 ---
popup = tk.Tk()
popup.title("기준 이미지 vs 유사한 이미지들")

msg = "이 이미지와 유사한 이미지들입니다!" if similar_results else "일치하는 이미지가 없습니다."

# !! 문구 꾸미기. 수정 ㅇㅋ
msg_label = tk.Label(popup, text=msg, font=("Arial", 16, "bold"), fg="blue")
msg_label.grid(row=0, column=0, columnspan=max(2, len(similar_results)), pady=(10, 0))

# 기준 이미지
query_img = Image.open(query_image_path).resize((256, 256))
query_img_tk = ImageTk.PhotoImage(query_img)
label1 = tk.Label(popup, image=query_img_tk, text="기준 이미지", compound="top")
label1.image = query_img_tk
label1.grid(row=1, column=0, padx=20, pady=20)

img_refs = [query_img_tk]  # 이미지 참조 유지

# 유사한 이미지들 (원본 전체). 4장씩
IMAGES_PER_ROW = 4

if similar_results:
    display_images = similar_results
else:
    display_images = []

if display_images:
    for idx, (img_path, sim, type_) in enumerate(display_images):
        img = Image.open(img_path).resize((256, 256))
        img_tk = ImageTk.PhotoImage(img)
        label = tk.Label(
            popup,
            image=img_tk,
            text=f"{os.path.basename(img_path)} ({type_})\n유사도: {sim:.4f}",
            compound="top"
        )
        label.image = img_tk
        img_refs.append(img_tk)
        row = 1 + (idx // IMAGES_PER_ROW)
        col = (idx % IMAGES_PER_ROW) + 1  # +1: 0번은 기준 이미지
        label.grid(row=row, column=col, padx=20, pady=20)

popup.mainloop()
