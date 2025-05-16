from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import mediapipe as mp
from deepface import DeepFace
import cv2
import shutil
from pathlib import Path

router = APIRouter()

# 모델 준비
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO('yolov8n.pt')
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def detect_objects(image_path, conf=0.3):
    results = yolo_model.predict(image_path, conf=conf, device=device, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    return [(*boxes[i], int(class_ids[i]), confs[i]) for i in range(len(boxes))]

def crop_object(image: Image.Image, box):
    xmin, ymin, xmax, ymax = map(int, box[:4])
    return image.crop((xmin, ymin, xmax, ymax))

def get_clip_embedding_from_pil(pil_img):
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features[0].cpu().numpy()

def detect_faces(image_path):
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
            face_img = image[ymin:ymax, xmin:xmax]
            if face_img.size == 0:
                continue
            try:
                embedding = DeepFace.represent(
                    face_img,
                    model_name="Facenet",
                    enforce_detection=False,
                    detector_backend="skip",
                    device_name="cuda" if device == 'cuda' else 'cpu')
                if embedding:
                    faces.append((xmin, ymin, xmax, ymax, embedding[0]['embedding']))
            except:
                continue
    return faces

def compare_faces(face1_embedding, face2_embedding):
    return np.dot(face1_embedding, face2_embedding) / (np.linalg.norm(face1_embedding) * np.linalg.norm(face2_embedding))

# 업로드 엔드포인트 (data/{user_folder}에 저장)
@router.post("/imgtoimg/upload/{user_folder}")
async def upload_imgtoimg(user_folder: str, file: UploadFile = File(...)):
    save_dir = Path("data") / user_folder
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"message": "업로드 완료", "filename": file.filename}

# 검색 엔드포인트 (data/{user_folder} 내 이미지들과 업로드 쿼리 이미지 비교)
@router.post("/imgtoimg/search/{user_folder}")
def img_to_img_search(user_folder: str, query_img: UploadFile = File(...)):
    # 업로드 파일 저장 (임시)
    temp_dir = "temp/imgtoimg_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    query_img_path = os.path.join(temp_dir, query_img.filename)
    with open(query_img_path, "wb") as f:
        shutil.copyfileobj(query_img.file, f)

    # 비교 대상 이미지 폴더를 data/{user_folder}로 변경
    image_folder = os.path.join("data", user_folder)
    if not os.path.exists(image_folder):
        raise HTTPException(status_code=404, detail=f"유저 폴더를 찾을 수 없습니다: {user_folder}")
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        raise HTTPException(status_code=404, detail="비교할 이미지가 없습니다.")

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
    query_faces = detect_faces(query_img_path)
    query_embedding = get_clip_embedding_from_pil(Image.open(query_img_path).convert("RGB"))

    results = []
    # 얼굴 유사도 계산
    if query_faces:
        query_face_embedding = query_faces[0][4]
        for img_path in image_files:
            faces = detect_faces(img_path)
            for face in faces:
                face_embedding = face[4]
                similarity = compare_faces(query_face_embedding, face_embedding)
                results.append({"img_path": img_path, "similarity": float(similarity), "type": "face"})
    # 객체(전체이미지) 유사도 계산
    for img_path, obj_crops in all_image_objects:
        for crop, emb, box, class_id, conf in obj_crops:
            cosine_similarity = float(np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb)))
            results.append({"img_path": img_path, "similarity": cosine_similarity, "type": "object"})
    # 유사도 순 정렬 및 0.6 이상만 반환
    results = sorted([r for r in results if r["similarity"] >= 0.6], key=lambda x: x["similarity"], reverse=True)
    # 중복 제거 (가장 높은 유사도만)
    unique = {}
    for r in results:
        if r["img_path"] not in unique or r["similarity"] > unique[r["img_path"]]["similarity"]:
            unique[r["img_path"]] = r
    return list(unique.values()) 