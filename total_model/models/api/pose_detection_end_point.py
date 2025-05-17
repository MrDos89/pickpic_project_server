import os
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ..pose_detection_model import PoseDetector
from ..pose_detection_model import pose_detection
import mediapipe as mp
from ultralytics import YOLO
from pathlib import Path
from fastapi import HTTPException
from pydantic import BaseModel

router = APIRouter()
detector = PoseDetector()

# 프로젝트 루트 디렉토리 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEMP_DIR = PROJECT_ROOT / "temp"

class PoseDetectQuery(BaseModel):
    pose_type: str

@router.post("/detect-pose/{user_folder}")
async def detect_pose(user_folder: str, query: PoseDetectQuery):
    try:
        user_data_dir = DATA_DIR / user_folder
        if not user_data_dir.exists():
            raise HTTPException(status_code=404, detail=f"유저 폴더를 찾을 수 없습니다: {user_folder}")

        # user_folder 내 모든 이미지 파일 순회
        image_files = [f for f in os.listdir(user_data_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        matched_images = []
        for image_name in image_files:
            image_path = str(user_data_dir / image_name)
            try:
                # pose_type에 따라 분류 함수 호출
                if query.pose_type == "브이":
                    result, _ = detector.detect_v_pose(image_path)
                elif query.pose_type == "하트":
                    result, _ = detector.detect_heart(image_path)
                elif query.pose_type == "최고":
                    result, _ = detector.detect_thumbs(image_path)
                elif query.pose_type in ["body", "만세", "점프", "서있음", "앉음", "누워있음"]:
                    model_path = str(Path(__file__).parent.parent / "pose_detection_model" / "yolov8n-pose.pt")
                    yolo = YOLO(model_path)
                    img = cv2.imread(image_path)
                    results = yolo.predict(img, stream=False, verbose=False)
                    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
                    if len(boxes) == 0:
                        continue
                    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2) as pose:
                        for box in boxes:
                            x1, y1, x2, y2 = box.astype(int)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                            person_crop = img[y1:y2, x1:x2]
                            if person_crop.size == 0:
                                continue
                            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                            results_mp = pose.process(person_rgb)
                            if results_mp.pose_landmarks:
                                pose_result = pose_detection.classify_pose_mediapipe(results_mp.pose_landmarks.landmark)
                                # pose_type이 결과에 포함되어 있으면 추가
                                if query.pose_type in pose_result:
                                    matched_images.append(image_name)
                                    break
                    continue
                else:
                    continue
                print(image_name, result)
                # pose_type이 결과에 포함되어 있으면 추가
                if query.pose_type in result:
                    matched_images.append(image_name)
            except Exception:
                continue
        return {"results": matched_images}
    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"처리 중 오류가 발생했습니다: {str(e)}"}
        )

@router.get("/pose-types")
async def get_pose_types():
    return {
        "pose_types": ["브이", "하트", "최고","만세","점프","서있음","앉음","누워있음"]
        # fist, military, okay 포즈 타입 추가
    }

@router.post("/classify-pose/{model_type}")
async def classify_pose_api(model_type: str, file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse(status_code=400, content={"message": "이미지를 불러올 수 없습니다."})
    try:
        # model_type 구분 없이 model.py의 classify/ensemble 함수만 사용
        result_text = pose_detection.classify_pose_ensemble(image)
        return {"result": result_text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"처리 중 오류가 발생했습니다: {str(e)}"}) 