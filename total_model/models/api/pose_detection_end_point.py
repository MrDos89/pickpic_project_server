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
    image_name: str

@router.post("/detect-pose/{user_folder}")
async def detect_pose(user_folder: str, query: PoseDetectQuery):
    try:
        user_data_dir = DATA_DIR / user_folder
        if not user_data_dir.exists():
            raise HTTPException(status_code=404, detail=f"유저 폴더를 찾을 수 없습니다: {user_folder}")

        image_path = str(user_data_dir / query.image_name)
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="이미지 파일을 찾을 수 없습니다.")

        # 포즈 타입에 따라 적절한 함수 호출
        if pose_type == "v":
            result, processed_image = detector.detect_v_pose(temp_path)
        # elif pose_type == "fist":
        #     result, processed_image = detector.detect_fist(temp_path)
        elif pose_type == "heart":
            result, processed_image = detector.detect_heart(temp_path)
        # elif pose_type == "military":
        #     result, processed_image = detector.detect_military(temp_path)
        # elif pose_type == "okay":
        #     result, processed_image = detector.detect_okay(temp_path)
        elif pose_type == "thumbs":
            result, processed_image = detector.detect_thumbs(temp_path)
        elif pose_type in ["body", "만세", "점프", "서있음", "앉음", "누워있음"]:
            # YOLO로 사람 박스 검출 후 mediapipe로 랜드마크 추출, classify_pose_mediapipe 호출
            model_path = str(Path(__file__).parent.parent / "pose_detection_model" / "yolov8n-pose.pt")
            yolo = YOLO(model_path)
            img = cv2.imread(temp_path)
            results = yolo.predict(img, stream=False, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
            if len(boxes) == 0:
                result = "사람이 감지되지 않았습니다."
                processed_image = img
            else:
                result_texts = []
                img_result = img.copy()
                with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2) as pose:
                    for idx, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                        person_crop = img[y1:y2, x1:x2]
                        if person_crop.size == 0:
                            result_texts.append(f"사람 {idx+1}: 박스 크기 오류")
                            continue
                        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                        results_mp = pose.process(person_rgb)
                        if results_mp.pose_landmarks:
                            pose_result = pose_detection.classify_pose_mediapipe(results_mp.pose_landmarks.landmark)
                            result_texts.append(f"사람 {idx+1}: {pose_result}")
                            mp.solutions.drawing_utils.draw_landmarks(person_crop, results_mp.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                            img_result[y1:y2, x1:x2] = person_crop
                        else:
                            result_texts.append(f"사람 {idx+1}: 포즈 인식 실패")
                result = " | ".join(result_texts)
                processed_image = img_result
        else:
            return JSONResponse(
                status_code=400,
                content={"message": "지원하지 않는 포즈 타입입니다."}
            )

        result_image_path = str(user_data_dir / "result_image.jpg")
        cv2.imwrite(result_image_path, processed_image)

        return {
            "message": result,
            "result_image_name": os.path.basename(result_image_path)
        }

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
        "pose_types": ["fist", "v", "heart", "military", "okay", "thumbs","만세","점프","서있음","앉음","누워있음"]
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