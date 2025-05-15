from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ..pose_detector import PoseDetector
from ..models import model
import mediapipe as mp
from ultralytics import YOLO

router = APIRouter()
detector = PoseDetector()

@router.post("/detect-pose/{pose_type}")
async def detect_pose(pose_type: str, file: UploadFile = File(...)):
    # 이미지 파일 읽기
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return JSONResponse(
            status_code=400,
            content={"message": "이미지를 불러올 수 없습니다."}
        )
    
    # 임시 파일로 저장
    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, image)
    
    try:
        # 포즈 타입에 따라 적절한 함수 호출
        if pose_type == "fist":
            result, processed_image = detector.detect_fist(temp_path)
        elif pose_type == "v":
            result, processed_image = detector.detect_v_pose(temp_path)
        elif pose_type == "heart":
            result, processed_image = detector.detect_heart(temp_path)
        elif pose_type == "military":
            result, processed_image = detector.detect_military(temp_path)
        elif pose_type == "okay":
            result, processed_image = detector.detect_okay(temp_path)
        elif pose_type == "thumbs":
            result, processed_image = detector.detect_thumbs(temp_path)
        else:
            return JSONResponse(
                status_code=400,
                content={"message": "지원하지 않는 포즈 타입입니다."}
            )
        
        # 처리된 이미지를 base64로 인코딩
        _, buffer = cv2.imencode('.jpg', processed_image)
        image_base64 = buffer.tobytes()
        
        return {
            "message": result,
            "image": image_base64
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"처리 중 오류가 발생했습니다: {str(e)}"}
        )
    finally:
        # 임시 파일 삭제
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/pose-types")
async def get_pose_types():
    return {
        "pose_types": ["fist", "v", "heart", "military", "okay", "thumbs"]
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
        result_text = model.classify_pose_ensemble(image)
        return {"result": result_text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"처리 중 오류가 발생했습니다: {str(e)}"}) 