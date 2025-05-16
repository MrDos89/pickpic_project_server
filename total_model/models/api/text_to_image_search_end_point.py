from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from models.text_image_model.text_to_image_model import find_similar_images_by_clip, save_clip_image_features
import os
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEMP_DIR = PROJECT_ROOT / "temp"

# 이미지 검색 설정
SIMILARITY_THRESHOLD = 0.066

router = APIRouter()

class SearchQuery(BaseModel):
    text: str
    similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD

class SearchResponse(BaseModel):
    total_images: int
    matched_images: int
    results: List[Dict]

@router.post("/search/{user_folder}", response_model=SearchResponse)
async def search_images(user_folder: str, query: SearchQuery):
    try:
        # 유저 데이터 폴더 경로 설정
        user_data_dir = DATA_DIR / user_folder
        user_temp_dir = TEMP_DIR / user_folder

        # 폴더가 존재하는지 확인
        if not user_data_dir.exists():
            raise HTTPException(status_code=404, detail=f"유저 폴더를 찾을 수 없습니다: {user_folder}")
        
        # 전체 이미지 개수 계산
        total_images = len([f for f in os.listdir(user_data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        results = find_similar_images_by_clip(
            query.text,
            str(user_data_dir),
            str(user_temp_dir),
            similarity_threshold=query.similarity_threshold
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="유사한 이미지를 찾을 수 없습니다.")
            
        return {
            "total_images": total_images,
            "matched_images": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/{user_folder}/update-features")
async def update_features(user_folder: str):
    try:
        user_data_dir = DATA_DIR / user_folder
        user_temp_dir = TEMP_DIR / user_folder

        if not user_data_dir.exists():
            raise HTTPException(status_code=404, detail=f"유저 폴더를 찾을 수 없습니다: {user_folder}")

        save_clip_image_features(str(user_data_dir), str(user_temp_dir))
        return {"message": "이미지 특징 벡터가 업데이트되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/{user_folder}", response_model=SearchResponse)
async def search_images_get(
    user_folder: str,
    text: str,
    similarity_threshold: float = SIMILARITY_THRESHOLD
):
    try:
        user_data_dir = DATA_DIR / user_folder
        user_temp_dir = TEMP_DIR / user_folder

        if not user_data_dir.exists():
            raise HTTPException(status_code=404, detail=f"유저 폴더를 찾을 수 없습니다: {user_folder}")

        total_images = len([f for f in os.listdir(user_data_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        results = find_similar_images_by_clip(
            text,
            str(user_data_dir),
            str(user_temp_dir),
            similarity_threshold=similarity_threshold
        )
        if not results:
            raise HTTPException(status_code=404, detail="유사한 이미지를 찾을 수 없습니다.")
        return {
            "total_images": total_images,
            "matched_images": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 