from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from src.core.image_search import find_similar_images_by_clip, save_clip_image_features
from src.core.config import DATA_DIR, TEMP_DIR, SIMILARITY_THRESHOLD
import os

router = APIRouter()

class SearchQuery(BaseModel):
    text: str
    similarity_threshold: Optional[float] = SIMILARITY_THRESHOLD

class SearchResponse(BaseModel):
    total_images: int
    matched_images: int
    results: List[Dict]

@router.post("/search", response_model=SearchResponse)
async def search_images(query: SearchQuery):
    try:
        # 전체 이미지 개수 계산
        total_images = len([f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        results = find_similar_images_by_clip(
            query.text,
            DATA_DIR,
            TEMP_DIR,
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

@router.post("/update-features")
async def update_features():
    try:
        save_clip_image_features(DATA_DIR, TEMP_DIR)
        return {"message": "이미지 특징 벡터가 업데이트되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search", response_model=SearchResponse)
async def search_images_get(
    text: str,
    similarity_threshold: float = SIMILARITY_THRESHOLD
):
    try:
        total_images = len([f for f in os.listdir(DATA_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        results = find_similar_images_by_clip(
            text,
            DATA_DIR,
            TEMP_DIR,
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