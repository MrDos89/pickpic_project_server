from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from utils.image_matcher import find_best_image
import os

app = FastAPI()

USER_PHOTOS_DIR = os.path.join(os.path.dirname(__file__), 'user_photos')

class FindImageRequest(BaseModel):
    text: str
    method: str = 'clip'  # 'clip' 또는 'keyword' 중 선택
    return_type: str = 'base64'  # 'base64' 또는 'url' 중 선택

@app.post("/find_image")
async def find_image(req: FindImageRequest):
    image_result = find_best_image(req.text, USER_PHOTOS_DIR, method=req.method)
    if image_result is None:
        raise HTTPException(status_code=404, detail="관련 이미지를 찾을 수 없습니다.")
    if req.return_type == 'base64':
        return {"image": image_result['base64']}
    else:
        return {"image_url": f"/user_photos/{image_result['filename']}"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail}) 