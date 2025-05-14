import uvicorn
from fastapi import FastAPI
from .api.image_search import router as image_search_router
from .utils.config import API_TITLE, API_DESCRIPTION, API_VERSION

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# 라우터 등록
app.include_router(image_search_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True) 