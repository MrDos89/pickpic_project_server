import os

# 프로젝트 루트 기준 경로
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")

# 유사도 임계값 (기본값)
SIMILARITY_THRESHOLD = 0.0 

# FastAPI 문서 정보
API_TITLE = "Text & Image Pose Detection API"
API_DESCRIPTION = "텍스트/이미지 기반 포즈 인식 및 검색 API"
API_VERSION = "1.0.0" 