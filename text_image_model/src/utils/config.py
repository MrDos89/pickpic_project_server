import os

# 프로젝트 루트 디렉토리 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")

# 디렉토리가 없으면 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# 이미지 검색 설정
SIMILARITY_THRESHOLD = 0.066
TOP_N_RESULTS = 100

# FastAPI 설정
API_TITLE = "이미지 검색 API"
API_DESCRIPTION = "CLIP 모델을 사용한 이미지 검색 API"
API_VERSION = "1.0.0" 