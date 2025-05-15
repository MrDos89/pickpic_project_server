<<<<<<< Updated upstream
=======
<<<<<<< HEAD
import uvicorn
from fastapi import FastAPI
from src.api.v1.image_search import router as image_search_router
from src.api.v1.pose_detection import router as pose_detection_router
from src.core.config import API_TITLE, API_DESCRIPTION, API_VERSION
import sys
import os
from src.models import model
=======
>>>>>>> Stashed changes
try:
    import uvicorn
    from fastapi import FastAPI
    from api.v1.image_search import router as image_search_router
    from api.v1.pose_detection import router as pose_detection_router
    from core.config import API_TITLE, API_DESCRIPTION, API_VERSION
    import sys
    import os
    from models import model1, model2
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    import uvicorn
    from fastapi import FastAPI
    from api.v1.image_search import router as image_search_router
    from api.v1.pose_detection import router as pose_detection_router
    from core.config import API_TITLE, API_DESCRIPTION, API_VERSION
    import os
    from models import model1, model2
<<<<<<< Updated upstream
=======
>>>>>>> 631694bf0fb945e23ae8ce9fcd505b673c127b7e
>>>>>>> Stashed changes

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# 라우터 등록
app.include_router(image_search_router, prefix="/api/v1")
app.include_router(pose_detection_router, prefix="/api/v1/pose")

# CLI 기반 모델 선택 기능 추가 (기존 main.py 통합)
def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        # main.py 기준으로 data 폴더 경로 설정
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

data_path = get_base_path()

models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if models_path not in sys.path:
    sys.path.insert(0, models_path)

def cli_main():
    print("===================================")
    print(" 포즈 검출 프로그램 (Ensemble Model)")
    print("===================================")
    print("0. 종료")
    print("-----------------------------------")
    while True:
        choice = input("엔트리포인트를 실행하려면 0을 입력하세요 (종료: 0): ").strip()
        if choice == "0":
            break
        else:
            print("Ensemble 모델 실행!")
            model.main(data_path)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        cli_main()
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 