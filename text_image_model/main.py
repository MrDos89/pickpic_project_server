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
    print(" 포즈 검출 프로그램")
    print("===================================")
    print("1. Mediapipe Pose Detection")
    print("2. YOLOv8 Pose Detection")
    print("0. 종료")
    print("-----------------------------------")

    while True:
        choice = input("모델을 선택하세요 (1/2/0): ").strip()
        if choice == "1":
            model1.main(data_path)
        elif choice == "2":
            model2.main(data_path)
        elif choice == "0":
            break
        else:
            print("잘못 입력했습니다. 다시 입력하세요.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        cli_main()
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 