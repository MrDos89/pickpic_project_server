import sys
import os

def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

data_path = get_base_path()

models_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
if models_path not in sys.path:
    sys.path.insert(0, models_path)

import model1
import model2

def main():
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
    main()
