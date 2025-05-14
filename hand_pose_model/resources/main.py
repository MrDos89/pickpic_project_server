import sys
import os

def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        # 현재 파일 기준으로 상위 두 폴더 → data 찾기
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

data_path = get_base_path()   # 반드시 이렇게 변경

# models 폴더에 있는 각 모듈들 불러오기
import fist
import fy
import hands
import heart
import military
import okay
import thumbs

def main():
    # print("===================================")
    # print(" 포즈 검출 프로그램")
    # print("===================================")
    # print("1. 주먹")
    # print("2. 브이")
    # print("3. 하트")
    # print("4. 경례")
    # print("5. 오케이")
    # print("6. 굿")
    # print("7. 우~")
    # print("0. 종료")
    # print("-----------------------------------")

    # while True:
    #     choice = input("모델을 선택하세요: ").strip()
    #     if choice == "1":
    #         fist.main(data_path)  # fist.py 실행
    #     elif choice == "2":
    #         fy.main(data_path)  # fy.py 실행
    #     elif choice == "3":
    #         heart.main(data_path)  # heart.py 실행
    #     elif choice == "4":
    #         military.main(data_path)  # military.py 실행
    #     elif choice == "5":
    #         okay.main(data_path)  # okay.py 실행
    #     elif choice == "6":
    #         thumbs.main(data_path)  # thumbs.py 실행
    #     elif choice == "7":
    #         thumbsDown.main(data_path)  # thumbsDown.py 실행
    #     elif choice == "0":
    #         break
    #     else:
    #         print("잘못 입력했습니다. 다시 입력하세요.")
    pass
if __name__ == "__main__":
    main()
