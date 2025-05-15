import cv2
import os
import sys
from hand_model import classify_hand_pose, mp_drawing, mp_hands

def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

def main():
    base_path = get_base_path()

    # data 폴더 내 이미지 파일 하나만 처리한다고 가정 (여러 개면 리스트로 반복 가능)
    image_files = [f for f in os.listdir(base_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    if not image_files:
        print(f"❌ {base_path} 폴더에 이미지 파일이 없습니다.")
        return

    for image_file in image_files:
        image_path = os.path.join(base_path, image_file)
        print(f"처리중인 이미지: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            print("❌ 이미지를 불러올 수 없습니다.")
            continue

        poses = classify_hand_pose(image)

        h, w, _ = image.shape

        if poses:
            for i, pose in enumerate(poses.keys()):
                cv2.putText(image, f"{pose} Pose", (30, 60 + i*40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            print(f"✅ 인식된 포즈: {', '.join(poses.keys())}")
        else:
            cv2.putText(image, "No recognized pose", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            print("❌ 인식된 포즈가 없습니다.")

        cv2.imshow('Pose Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
