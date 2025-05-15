import os
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def classify_pose_mediapipe(landmarks):
    lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]

    def distance(a, b):
        return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5

    detected = []
    if lw.y < nose.y and rw.y < nose.y:
        detected.append("만세 포즈")
    # if (lw.y < ls.y and rw.y >= rs.y) or (rw.y < rs.y and lw.y >= ls.y):
    #     detected.append("손 흔들기 포즈")
    # shoulder_width = abs(rs.x - ls.x)
    # if abs(lw.x - ls.x) > shoulder_width * 1.5 and abs(rw.x - rs.x) > shoulder_width * 1.5:
    #     detected.append("팔 벌리기 포즈")
    # hip_center_y = (lh.y + rh.y) / 2
    # if abs(lw.y - hip_center_y) < 0.05 and abs(rw.y - hip_center_y) < 0.05:
    #     detected.append("두 손 허리 포즈")
    # if distance(lw, rw) < 0.05:
    #     detected.append("손 모으기 포즈")

    jump = (la.y < lh.y - 0.05 and ra.y < rh.y - 0.05) or (la.y < lk.y and ra.y < rk.y) or (abs(la.y - lk.y) < 0.08 and abs(ra.y - rk.y) < 0.08)
    stand = abs(lh.y - lk.y) < 0.1 and abs(lk.y - la.y) < 0.1 or abs(rh.y - rk.y) < 0.1 and abs(rk.y - ra.y) < 0.1
    sit = lh.y > lk.y and rh.y > rk.y
    # lunge = abs(lk.y - rk.y) > 0.2
    lying = max([lw.y, rw.y, ls.y, rs.y, lh.y, rh.y, lk.y, rk.y, la.y, ra.y]) - min([lw.y, rw.y, ls.y, rs.y, lh.y, rh.y, lk.y, rk.y, la.y, ra.y]) < 0.1

    if jump:
        detected.append("점프샷 포즈")
    elif stand:
        detected.append("서있는 포즈")
    elif sit:
        detected.append("앉은 포즈")
    # elif lunge:
    #     detected.append("런지 / 스트레칭 포즈")
    elif lying:
        detected.append("누워있는 포즈")

    return ", ".join(detected) + "입니다!" if detected else "일반 포즈 (미분류)입니다."

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    image_dir = os.path.join(base_dir, "data")
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n-pose.pt")
    yolo = YOLO(model_path)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for file in image_files:
        path = os.path.join(image_dir, file)
        img = cv2.imread(path)
        print(f"\n[파일명: {file}]")

        results = yolo.predict(img, stream=False, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

        if len(boxes) == 0:
            print("[YOLOv8] 사람 없음")
            continue

        print(f"[YOLOv8] 사람 {len(boxes)}명 감지")

        with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

                person_crop = img[y1:y2, x1:x2]
                if person_crop.size == 0:
                    print(f" → 사람 {idx+1}: 박스 크기 오류 → 스킵")
                    continue

                person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                results_mp = pose.process(person_rgb)

                if results_mp.pose_landmarks:
                    pose_result = classify_pose_mediapipe(results_mp.pose_landmarks.landmark)
                    print(f" → 사람 {idx+1}: {pose_result}")
                else:
                    print(f" → 사람 {idx+1}: 포즈 인식 실패")

        img_yolo = results[0].plot()
        cv2.imshow("YOLO + Mediapipe Ensemble", img_yolo)

        key = cv2.waitKey(1000)
        if key == 27 or key == ord('0'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
