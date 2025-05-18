import os
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from models.hands_detection_model.hands_detection import is_hands, is_heart, is_thumbs

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
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    detected = []

    # ✅ 만세 포즈
    if lw.y < nose.y and rw.y < nose.y:
        detected.append("만세")

    # ✅ 점프 포즈 기준
    jump = (
        (la.y < lh.y - 0.05 and ra.y < rh.y - 0.05) or
        (la.y < lk.y and ra.y < rk.y) or
        (abs(la.y - lk.y) < 0.08 and abs(ra.y - rk.y) < 0.08)
    )

    # ✅ 다리 길이 계산
    left_upper_leg = abs(lh.y - lk.y)
    left_lower_leg = abs(lk.y - la.y)
    right_upper_leg = abs(rh.y - rk.y)
    right_lower_leg = abs(rk.y - ra.y)

    upper_leg_avg = (left_upper_leg + right_upper_leg) / 2
    lower_leg_avg = (left_lower_leg + right_lower_leg) / 2

    # ✅ 다리 길이 조건 완화
    upper_leg_ok = 0.09 < upper_leg_avg < 0.26
    lower_leg_ok = 0.15 < lower_leg_avg < 0.32
    left_leg_ok = 0.07 < left_upper_leg < 0.3 and 0.12 < left_lower_leg < 0.35
    right_leg_ok = 0.07 < right_upper_leg < 0.3 and 0.12 < right_lower_leg < 0.35

    # ✅ 발 간 거리 완화
    foot_distance_x = abs(la.x - ra.x)
    foot_gap_ok = foot_distance_x < 0.6

    # ✅ 전체 좌표 범위
    x_vals = [lw.x, rw.x, ls.x, rs.x, lh.x, rh.x, lk.x, rk.x, la.x, ra.x]
    y_vals = [lw.y, rw.y, ls.y, rs.y, lh.y, rh.y, lk.y, rk.y, la.y, ra.y]
    x_range = max(x_vals) - min(x_vals)
    y_range = max(y_vals) - min(y_vals)

    # ✅ 누운 포즈: 세로로 짧고 가로로 넓음
    lying = y_range < 0.18 and x_range > 0.45

    # 서있음 (더 완화)
    stand_y_ok = y_range > 0.13
    stand_x_ok = x_range < 0.7
    foot_gap_ok = foot_distance_x < 0.85
    left_side_straight = abs(ls.x - la.x) < 0.13
    right_side_straight = abs(rs.x - ra.x) < 0.13
    straight_ok = left_side_straight or right_side_straight
    stand = stand_y_ok and stand_x_ok and foot_gap_ok and straight_ok

    # 앉음 조건 개선
    left_sit = (lh.y > lk.y) and (lk.y < la.y) and (abs(lh.y - la.y) < 0.35)
    right_sit = (rh.y > rk.y) and (rk.y < ra.y) and (abs(rh.y - ra.y) < 0.35)
    sit = left_sit or right_sit or (x_range > 0.35 and y_range < 0.32)

    # ✅ 포즈 분류 순서 (lying 먼저)
    if lying:
        detected.append("누워있음")
    elif jump:
        detected.append("점프")
    elif sit:
        detected.append("앉음")
    elif stand:
        detected.append("서있음")
    else:
        detected.append("일반 포즈 (미분류)")

    return ", ".join(detected) + "입니다!"



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

class PoseDetector:
    def __init__(self):
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        # YOLO 모델 초기화 (한 번만 로드)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n-pose.pt")
        self.yolo = YOLO(model_path)

    def detect_v_pose(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return "이미지 없음", None
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            if not results.multi_hand_landmarks:
                return "not found", image
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                if is_hands(lm):
                    return "브이", image
        return "not found", image

    def detect_heart(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return "이미지 없음", None
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            if not results.multi_hand_landmarks:
                return "not found", image
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                if is_heart(lm, image.shape[0], image.shape[1]):
                    return "하트", image
        return "not found", image

    def detect_thumbs(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return "이미지 없음", None
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            if not results.multi_hand_landmarks:
                return "not found", image
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                if is_thumbs(lm):
                    return "최고", image
        return "not found", image


    def detect_body_pose(self, image_path):

        img = cv2.imread(image_path)
        if img is None:
            return "이미지를 불러올 수 없습니다.", np.zeros((256, 256, 3), dtype=np.uint8)

        results = self.yolo.predict(img, stream=False, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []

        if len(boxes) == 0:
            return "사람이 감지되지 않았습니다.", img

        result_texts = []
        img_result = img.copy()
        with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                person_crop = img[y1:y2, x1:x2]
                if person_crop.size == 0:
                    result_texts.append(f"사람 {idx+1}: 박스 크기 오류")
                    continue
                person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                results_mp = pose.process(person_rgb)
                if results_mp.pose_landmarks:
                    pose_result = classify_pose_mediapipe(results_mp.pose_landmarks.landmark)
                    result_texts.append(f"사람 {idx+1}: {pose_result}")
                    # 시각화
                    mp_drawing.draw_landmarks(person_crop, results_mp.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    img_result[y1:y2, x1:x2] = person_crop
                else:
                    result_texts.append(f"사람 {idx+1}: 포즈 인식 실패")
        return " | ".join(result_texts), img_result


if __name__ == "__main__":
    main()
