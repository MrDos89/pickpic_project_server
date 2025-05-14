# YOLOv8 모델(사람 여러명 포즈 인식 가능)
import os
import cv2
import numpy as np
from ultralytics import YOLO

def main(resources_path):
    # 모델 파일 절대경로
    model_path = os.path.join(resources_path, "models", "yolov8n-pose.pt")
    yolo_model = YOLO(model_path)

    # image 폴더 절대경로
    image_folder = resources_path

    def classify_pose_yolo(keypoints):
        def get_point(idx):
            return keypoints[idx][0], keypoints[idx][1]

        lw_x, lw_y = get_point(9)
        rw_x, rw_y = get_point(10)
        ls_x, ls_y = get_point(5)
        rs_x, rs_y = get_point(6)
        lh_x, lh_y = get_point(11)
        rh_x, rh_y = get_point(12)
        lk_x, lk_y = get_point(13)
        rk_x, rk_y = get_point(14)
        la_x, la_y = get_point(15)
        ra_x, ra_y = get_point(16)

        def distance(p1, p2):
            return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

        detected_poses = []

        if lw_y < ls_y and rw_y < rs_y:
            detected_poses.append("만세 포즈")

        if (lw_y < ls_y and rw_y >= rs_y) or (rw_y < rs_y and lw_y >= ls_y):
            detected_poses.append("손 흔들기 포즈")

        shoulder_width = abs(rs_x - ls_x)
        if abs(lw_x - ls_x) > shoulder_width * 1.5 and abs(rw_x - rs_x) > shoulder_width * 1.5:
            detected_poses.append("팔 벌리기 포즈")

        hip_center_y = (lh_y + rh_y) / 2
        if abs(lw_y - hip_center_y) < 0.05 and abs(rw_y - hip_center_y) < 0.05:
            detected_poses.append("두 손 허리 포즈")

        if distance((lw_x, lw_y), (rw_x, rw_y)) < 0.05:
            detected_poses.append("손 모으기 포즈")

        jump_condition = (
            (la_y < lh_y - 0.05 and ra_y < rh_y - 0.05) or
            (la_y < lk_y and ra_y < rk_y) or
            (abs(la_y - lk_y) < 0.08 and abs(ra_y - rk_y) < 0.08)
        )

        if jump_condition:
            detected_poses.append("점프샷 포즈")

        left_standing = abs(lh_y - lk_y) < 0.1 and abs(lk_y - la_y) < 0.1
        right_standing = abs(rh_y - rk_y) < 0.1 and abs(rk_y - ra_y) < 0.1

        if not jump_condition and (left_standing or right_standing):
            detected_poses.append("서있는 포즈")

        if lh_y > lk_y and rh_y > rk_y:
            detected_poses.append("앉은 포즈")

        if abs(lk_y - rk_y) > 0.2:
            detected_poses.append("런지 / 스트레칭 포즈")

        y_vals = [lw_y, rw_y, ls_y, rs_y, lh_y, rh_y, lk_y, rk_y, la_y, ra_y]
        if max(y_vals) - min(y_vals) < 0.1:
            detected_poses.append("누워있는 포즈")

        return ", ".join(detected_poses) + "입니다!" if detected_poses else "일반 포즈 (미분류)입니다."

    # 이미지 폴더 읽기
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for file in image_files:
        path = os.path.join(image_folder, file)
        img = cv2.imread(path)

        if img is None:
            print(f"[경고] {file} 이미지 불러오기 실패")
            continue

        results = yolo_model.predict(img, stream=False, verbose=False)
        if results[0].keypoints is None:
            print(f"\n[파일명: {file}] 사람 없음")
            continue

        print(f"\n[파일명: {file}]")
        for i, keypoints in enumerate(results[0].keypoints.xy):
            result_text = classify_pose_yolo(keypoints)
            print(f"사람 {i+1}: {result_text}")

        result_img = results[0].plot()
        cv2.imshow("YOLO Pose Detection", result_img)

        key = cv2.waitKey(1000)
        if key == 27 or key == ord('0'):
            break

    cv2.destroyAllWindows()
