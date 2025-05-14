import cv2
import mediapipe as mp

# mediapipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 이미지 불러오기
image_path = 'C:/Users/201-09/miniProject/handsup.jpg'
image = cv2.imread('handsup.jpg')

# mediapipe로 자세 분석
with mp_pose.Pose(static_image_mode=True) as pose:
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        # 랜드마크 그리기 (시각화용)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 랜드마크 가져오기
        landmarks = results.pose_landmarks.landmark
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # 만세 판별: 양손이 양쪽 어깨보다 위 (y값 작음 → 이미지 좌표계에서 위쪽)
        if (left_wrist.y < left_shoulder.y) and (right_wrist.y < right_shoulder.y):
            print("만세 포즈입니다!")
        else:
            print("만세 포즈가 아닙니다.")

        # 결과 이미지 보여주기
        cv2.imshow("Pose Landmarks", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("사람을 인식하지 못했습니다.")
