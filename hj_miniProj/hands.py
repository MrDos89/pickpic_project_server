# import cv2
# import mediapipe as mp

# # MediaPipe Hands 모델 설정
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # 이미지 읽기
# image = cv2.imread('v2.png')

# # BGR 이미지를 RGB로 변환
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Hands 모델로 랜드마크 추출
# results = hands.process(image_rgb)

# # 손 랜드마크가 있을 때
# if results.multi_hand_landmarks:
#     for hand_landmarks in results.multi_hand_landmarks:
#         for landmark in hand_landmarks.landmark:
#             h, w, c = image.shape
#             cx, cy = int(landmark.x * w), int(landmark.y * h)
#             cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)  # 랜드마크 그리기

# # 결과 이미지 보기
# cv2.imshow('Hand V Pose Analysis', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# --------------------------------------

# import cv2
# import mediapipe as mp
# import math

# # MediaPipe Hands 모델 설정
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # 이미지 파일 경로 지정
# # image_path = 'path_to_your_image/hand_v_pose.jpg'

# # 이미지 읽기
# image = cv2.imread('v2.png')

# # BGR 이미지를 RGB로 변환
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Hands 모델로 랜드마크 추출
# results = hands.process(image_rgb)

# # 손 랜드마크가 있을 때
# if results.multi_hand_landmarks:
#     for hand_landmarks in results.multi_hand_landmarks:
#         # 손목, 엄지, 검지 랜드마크 추출
#         wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
#         thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#         index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

#         # 손목과 손끝 좌표 추출 (이미지 크기에 맞게 변환)
#         h, w, c = image.shape
#         wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
#         thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
#         index_finger_x, index_finger_y = int(index_finger.x * w), int(index_finger.y * h)

#         # V 포즈인지 확인 (손목, 엄지, 검지의 각도를 계산)
#         def calculate_angle(p1, p2, p3):
#             # 각도 계산 (세 점 p1, p2, p3에서 p2를 기준으로 각도 구하기)
#             x1, y1 = p1
#             x2, y2 = p2
#             x3, y3 = p3
#             angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
#             return abs(angle)

#         # 엄지와 검지 사이의 각도 계산
#         angle = calculate_angle((wrist_x, wrist_y), (thumb_x, thumb_y), (index_finger_x, index_finger_y))

#         # 브이 포즈 여부 판단 (예: 각도가 45도 이상이면 브이 포즈로 판단)
#         if angle > 45:
#             # cv2.putText(image, "V Pose Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             print('브이')
#         else:
#             # cv2.putText(image, "Not V Pose", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             print('None')

#         # 랜드마크 표시
#         for landmark in hand_landmarks.landmark:
#             cx, cy = int(landmark.x * w), int(landmark.y * h)
#             cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

# # 결과 이미지 보기
# cv2.imshow('Hand Pose Analysis', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# -------------------------------

import cv2
import mediapipe as mp
import math

# MediaPipe Hands 모델 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 이미지 파일 경로 지정
image = cv2.imread('vv2.jpg')

# BGR 이미지를 RGB로 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Hands 모델로 랜드마크 추출
results = hands.process(image_rgb)

# V 포즈 감지 여부 변수
v_pose_detected = False

# 손 랜드마크가 있을 때
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # 손목, 엄지, 검지 랜드마크 추출
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # 손목과 손끝 좌표 추출 (이미지 크기에 맞게 변환)
        h, w, c = image.shape
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
        thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
        index_finger_x, index_finger_y = int(index_finger.x * w), int(index_finger.y * h)

        # V 포즈인지 확인 (손목, 엄지, 검지의 각도를 계산)
        def calculate_angle(p1, p2, p3):
            # 각도 계산 (세 점 p1, p2, p3에서 p2를 기준으로 각도 구하기)
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
            return abs(angle)

        # 엄지와 검지 사이의 각도 계산
        angle = calculate_angle((wrist_x, wrist_y), (thumb_x, thumb_y), (index_finger_x, index_finger_y))

        # 브이 포즈 여부 판단 (예: 각도가 45도 이상이면 브이 포즈로 판단)
        if angle > 45:
            v_pose_detected = True
            break  # 하나의 손에서 브이 포즈가 감지되면 루프를 종료

# 결과 출력
if v_pose_detected:
    print('브이')
else:
    print('None')

# 결과 이미지 보기
cv2.imshow('Hand Pose Analysis', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
