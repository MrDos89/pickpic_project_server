# import cv2
# import mediapipe as mp
# import math
# import os

# # MediaPipe Hands 모델 설정
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # 이미지 파일 경로 지정
# image_path = 'v2.png'  # 예시로 이미지 경로 수정

# # 이미지 경로 확인
# if not os.path.exists(image_path):
#     print(f"이미지 파일이 존재하지 않습니다: {image_path}")
# else:
#     print(f"이미지 파일을 읽고 있습니다: {image_path}")

# # 이미지 읽기
# image = cv2.imread(image_path)

# # 이미지가 제대로 로드되었는지 확인
# if image is None:
#     print("이미지를 열 수 없습니다. 파일 경로를 확인하세요.")
# else:
#     print("이미지 로드 성공")

# # BGR 이미지를 RGB로 변환
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Hands 모델로 랜드마크 추출
# results = hands.process(image_rgb)

# # V 포즈 감지 여부 변수
# v_pose_detected = False

# # 손 랜드마크가 있을 때
# if results.multi_hand_landmarks:
#     for hand_landmarks in results.multi_hand_landmarks:
#         # 검지와 중지의 첫 번째 관절 (proximal)과 끝 (tip) 랜드마크 추출
#         index_finger_proximal = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
#         index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
#         middle_finger_proximal = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
#         middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
#         # 이미지 크기 추출
#         h, w, c = image.shape
        
#         # 검지와 중지의 관절 위치 변환
#         index_x1, index_y1 = int(index_finger_proximal.x * w), int(index_finger_proximal.y * h)
#         index_x2, index_y2 = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

#         middle_x1, middle_y1 = int(middle_finger_proximal.x * w), int(middle_finger_proximal.y * h)
#         middle_x2, middle_y2 = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)

#         # 각도 계산 함수
#         def calculate_angle(p1, p2, p3):
#             # 각도 계산 (세 점 p1, p2, p3에서 p2를 기준으로 각도 구하기)
#             x1, y1 = p1
#             x2, y2 = p2
#             x3, y3 = p3
#             angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
#             return abs(angle)

#         # 검지와 중지 사이의 각도 계산
#         angle = calculate_angle((index_x1, index_y1), (index_x2, index_y2), (middle_x1, middle_y1))

#         # 20도 이상이면 V 포즈로 판단
#         if angle > 20:
#             v_pose_detected = True
#             print("브이 포즈 감지됨")
#         else:
#             print("브이 포즈 아님")

#         # 랜드마크 표시
#         for landmark in hand_landmarks.landmark:
#             cx, cy = int(landmark.x * w), int(landmark.y * h)
#             cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

# # 결과 이미지 보기
# cv2.imshow('Hand Pose Analysis', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp

def is_reverse_v(hand_landmarks):
    if hand_landmarks and len(hand_landmarks.landmark) >= 21:
        index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        ring_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
        pinky_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
        wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST] # 손목 랜드마크 추가

        v_finger_avg_y = (index_finger_tip.y + middle_finger_tip.y) / 2
        other_finger_avg_y = (thumb_tip.y + ring_finger_tip.y + pinky_finger_tip.y) / 3

        # 거꾸로 된 브이 동작일 가능성이 높은 조건 (임계값 조정 필요)
        if (v_finger_avg_y > other_finger_avg_y + 0.03 and  # 임계값 조정
            index_finger_tip.y > wrist.y and
            middle_finger_tip.y > wrist.y):
            return True
        else:
            return False
    return False

# 나머지 코드는 이전과 동일하게 유지
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
image_path = "v3.png"  # 실제 이미지 경로로 변경
image = cv2.imread(image_path)

if image is not None:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            if is_reverse_v(hand_landmarks):
                print("거꾸로 된 브이 감지!")
            else:
                print("일반적인 브이 또는 다른 동작 감지.")
        cv2.imshow("Hand Landmarks", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("손을 감지하지 못했습니다.")
mp_hands.close()