import cv2
import mediapipe as mp
import math

# MediaPipe Hands 모델 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 이미지 파일 경로 지정
image_path = 'vv.jpg'  # 여기에 테스트할 이미지 경로 입력

# 이미지 읽기
image = cv2.imread(image_path)

# 이미지가 정상적으로 로드되었는지 확인
if image is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
    exit()

# BGR 이미지를 RGB로 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Hands 모델로 랜드마크 추출
results = hands.process(image_rgb)

# 손 랜드마크가 있을 때
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # 손목, 엄지 손가락의 랜드마크 추출
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]  # 엄지 첫 번째 관절
        thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]  # 엄지 두 번째 관절

        # **검지, 중지, 약지, 새끼손가락 랜드마크 추가**
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_TIP]
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_PIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_TIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_PIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_TIP]
        ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_PIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

        # 엄지 손가락이 위로 향하는지 확인
        if thumb_tip.y < thumb_ip.y and thumb_tip.y < wrist.y:
            # 나머지 손가락이 접혀 있는지 확인
            fingers_folded = (
                index_tip.y > index_pip.y and  # 검지 접힘
                middle_tip.y > middle_pip.y and  # 중지 접힘
                ring_tip.y > ring_pip.y and  # 약지 접힘
                pinky_tip.y > pinky_pip.y  # 새끼손가락 접힘
            )

            if fingers_folded:  # 모든 손가락이 접혀 있으면
                print("✅ 엄지 척 감지됨!")
            else:
                print("❌ 엄지 척이 아닙니다. 다른 손가락이 펴져 있음.")
        else:
            print("❌ 엄지 척이 아닙니다.")

        # 랜드마크 표시
        h, w, c = image.shape
        for landmark in hand_landmarks.landmark:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

# 결과 이미지 보기
cv2.imshow('Thumbs Up Analysis', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# import math

# # MediaPipe Hands 모델 설정
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # 이미지 파일 경로 지정
# image_path = 'thumbs.jpg'  # 여기에 테스트할 이미지 경로 입력

# # 이미지 읽기
# image = cv2.imread(image_path)

# # 이미지가 정상적으로 로드되었는지 확인
# if image is None:
#     print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
#     exit()

# # BGR 이미지를 RGB로 변환
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Hands 모델로 랜드마크 추출
# results = hands.process(image_rgb)

# # 손 랜드마크가 있을 때
# if results.multi_hand_landmarks:
#     for hand_landmarks in results.multi_hand_landmarks:
#         # 손목, 엄지 손가락의 랜드마크 추출
#         thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#         thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]  # 엄지 첫 번째 관절
#         thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]  # 엄지 두 번째 관절

#         # 이미지 크기에서 좌표 변환 (이미지 크기 맞추기)
#         h, w, c = image.shape
#         thumb_tip_x, thumb_tip_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
#         thumb_ip_x, thumb_ip_y = int(thumb_ip.x * w), int(thumb_ip.y * h)
#         thumb_mcp_x, thumb_mcp_y = int(thumb_mcp.x * w), int(thumb_mcp.y * h)

#         # 엄지 손가락 각도 계산 함수
#         def calculate_angle(p1, p2, p3):
#             x1, y1 = p1
#             x2, y2 = p2
#             x3, y3 = p3
#             angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
#             return abs(angle)

#         # 엄지 손가락의 두 번째 관절(THUMB_MCP)과 첫 번째 관절(THUMB_IP) 사이의 각도 계산
#         angle1 = calculate_angle((thumb_mcp_x, thumb_mcp_y), (thumb_ip_x, thumb_ip_y), (thumb_tip_x, thumb_tip_y))

#         # 엄지 손가락이 위로 올라갔는지 판단 (예: 각도가 70도 이상이면 엄지가 위로 향한 것으로 판단)
#         if angle1 > 70:
#             print("✅ 최고! 따봉!")
#         else:
#             print("❌")

# else:
#     print("손을 감지할 수 없습니다.")

# # 결과 이미지 보기 (선택 사항: 이미지 표시 없이 실행하려면 주석 처리)
# cv2.imshow('Thumbs Up Analysis', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()