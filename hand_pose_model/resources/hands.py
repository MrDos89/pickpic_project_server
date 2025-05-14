import cv2
import mediapipe as mp
import math

# MediaPipe Hands 모델 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(point1, point2):
    """두 점 사이의 거리를 계산합니다."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# 이미지 파일 경로 지정
image = cv2.imread('..\\data\\vPose6.jpg')

# BGR 이미지를 RGB로 변환
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Hands 모델로 랜드마크 추출
results = hands.process(image_rgb)

# V 포즈 감지 여부 변수
v_pose_detected = False

# 손 랜드마크가 있을 때
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # 손가락 랜드마크 추출 (끝과 중간 마디)
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        # 좌표 추출 (이미지 크기에 맞게 변환)
        h, w, c = image.shape
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
        thumb_tip_x, thumb_tip_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
        index_finger_tip_x_raw, index_finger_tip_y_raw = index_finger_tip.x, index_finger_tip.y
        index_finger_pip_x, index_finger_pip_y = int(index_finger_pip.x * w), int(index_finger_pip.y * h)
        middle_finger_tip_x, middle_finger_tip_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)

        # 사용할 검지 손가락 좌표 (끝이 감지 안 되면 중간 마디 사용)
        index_finger_x, index_finger_y = index_finger_tip_x_raw * w, index_finger_tip_y_raw * h
        if index_finger_x == 0 and index_finger_y == 0: # 끝이 감지 안 된 경우 (좌표가 0일 가능성)
            index_finger_x, index_finger_y = index_finger_pip_x, index_finger_pip_y

        # V 포즈인지 확인 (손목, 엄지, 검지의 각도를 계산)
        def calculate_angle(p1, p2, p3):
            # 각도 계산 (세 점 p1, p2, p3에서 p2를 기준으로 각도 구하기)
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
            return abs(angle)

        # 엄지와 검지 사이의 각도 계산
        angle_thumb_index = calculate_angle((wrist_x, wrist_y), (thumb_tip_x, thumb_tip_y), (int(index_finger_x), int(index_finger_y)))

        # 엄지와 검지 끝 (또는 중간 마디) 사이의 거리 계산
        distance_thumb_index = calculate_distance((thumb_tip_x, thumb_tip_y), (int(index_finger_x), int(index_finger_y)))
        distance_threshold = w * 0.08 # 거리 임계값 약간 낮춤

        # 검지와 중지 끝의 Y 좌표 비교 (아래 방향 브이 감지)
        index_below_middle = int(index_finger_y) > middle_finger_tip_y

        # 브이 포즈 여부 판단 (임계값 조정 및 손가락 잘림 고려)
        if angle_thumb_index > 25 and distance_thumb_index > distance_threshold: # 각도 임계값 더 낮춤
            # 일반적인 브이 포즈 (손가락 위쪽)
            if int(index_finger_y) < wrist_y and thumb_tip_y < wrist_y:
                v_pose_detected = True
            # 아래 방향 브이 포즈 (검지가 중지보다 아래)
            elif index_below_middle and thumb_tip_y > wrist_y and middle_finger_tip_y > wrist_y:
                v_pose_detected = True
            elif not index_below_middle and thumb_tip_y > wrist_y and int(index_finger_y) > wrist_y and angle_thumb_index > 50: # 아래 방향이지만 각도가 큰 경우
                v_pose_detected = True

        if v_pose_detected:
            break  # 하나의 손에서 브이 포즈가 감지되면 루프를 종료

# 결과 출력
if v_pose_detected:
    print('브이')
else:
    print('아님')

# 결과 이미지 보기 (랜드마크 추가)
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

cv2.imshow('Hand Pose Analysis', image)
cv2.waitKey(0)
cv2.destroyAllWindows()