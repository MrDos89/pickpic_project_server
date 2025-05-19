import mediapipe as mp
import math
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def is_heart(lm, image_height=None, image_width=None, threshold=0.2, required_conditions=2):
    
    def distance(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    # 먼저 V 포즈인지 확인 - V 포즈면 바로 False 반환
    if image_height is not None and image_width is not None:
        if is_v(lm, image_height, image_width):
            return False
    
    # 손가락 하트 조건
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = lm[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = lm[mp_hands.HandLandmark.PINKY_TIP]

    thumb_index_dist = distance(thumb_tip, index_tip)
    other_folded = (
        middle_tip.y > lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
        ring_tip.y > lm[mp_hands.HandLandmark.RING_FINGER_PIP].y and
        pinky_tip.y > lm[mp_hands.HandLandmark.PINKY_PIP].y
    )
    if thumb_index_dist < 0.05 and other_folded:
        return True

    # 볼하트 조건 (image_height, image_width 필요)
    if image_height is not None and image_width is not None:
        index_base = lm[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        wrist = lm[mp_hands.HandLandmark.WRIST]

        index_tip_y = index_tip.y * image_height
        index_base_y = index_base.y * image_height
        wrist_y = wrist.y * image_height

        satisfied = 0

        # 조건 1: 손가락 간 거리 가까움
        distances = [
            distance(index_tip, middle_tip),
            distance(index_tip, ring_tip),
            distance(index_tip, pinky_tip),
            distance(middle_tip, ring_tip),
            distance(middle_tip, pinky_tip),
            distance(ring_tip, pinky_tip)
        ]
        close_count = sum(1 for dist in distances if dist < threshold)
        if close_count >= 2:
            satisfied += 1

        # 조건 2: 손이 너무 아래 있지 않음
        if index_tip_y < image_height * 0.85 and wrist_y < image_height * 0.95:
            satisfied += 1

        # 조건 3: 손끝이 손등보다 아래(구부려짐)
        if index_tip_y < index_base_y + 10: # 픽셀 단위로 조정될 수 있음
            satisfied += 1

        # 조건 4: 엄지와 검지 간 x좌표 가까움
        index_thumb_x_diff = abs(index_tip.x - thumb_tip.x)
        if index_thumb_x_diff < 0.25: # 정규화된 좌표 기준
            satisfied += 1

        if satisfied >= required_conditions:
            return True

    return False

def is_v(lm, image_height, image_width):
    """V 포즈 감지 함수"""
    # 랜드마크 추출
    wrist = lm[mp_hands.HandLandmark.WRIST]
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = lm[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = lm[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = lm[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = lm[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = lm[mp_hands.HandLandmark.PINKY_PIP]
    
    # 좌표 변환
    w, h = image_width, image_height
    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
    thumb_tip_x, thumb_tip_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
    index_tip_x, index_tip_y = int(index_tip.x * w), int(index_tip.y * h)
    index_pip_x, index_pip_y = int(index_pip.x * w), int(index_pip.y * h)
    middle_tip_x, middle_tip_y = int(middle_tip.x * w), int(middle_tip.y * h)
    middle_pip_x, middle_pip_y = int(middle_pip.x * w), int(middle_pip.y * h)
    ring_tip_x, ring_tip_y = int(ring_tip.x * w), int(ring_tip.y * h)
    pinky_tip_x, pinky_tip_y = int(pinky_tip.x * w), int(pinky_tip.y * h)
    
    # 각도 계산 함수
    def calculate_angle(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        return abs(angle)
    
    # 거리 계산 함수
    def calculate_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # 엄지-검지 각도 계산
    angle_thumb_index = calculate_angle((wrist_x, wrist_y), (thumb_tip_x, thumb_tip_y), (index_tip_x, index_tip_y))
    
    # 엄지-검지 거리 계산
    distance_thumb_index = calculate_distance((thumb_tip_x, thumb_tip_y), (index_tip_x, index_tip_y))
    distance_threshold = w * 0.08
    
    # 검지와 중지 상태 확인
    index_below_middle = index_tip_y > middle_tip_y
    
    # 약지, 소지가 접혀있는지 확인
    ring_folded_by_middle = ring_tip_y > middle_tip_y * 0.5
    pinky_folded_by_middle = pinky_tip_y > middle_tip_y * 0.5
    
    # 손가락 길이 임계값
    finger_length_threshold = w * 0.02
    
    # 손가락 길이 및 펴짐 상태 확인
    index_length = calculate_distance((index_pip_x, index_pip_y), (index_tip_x, index_tip_y))
    middle_length = calculate_distance((middle_pip_x, middle_pip_y), (middle_tip_x, middle_tip_y))
    
    index_extended = index_length > finger_length_threshold
    middle_extended = middle_length > finger_length_threshold
    
    # 검지-중지 간격 확인
    distance_index_middle = calculate_distance((index_tip_x, index_tip_y), (middle_tip_x, middle_tip_y))
    index_middle_spread = distance_index_middle > w * 0.05
    
    # V 포즈 판정
    if angle_thumb_index > 25 and distance_thumb_index > distance_threshold and index_extended and middle_extended and index_middle_spread:
        # 일반 V 포즈
        if index_tip_y < wrist_y and thumb_tip_y < wrist_y and ring_folded_by_middle and pinky_folded_by_middle:
            return True
        # 아래 방향 V 포즈
        elif index_below_middle and thumb_tip_y > wrist_y and middle_tip_y > wrist_y and ring_folded_by_middle and pinky_folded_by_middle:
            return True
        elif not index_below_middle and thumb_tip_y > wrist_y and index_tip_y > wrist_y and angle_thumb_index > 50 and ring_folded_by_middle and pinky_folded_by_middle:
            return True
    
    return False

def is_thumbs(landmarks):
    """따봉 포즈인지 확인"""
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    
    # 엄지가 위로 향하고 다른 손가락들이 접혀있는지
    thumb_up = thumb_tip.y < thumb_ip.y and thumb_tip.y < wrist.y + 0.05 
    
    # 다른 손가락들이 접혀 있는지 확인 (y 좌표 기준, 이미지 위쪽이 0)
    # 각 손가락 끝(TIP)이 중간 마디(PIP)보다 y값이 크면 접힌 것으로 간주 (더 아래쪽에 위치)
    # 약간의 오차를 허용하기 위해 -0.02 (또는 +0.02, 이미지 좌표계에 따라) 추가
    fingers_folded = (
        index_tip.y > index_pip.y - 0.02 and
        middle_tip.y > middle_pip.y - 0.02 and
        ring_tip.y > ring_pip.y - 0.02 and
        pinky_tip.y > pinky_pip.y - 0.02
    )
    
    # 엄지손가락이 다른 모든 손가락 끝보다 위에 있는지 확인
    thumb_highest = (
        thumb_tip.y < index_tip.y and
        thumb_tip.y < middle_tip.y and
        thumb_tip.y < ring_tip.y and
        thumb_tip.y < pinky_tip.y
    )
    return thumb_up and thumb_highest and fingers_folded

def classify_hand_pose(image):
    """
    image: BGR np.ndarray (cv2.imread 결과)
    return: dict {포즈이름: True}
    """
    image_height, image_width = image.shape[:2]
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return {}

    pose_results = {}

    for hand_landmarks in results.multi_hand_landmarks:
        lm = hand_landmarks.landmark

        if is_heart(lm, image_height, image_width):
            pose_results['Heart'] = True
        if is_v(lm):
            pose_results['Hands'] = True
        if is_thumbs(lm):
            pose_results['Thumbs'] = True

    return pose_results

def get_pose_func_map():
    """
    포즈 키워드와 판별 함수 매핑 딕셔너리 반환
    키워드는 소문자로 통일
    """
    return {
        "하트": lambda lm, h, w: is_heart(lm, h, w),
        "브이": lambda lm, h, w: is_v(lm),
        "최고": lambda lm, h, w: is_thumbs(lm),
    }

def detect_pose_by_keyword(image, keyword):
    """
    image: BGR np.ndarray (cv2.imread 결과)
    keyword: 판별할 포즈 키워드 (소문자)
    return: True/False (해당 포즈 감지 여부)
    """
    image_height, image_width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        print("손 인식 실패:", keyword)
        return False
    pose_func_map = get_pose_func_map()
    func = pose_func_map.get(keyword.lower())
    if func is None:
        print("키워드 매칭 실패:", keyword)
        return False
    for hand_landmarks in results.multi_hand_landmarks:
        lm = hand_landmarks.landmark
        print("landmark 개수:", len(lm))
        if func(lm, image_height, image_width):
            print("포즈 감지 성공:", keyword)
            return True
    print("포즈 감지 실패:", keyword)
    return False