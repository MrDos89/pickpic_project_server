# hand_model.py
import mediapipe as mp
import math
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# def is_fist(lm):
#     thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
#     index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#     if distance(thumb_tip, index_tip) < 0.05:
#         return False

#     fingers_folded = []
#     for tip_id in [
#         mp_hands.HandLandmark.INDEX_FINGER_TIP,
#         mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
#         mp_hands.HandLandmark.RING_FINGER_TIP,
#         mp_hands.HandLandmark.PINKY_TIP
#     ]:
#         dip_id = tip_id - 3
#         if lm[tip_id].y > lm[dip_id].y:
#             fingers_folded.append(True)
#         else:
#             fingers_folded.append(False)

#     index_pip = lm[mp_hands.HandLandmark.INDEX_FINGER_PIP]
#     middle_pip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
#     ring_pip = lm[mp_hands.HandLandmark.RING_FINGER_PIP]
#     pinky_pip = lm[mp_hands.HandLandmark.PINKY_PIP]

#     thumb_folded = (thumb_tip.x < index_pip.x and
#                     thumb_tip.x < middle_pip.x and
#                     thumb_tip.x < ring_pip.x and
#                     thumb_tip.x < pinky_pip.x)

#     return all(fingers_folded) or thumb_folded

# def is_fy(lm):
#     middle_tip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
#     middle_pip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
#     middle_mcp = lm[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

#     index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#     ring_tip = lm[mp_hands.HandLandmark.RING_FINGER_TIP]
#     pinky_tip = lm[mp_hands.HandLandmark.PINKY_TIP]

#     is_middle_extended = middle_tip.y < middle_pip.y < middle_mcp.y
#     are_others_folded = (
#         index_tip.y > lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
#         ring_tip.y > lm[mp_hands.HandLandmark.RING_FINGER_PIP].y and
#         pinky_tip.y > lm[mp_hands.HandLandmark.PINKY_PIP].y
#     )
#     return is_middle_extended and are_others_folded

def is_heart(lm, image_height, image_width):
    # 기본 하트 조건: 엄지와 검지가 가까움
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = lm[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = lm[mp_hands.HandLandmark.PINKY_TIP]

    thumb_index_dist = distance(thumb_tip, index_tip)

    # 다른 손가락이 접혀있는지 확인
    other_folded = (
        middle_tip.y > lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
        ring_tip.y > lm[mp_hands.HandLandmark.RING_FINGER_PIP].y and
        pinky_tip.y > lm[mp_hands.HandLandmark.PINKY_PIP].y
    )

    is_finger_heart = thumb_index_dist < 0.05 and other_folded

    # 얼굴 근처인지 판단
    hand_position = lm[mp_hands.HandLandmark.WRIST]
    hand_y = hand_position.y * image_height
    index_tip_y = index_tip.y * image_height

    is_cheek_position = hand_y < image_height * 0.5 and index_tip_y < image_height * 0.5

    # 손가락 하트 또는 볼 하트
    return is_finger_heart or (thumb_index_dist < 0.05 and is_cheek_position)

def is_hands(lm):
    def calculate_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def calculate_angle(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        return abs(angle)

    wrist = lm[mp.solutions.hands.HandLandmark.WRIST]
    thumb_tip = lm[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_finger_tip = lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_pip = lm[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
    middle_finger_tip = lm[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]

    wrist_x, wrist_y = wrist.x, wrist.y
    thumb_tip_x, thumb_tip_y = thumb_tip.x, thumb_tip.y
    index_tip_x_raw, index_tip_y_raw = index_finger_tip.x, index_finger_tip.y
    index_pip_x, index_pip_y = index_finger_pip.x, index_finger_pip.y
    middle_tip_x, middle_tip_y = middle_finger_tip.x, middle_finger_tip.y

    index_x = index_tip_x_raw if (index_tip_x_raw != 0 or index_tip_y_raw != 0) else index_pip_x
    index_y = index_tip_y_raw if (index_tip_x_raw != 0 or index_tip_y_raw != 0) else index_pip_y

    angle_thumb_index = calculate_angle(
        (wrist_x, wrist_y), (thumb_tip_x, thumb_tip_y), (index_x, index_y)
    )

    distance_thumb_index = calculate_distance((thumb_tip_x, thumb_tip_y), (index_x, index_y))

    distance_threshold = 0.08

    index_below_middle = index_y > middle_tip_y

    if angle_thumb_index > 25 and distance_thumb_index > distance_threshold:
        if index_y < wrist_y and thumb_tip_y < wrist_y:
            return True
        elif index_below_middle and thumb_tip_y > wrist_y and middle_tip_y > wrist_y:
            return True
        elif not index_below_middle and thumb_tip_y > wrist_y and index_y > wrist_y and angle_thumb_index > 50:
            return True

    return False

# def is_military(lm):
#     def finger_extended(tip_id):
#         pip_id = tip_id - 2
#         # 끝 마디가 중간 마디 위에 있으면 펴진 상태로 간주
#         return lm[tip_id].y < lm[pip_id].y

#     # 손가락들이 펴져있는지 체크
#     index_extended = finger_extended(mp_hands.HandLandmark.INDEX_FINGER_TIP)
#     middle_extended = finger_extended(mp_hands.HandLandmark.MIDDLE_FINGER_TIP)
#     ring_extended = finger_extended(mp_hands.HandLandmark.RING_FINGER_TIP)
#     pinky_extended = finger_extended(mp_hands.HandLandmark.PINKY_TIP)

#     # 엄지 각도 계산 함수
#     def calculate_angle(p1, p2, p3):
#         v1 = [p1.x - p2.x, p1.y - p2.y, p1.z - p2.z]
#         v2 = [p3.x - p2.x, p3.y - p2.y, p3.z - p2.z]
#         dot_product = sum(a * b for a, b in zip(v1, v2))
#         magnitude_v1 = math.sqrt(sum(a * a for a in v1))
#         magnitude_v2 = math.sqrt(sum(a * a for a in v2))
#         if magnitude_v1 * magnitude_v2 == 0:
#             return 0
#         cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
#         angle = math.acos(min(1.0, max(-1.0, cos_theta)))
#         return math.degrees(angle)

#     thumb_angle = calculate_angle(
#         lm[mp_hands.HandLandmark.THUMB_CMC],
#         lm[mp_hands.HandLandmark.THUMB_MCP],
#         lm[mp_hands.HandLandmark.THUMB_IP]
#     )
#     thumb_extended = thumb_angle > 30  # 어느 정도 펴져있음

#     # 검지 끝 위치가 화면 위쪽 40% 이내에 있으면 이마 근처로 간주
#     index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#     is_near_forehead = index_tip.y < 0.4

#     # 조건 모두 만족 시 경례 포즈
#     if all([index_extended, middle_extended, ring_extended, pinky_extended, thumb_extended, is_near_forehead]):
#         return True
#     else:
#         return False

# def is_okay(lm):
#     def calculate_distance(p1, p2):
#         return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

#     def is_finger_extended(tip_id, pip_id, mcp_id):
#         # 손가락 끝이 중간 마디 위에, 중간 마디가 근위 마디 위에 있으면 펴진 것으로 판단
#         return lm[tip_id].y < lm[pip_id].y < lm[mcp_id].y

#     thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
#     index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#     distance = calculate_distance(thumb_tip, index_tip)

#     middle_extended = is_finger_extended(mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
#                                          mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
#                                          mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
#     ring_extended = is_finger_extended(mp_hands.HandLandmark.RING_FINGER_TIP,
#                                        mp_hands.HandLandmark.RING_FINGER_PIP,
#                                        mp_hands.HandLandmark.RING_FINGER_MCP)
#     pinky_extended = is_finger_extended(mp_hands.HandLandmark.PINKY_TIP,
#                                         mp_hands.HandLandmark.PINKY_PIP,
#                                         mp_hands.HandLandmark.PINKY_MCP)

#     # 엄지와 검지 끝이 가까이 붙어 있고, 나머지 세 손가락은 펴져 있으면 OKAY 포즈
#     if distance < 0.05 and middle_extended and ring_extended and pinky_extended:
#         return True
#     else:
#         return False

def is_thumbs(lm):
    # lm: list of landmarks (normalized)
    # 엄지 끝과 관절 위치 비교 + 나머지 손가락 접힘 확인
    wrist = lm[mp_hands.HandLandmark.WRIST]
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = lm[mp_hands.HandLandmark.THUMB_IP]

    index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = lm[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = lm[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = lm[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = lm[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = lm[mp_hands.HandLandmark.PINKY_PIP]

    thumb_up = thumb_tip.y < thumb_ip.y and thumb_tip.y < wrist.y
    fingers_folded = (
        index_tip.y > index_pip.y and
        middle_tip.y > middle_pip.y and
        ring_tip.y > ring_pip.y and
        pinky_tip.y > pinky_pip.y
    )
    return thumb_up and fingers_folded

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

        # if is_fist(lm):
        #     pose_results['Fist'] = True
        # if is_fy(lm):
        #     pose_results['FY'] = True
        if is_heart(lm, image_height, image_width):
            pose_results['Heart'] = True
        if is_hands(lm):
            pose_results['Hands'] = True
        # if is_military(lm):
        #     pose_results['Military'] = True
        # if is_okay(lm):
        #     pose_results['Okay'] = True
        if is_thumbs(lm):
            pose_results['Thumbs'] = True

    return pose_results

def get_pose_func_map():
    """
    포즈 키워드와 판별 함수 매핑 딕셔너리 반환
    키워드는 소문자로 통일
    """
    return {
        "heart": lambda lm, h, w: is_heart(lm, h, w),
        "v": lambda lm, h, w: is_hands(lm),
        "thumbs": lambda lm, h, w: is_thumbs(lm),
        # "fist": lambda lm, h, w: is_fist(lm),  # 주석 해제 시 사용
        # "okay": lambda lm, h, w: is_okay(lm),
        # "fy": lambda lm, h, w: is_fy(lm),
        # "military": lambda lm, h, w: is_military(lm),
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
        return False
    pose_func_map = get_pose_func_map()
    func = pose_func_map.get(keyword.lower())
    if func is None:
        return False
    for hand_landmarks in results.multi_hand_landmarks:
        lm = hand_landmarks.landmark
        if func(lm, image_height, image_width):
            return True
    return False