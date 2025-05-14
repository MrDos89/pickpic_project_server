import cv2
import mediapipe as mp
import math

# 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# 이미지 불러오기
image_path = '..\\data\\mil.jpg'  # 경례 이미지 파일명
image = cv2.imread(image_path)
if image is None:
    print("❌ 이미지를 불러올 수 없습니다.")
    exit()

# 전처리
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = hands.process(image_rgb)

# 벡터 및 각도 계산 함수
def calculate_angle(p1, p2, p3):
    v1 = [p1.x - p2.x, p1.y - p2.y, p1.z - p2.z]
    v2 = [p3.x - p2.x, p3.y - p2.y, p3.z - p2.z]
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a * a for a in v1))
    magnitude_v2 = math.sqrt(sum(a * a for a in v2))
    if magnitude_v1 * magnitude_v2 == 0:
        return 0
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle = math.acos(min(1.0, max(-1.0, cos_theta)))
    return math.degrees(angle)

# 경례 포즈 판단 함수
def is_salute(lm):
    def finger_extended(tip_id):
        pip_id = tip_id - 2
        return lm[tip_id].y < lm[pip_id].y

    index_extended = finger_extended(mp_hands.HandLandmark.INDEX_FINGER_TIP)
    middle_extended = finger_extended(mp_hands.HandLandmark.MIDDLE_FINGER_TIP)
    ring_extended = finger_extended(mp_hands.HandLandmark.RING_FINGER_TIP)
    pinky_extended = finger_extended(mp_hands.HandLandmark.PINKY_TIP)

    # 엄지 각도
    thumb_angle = calculate_angle(
        lm[mp_hands.HandLandmark.THUMB_CMC],
        lm[mp_hands.HandLandmark.THUMB_MCP],
        lm[mp_hands.HandLandmark.THUMB_IP]
    )
    thumb_extended = thumb_angle > 30  # 완전 펴지지 않아도 OK

    # 손가락이 이마 근처에 있는지 판단: 검지 끝 위치가 화면 위쪽 40%에 위치하면 이마 근처로 간주
    index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    is_near_forehead = index_tip.y < 0.4

    return all([index_extended, middle_extended, ring_extended, pinky_extended, thumb_extended, is_near_forehead])

# 분석 및 출력
salute_detected = False

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        lm = hand_landmarks.landmark
        if is_salute(lm):
            salute_detected = True
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if salute_detected:
        print("경례 포즈입니다!")
        cv2.putText(image, "SALUTE Pose", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        print("❌ 경례 포즈가 아닙니다.")
        cv2.putText(image, "Not SALUTE Pose", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
else:
    print("❌ 손을 인식하지 못했습니다.")
    cv2.putText(image, "No hand detected", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

# 결과 표시
cv2.imshow("Salute Pose Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
