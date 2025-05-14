import cv2
import mediapipe as mp
import math

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# 이미지 불러오기
image_path = '..\\data\okay3.jpg'
image = cv2.imread(image_path)

if image is None:
    print("❌ 이미지를 불러올 수 없습니다.")
    exit()

# 이미지 전처리
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = hands.process(image_rgb)

# 유틸 함수 정의
def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def is_finger_extended(lm, tip_id, pip_id, mcp_id):
    return lm[tip_id].y < lm[pip_id].y < lm[mcp_id].y

def is_okay_pose(lm):
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = calculate_distance(thumb_tip, index_tip)

    middle_extended = is_finger_extended(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                         mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                                         mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
    ring_extended = is_finger_extended(lm, mp_hands.HandLandmark.RING_FINGER_TIP,
                                       mp_hands.HandLandmark.RING_FINGER_PIP,
                                       mp_hands.HandLandmark.RING_FINGER_MCP)
    pinky_extended = is_finger_extended(lm, mp_hands.HandLandmark.PINKY_TIP,
                                        mp_hands.HandLandmark.PINKY_PIP,
                                        mp_hands.HandLandmark.PINKY_MCP)

    return (distance < 0.05 and middle_extended and ring_extended and pinky_extended)

# 분석 및 시각화
okay_detected = False

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        lm = hand_landmarks.landmark
        if is_okay_pose(lm):
            okay_detected = True  # 한 손이라도 OKAY면 True
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if okay_detected:
        print("👌 OKAY 포즈입니다!")
        cv2.putText(image, "OKAY Pose", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        print("❌ OKAY 포즈가 아닙니다.")
        cv2.putText(image, "Not OKAY Pose", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
else:
    print("❌ 손을 인식하지 못했습니다.")
    cv2.putText(image, "No hand detected", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

# 결과 출력
cv2.imshow("OKAY Pose Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
