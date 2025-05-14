import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

image_path = '..\\data\\fist2.jpg'
image = cv2.imread(image_path)

if image is None:
    print("❌ 이미지를 불러올 수 없습니다.")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = hands.process(image_rgb)

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_angle(v1, v2):
    dot_product = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle = math.acos(cos_theta)
    return math.degrees(angle)

def calculate_vector(p1, p2):
    return [p2.x - p1.x, p2.y - p1.y, p2.z - p1.z]

def is_fist(lm):
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    if distance(thumb_tip, index_tip) < 0.05:
        return False

    fingers_folded = []
    for tip_id in [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]:
        dip_id = tip_id - 3
        if lm[tip_id].y > lm[dip_id].y:
            fingers_folded.append(True)
        else:
            fingers_folded.append(False)

    index_pip = lm[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = lm[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = lm[mp_hands.HandLandmark.PINKY_PIP]

    thumb_folded = (thumb_tip.x < index_pip.x and
                    thumb_tip.x < middle_pip.x and
                    thumb_tip.x < ring_pip.x and
                    thumb_tip.x < pinky_pip.x)

    return all(fingers_folded) or thumb_folded

fist_detected = False

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        lm = hand_landmarks.landmark
        if is_fist(lm):
            fist_detected = True
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if fist_detected:
        print("✊ 주먹 포즈입니다!")
        cv2.putText(image, "FIST Pose", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        print("❌ 주먹 포즈가 아닙니다.")
        cv2.putText(image, "Not FIST Pose", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
else:
    print("❌ 손을 인식하지 못했습니다.")
    cv2.putText(image, "No hand detected", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

cv2.imshow("Fist Pose Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()