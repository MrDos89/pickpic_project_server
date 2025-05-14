import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

image_path = '..\\data\\thumbs.jpg'

image = cv2.imread(image_path)

if image is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = hands.process(image_rgb)

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]  # 엄지 첫 번째 관절
        thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]  # 엄지 두 번째 관절

        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

        if thumb_tip.y < thumb_ip.y and thumb_tip.y < wrist.y:
            fingers_folded = (
                index_tip.y > index_pip.y and
                middle_tip.y > middle_pip.y and
                ring_tip.y > ring_pip.y and
                pinky_tip.y > pinky_pip.y
            )
            if fingers_folded:
                print("✅ 최고! 따봉! 감지됨!")
            else:
                print("❌ 최고! 따봉! 아닙니다.")
        else:
            print("❌ 엄지 척이 아닙니다.")

        h, w, c = image.shape
        for landmark in hand_landmarks.landmark:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
else:
    print("❌ 인식되지 않음.")
    cv2.putText(image, "No detected", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

cv2.imshow('Thumbs Up Analysis', image)
cv2.waitKey(0)
cv2.destroyAllWindows()