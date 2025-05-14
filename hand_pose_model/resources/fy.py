import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

image_path = '..\\data\\fy.jpg'
image = cv2.imread(image_path)

if image is None:
    print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = hands.process(image_rgb)

detected = False  # 중지만 펴진 손이 하나라도 있는지 여부

if results.multi_hand_landmarks:
    h, w, c = image.shape

    for hand_landmarks in results.multi_hand_landmarks:
        # 중지
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

        # 다른 손가락들
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # 중지가 펴져 있고, 나머지는 접혀 있는지 판단
        is_middle_extended = middle_tip.y < middle_pip.y and middle_pip.y < middle_mcp.y
        are_others_folded = (
            index_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
            ring_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
            pinky_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
        )

        if is_middle_extended and are_others_folded:
            detected = True  # 최소 하나의 손에서 조건 충족

        # 시각화
        for landmark in hand_landmarks.landmark:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

    # 전체 결과 출력
    if detected:
        print("✅ 중지만 펴짐!")
    else:
        print("❌ 중지만 펴지지 않음.")
else:
    print("❌ 손 인식 실패.")
    cv2.putText(image, "No hands detected", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

cv2.imshow('Middle Finger Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

