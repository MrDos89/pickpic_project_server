import cv2
import mediapipe as mp
import math

# 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=4, min_detection_confidence=0.5)

# 거리 계산
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# 손가락 하트 판별
def is_finger_heart(lm):
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

    return thumb_index_dist < 0.05 and other_folded

# 양손 하트 판단 (완화 버전)
def is_possible_two_hand_heart(lm_list):
    # 손가락 하트를 만든 손이 2개 이상이면 양손하트일 가능성 높음
    heart_hands = [lm for lm in lm_list if is_finger_heart(lm)]
    return len(heart_hands) >= 2

# 볼하트 판단
def is_cheek_heart(lm, image_height, image_width):
    # 1. 엄지와 검지가 조금 멀어도 되므로, 이전의 '손가락 하트' 조건을 완화
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    thumb_index_dist = distance(thumb_tip, index_tip)  # 엄지와 검지 사이의 거리 계산

    # 2. 손이 볼 근처에 있어야 한다는 조건 추가 (볼 위치가 얼굴 중앙 부근이어야 함)
    hand_position = lm[mp_hands.HandLandmark.WRIST]  # 손목 위치
    hand_y = hand_position.y * image_height
    
    # 손목이 얼굴 근처, 즉 이미지의 하단 절반에 위치해야 한다는 조건
    if hand_y > image_height * 0.5:  # 얼굴 아래쪽에 있을 때만 볼하트로 인식
        return False

    # 3. 손끝이 아래로 향해야 함 (이미지 상단을 향해야 함)
    index_tip_y = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
    if index_tip_y < image_height * 0.5:  # 손끝이 이미지 상단 부근에 있을 경우
        return True  # 볼하트로 인식
    
    return False


# 이미지 불러오기
image_path = '..\\data\\heart.jpg'  # 이미지 경로
image = cv2.imread(image_path)

if image is None:
    print("❌ 이미지를 불러올 수 없습니다.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    finger_heart_detected = False
    cheek_heart_detected = False
    possible_two_hand_heart = False

    if results.multi_hand_landmarks:
        lm_list = []
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            lm_list.append(lm)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_finger_heart(lm):
                finger_heart_detected = True
            if is_cheek_heart(lm, image.shape[0], image.shape[1]):
                cheek_heart_detected = True

        if is_possible_two_hand_heart(lm_list):
            possible_two_hand_heart = True

    # 최종 판별
    if cheek_heart_detected:
        print("💞 볼하트 감지!")
        cv2.putText(image, "Cheek Heart", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 100, 180), 3)
    elif finger_heart_detected:
        print("💗 손가락 하트 감지!")
        cv2.putText(image, "Finger Heart", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
    elif possible_two_hand_heart:
        print("💖 양손 하트 감지 (추정)!")
        cv2.putText(image, "Possible Two-Hand Heart", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 0, 200), 3)
    else:
        print("❌ 하트 포즈가 감지되지 않았습니다.")
        cv2.putText(image, "No Heart Pose", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # 출력
    cv2.imshow("Heart Pose Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
