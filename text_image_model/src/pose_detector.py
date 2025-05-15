import cv2
import mediapipe as mp
import math
import os

class PoseDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=4,
            min_detection_confidence=0.5
        )

    def calculate_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def calculate_angle(self, p1, p2, p3):
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

    def detect_fist(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return "❌ 이미지를 불러올 수 없습니다.", None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        fist_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                thumb_tip = lm[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = lm[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                if self.calculate_distance(thumb_tip, index_tip) < 0.05:
                    continue

                fingers_folded = []
                for tip_id in [
                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    self.mp_hands.HandLandmark.RING_FINGER_TIP,
                    self.mp_hands.HandLandmark.PINKY_TIP
                ]:
                    dip_id = tip_id - 3
                    if lm[tip_id].y > lm[dip_id].y:
                        fingers_folded.append(True)
                    else:
                        fingers_folded.append(False)

                if all(fingers_folded):
                    fist_detected = True
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            if fist_detected:
                cv2.putText(image, "FIST Pose", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                return "✊ 주먹 포즈입니다!", image
            else:
                cv2.putText(image, "Not FIST Pose", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                return "❌ 주먹 포즈가 아닙니다.", image
        else:
            cv2.putText(image, "No hand detected", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            return "❌ 손을 인식하지 못했습니다.", image

    def detect_v_pose(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return "❌ 이미지를 불러올 수 없습니다.", None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        v_pose_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                wrist = lm[self.mp_hands.HandLandmark.WRIST]
                thumb_tip = lm[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = lm[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

                angle = self.calculate_angle(wrist, thumb_tip, index_tip)
                distance = self.calculate_distance(thumb_tip, index_tip)

                if angle > 25 and distance > 0.1:
                    v_pose_detected = True
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            if v_pose_detected:
                cv2.putText(image, "V Pose", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                return "✌️ 브이 포즈입니다!", image
            else:
                cv2.putText(image, "Not V Pose", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                return "❌ 브이 포즈가 아닙니다.", image
        else:
            cv2.putText(image, "No hand detected", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            return "❌ 손을 인식하지 못했습니다.", image

    def detect_heart(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return "❌ 이미지를 불러올 수 없습니다.", None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        heart_detected = False
        heart_type = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                thumb_tip = lm[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = lm[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # 손가락 하트 체크
                if self.calculate_distance(thumb_tip, index_tip) < 0.05:
                    heart_detected = True
                    heart_type = "Finger Heart"
                    break

                # 볼하트 체크
                if thumb_tip.y < 0.4 and index_tip.y < 0.4:
                    heart_detected = True
                    heart_type = "Cheek Heart"
                    break

                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            if heart_detected:
                cv2.putText(image, f"{heart_type}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                return f"💗 {heart_type} 감지!", image
            else:
                cv2.putText(image, "Not Heart Pose", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                return "❌ 하트 포즈가 아닙니다.", image
        else:
            cv2.putText(image, "No hand detected", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            return "❌ 손을 인식하지 못했습니다.", image

    def detect_military(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return "❌ 이미지를 불러올 수 없습니다.", None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        salute_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                index_tip = lm[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # 경례 포즈 체크 (검지가 이마 근처에 있는지)
                if index_tip.y < 0.4:
                    salute_detected = True
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            if salute_detected:
                cv2.putText(image, "SALUTE Pose", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                return "경례 포즈입니다!", image
            else:
                cv2.putText(image, "Not SALUTE Pose", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                return "❌ 경례 포즈가 아닙니다.", image
        else:
            cv2.putText(image, "No hand detected", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            return "❌ 손을 인식하지 못했습니다.", image

    def detect_okay(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return "❌ 이미지를 불러올 수 없습니다.", None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        okay_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                thumb_tip = lm[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = lm[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # OKAY 포즈 체크
                if self.calculate_distance(thumb_tip, index_tip) < 0.05:
                    okay_detected = True
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            if okay_detected:
                cv2.putText(image, "OKAY Pose", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                return "👌 OKAY 포즈입니다!", image
            else:
                cv2.putText(image, "Not OKAY Pose", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                return "❌ OKAY 포즈가 아닙니다.", image
        else:
            cv2.putText(image, "No hand detected", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            return "❌ 손을 인식하지 못했습니다.", image

    def detect_thumbs(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return "❌ 이미지를 불러올 수 없습니다.", None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        thumbs_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                thumb_tip = lm[self.mp_hands.HandLandmark.THUMB_TIP]
                wrist = lm[self.mp_hands.HandLandmark.WRIST]
                
                # 엄지척 체크
                if thumb_tip.y < wrist.y:
                    thumbs_detected = True
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            if thumbs_detected:
                cv2.putText(image, "THUMBS UP", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                return "👍 엄지척 감지!", image
            else:
                cv2.putText(image, "Not THUMBS UP", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                return "❌ 엄지척이 아닙니다.", image
        else:
            cv2.putText(image, "No hand detected", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            return "❌ 손을 인식하지 못했습니다.", image 