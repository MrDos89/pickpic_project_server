import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def classify_pose(landmarks):
    lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    lk = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    rk = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    ra = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    def distance(a, b):
        return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5

    detected_poses = []

    if (lw.y < ls.y and rw.y < rs.y):
        detected_poses.append("만세 포즈")
    if ((lw.y < ls.y and rw.y >= rs.y) or (rw.y < rs.y and lw.y >= ls.y)):
        detected_poses.append("손 흔들기 포즈")
    shoulder_width = abs(rs.x - ls.x)
    if (abs(lw.x - ls.x) > shoulder_width * 1.5 and abs(rw.x - rs.x) > shoulder_width * 1.5):
        detected_poses.append("팔 벌리기 포즈")
    hip_center_y = (lh.y + rh.y) / 2
    if (abs(lw.y - hip_center_y) < 0.05 and abs(rw.y - hip_center_y) < 0.05):
        detected_poses.append("두 손 허리 포즈")
    if distance(lw, rw) < 0.05:
        detected_poses.append("손 모으기 포즈")

    # 점프샷 판별 threshold 수정
    torso_y = (ls.y + rs.y + lh.y + rh.y) / 4
    feet_y = (la.y + ra.y) / 2
    if (feet_y < torso_y - 0.1):
        detected_poses.append("점프샷 포즈")

    if (abs(lh.y - lk.y) < 0.1 and abs(lk.y - la.y) < 0.1 and
        abs(rh.y - rk.y) < 0.1 and abs(rk.y - ra.y) < 0.1):
        detected_poses.append("서있는 포즈")
    if (lh.y > lk.y and rh.y > rk.y):
        detected_poses.append("앉은 포즈")
    if (abs(lk.y - rk.y) > 0.2):
        detected_poses.append("런지 / 스트레칭 포즈")
    y_vals = [lw.y, rw.y, ls.y, rs.y, lh.y, rh.y, lk.y, rk.y, la.y, ra.y]
    if max(y_vals) - min(y_vals) < 0.1:
        detected_poses.append("누워있는 포즈")

    return ", ".join(detected_poses) + "입니다!" if detected_poses else "일반 포즈 (미분류)입니다."

# 이미지 경로
image_path = 'jump3.jpg'
image = cv2.imread(image_path)
if image is None:
    print("이미지를 불러오지 못했습니다. 경로를 확인하세요.")
    exit()

with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        print("포즈 랜드마크 검출 성공")
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        result_text = classify_pose(results.pose_landmarks.landmark)
        print(result_text)
        cv2.putText(image, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Pose Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("사람을 인식하지 못했습니다.")
