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

    # 1. 만세
    if (lw.y < ls.y and rw.y < rs.y):
        detected_poses.append("만세 포즈")

    # 2. 손 흔들기
    if ((lw.y < ls.y and rw.y >= rs.y) or (rw.y < rs.y and lw.y >= ls.y)):
        detected_poses.append("손 흔들기 포즈")

    # 3. 팔 벌리기
    shoulder_width = abs(rs.x - ls.x)
    if (abs(lw.x - ls.x) > shoulder_width * 1.5 and abs(rw.x - rs.x) > shoulder_width * 1.5):
        detected_poses.append("팔 벌리기 포즈")

    # 4. 두 손 허리
    hip_center_y = (lh.y + rh.y) / 2
    if (abs(lw.y - hip_center_y) < 0.05 and abs(rw.y - hip_center_y) < 0.05):
        detected_poses.append("두 손 허리 포즈")

    # 5. 손 모으기
    if distance(lw, rw) < 0.05:
        detected_poses.append("손 모으기 포즈")

    # 6. 점프샷
    torso_y = (ls.y + rs.y + lh.y + rh.y) / 4
    feet_y = (la.y + ra.y) / 2
    if (feet_y < torso_y - 0.2):
        detected_poses.append("점프샷 포즈")

    # 7. 서있음
    if (abs(lh.y - lk.y) < 0.1 and abs(lk.y - la.y) < 0.1 and
        abs(rh.y - rk.y) < 0.1 and abs(rk.y - ra.y) < 0.1):
        detected_poses.append("서있는 포즈")

    # 8. 앉기
    if (lh.y > lk.y and rh.y > rk.y):
        detected_poses.append("앉은 포즈")

    # 9. 런지 / 스트레칭
    if (abs(lk.y - rk.y) > 0.2):
        detected_poses.append("런지 / 스트레칭 포즈")

    # 10. 누워있음
    y_vals = [lw.y, rw.y, ls.y, rs.y, lh.y, rh.y, lk.y, rk.y, la.y, ra.y]
    if max(y_vals) - min(y_vals) < 0.1:
        detected_poses.append("누워있는 포즈")

    # 결과 반환
    if detected_poses:
        return ", ".join(detected_poses) + "입니다!"
    else:
        return "일반 포즈 (미분류)입니다."
