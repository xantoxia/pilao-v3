import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# 初始化模型（全局只加载一次，避免重复加载）
@st.cache_resource
def load_models():
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    return mp_pose, mp_hands, pose, hands

mp_pose, mp_hands, pose, hands = load_models()

# ---------------------- 工具函数 ----------------------
def get_coord(landmark, model_type='pose', img_width=640, img_height=480):
    if model_type == 'pose':
        return [landmark.x * img_width, landmark.y * img_height, landmark.z * img_width]
    elif model_type == 'hands':
        return [landmark.x * img_width, landmark.y * img_height, 0]

def calculate_angle(a, b, c, plane='sagittal'):
    try:
        a = np.array(a)[:3].astype('float64')
        b = np.array(b)[:3].astype('float64')
        c = np.array(c)[:3].astype('float64')

        ba = a - b
        bc = c - b

        if plane == 'sagittal':
            ba = np.array([0, ba[1], ba[2]])
            bc = np.array([0, bc[1], bc[2]])
        elif plane == 'frontal':
            ba = np.array([ba[0], 0, ba[2]])
            bc = np.array([bc[0], 0, bc[2]])
        elif plane == 'transverse':
            ba = ba[:2]
            bc = bc[:2]

        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)
        if ba_norm < 1e-6 or bc_norm < 1e-6:
            return 0.0

        cosine = np.dot(ba, bc) / (ba_norm * bc_norm)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    except:
        return 0.0

def calculate_neck_flexion(nose, shoulder_mid, hip_mid):
    try:
        nose = np.array(nose)[:2]
        shoulder_mid = np.array(shoulder_mid)[:2]
        hip_mid = np.array(hip_mid)[:2]
        torso_vector = hip_mid - shoulder_mid
        torso_angle = np.degrees(np.arctan2(torso_vector[1], torso_vector[0]))
        head_vector = nose - shoulder_mid
        head_angle = np.degrees(np.arctan2(head_vector[1], head_vector[0]))
        flexion_angle = head_angle - torso_angle
        if flexion_angle < 0:
            flexion_angle += 360
        if flexion_angle > 180:
            flexion_angle = 360 - flexion_angle
        return 180 - flexion_angle
    except:
        return 0.0

def calculate_trunk_flexion(shoulder_mid, hip_mid, knee_mid):
    try:
        torso_vector = np.array(hip_mid) - np.array(shoulder_mid)
        torso_angle = np.degrees(np.arctan2(torso_vector[1], torso_vector[0]))
        leg_vector = np.array(knee_mid) - np.array(hip_mid)
        leg_angle = np.degrees(np.arctan2(leg_vector[1], leg_vector[0]))
        flexion_angle = leg_angle - torso_angle
        if flexion_angle < 0:
            flexion_angle += 360
        if flexion_angle > 180:
            flexion_angle = 360 - flexion_angle
        return flexion_angle
    except:
        return 0.0

# ---------------------- 图像处理 ----------------------
def process_image(image):
    H, W, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(img_rgb)
    hands_result = hands.process(img_rgb)
    metrics = {'angles': {}}

    if pose_result.pose_landmarks:
        def get_pose_pt(landmark):
            return get_coord(pose_result.pose_landmarks.landmark[landmark], 'pose', W, H)

        joints = {
            'left_side': {
                'shoulder': get_pose_pt(mp_pose.PoseLandmark.LEFT_SHOULDER),
                'elbow': get_pose_pt(mp_pose.PoseLandmark.LEFT_ELBOW),
                'wrist': get_pose_pt(mp_pose.PoseLandmark.LEFT_WRIST),
                'hip': get_pose_pt(mp_pose.PoseLandmark.LEFT_HIP),
                'knee': get_pose_pt(mp_pose.PoseLandmark.LEFT_KNEE)
            },
            'right_side': {
                'shoulder': get_pose_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER),
                'elbow': get_pose_pt(mp_pose.PoseLandmark.RIGHT_ELBOW),
                'wrist': get_pose_pt(mp_pose.PoseLandmark.RIGHT_WRIST),
                'hip': get_pose_pt(mp_pose.PoseLandmark.RIGHT_HIP),
                'knee': get_pose_pt(mp_pose.PoseLandmark.RIGHT_KNEE)
            },
            'mid': {
                'shoulder_mid': [(get_pose_pt(mp_pose.PoseLandmark.LEFT_SHOULDER)[i] +
                                get_pose_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)[i])/2 for i in range(3)],
                'hip_mid': [(get_pose_pt(mp_pose.PoseLandmark.LEFT_HIP)[i] +
                           get_pose_pt(mp_pose.PoseLandmark.RIGHT_HIP)[i])/2 for i in range(3)],
                'knee_mid': [(get_pose_pt(mp_pose.PoseLandmark.LEFT_KNEE)[i] +
                            get_pose_pt(mp_pose.PoseLandmark.RIGHT_KNEE)[i])/2 for i in range(3)]
            },
            'nose': get_pose_pt(mp_pose.PoseLandmark.NOSE)
        }

        if hands_result.multi_hand_landmarks:
            for hand in hands_result.multi_hand_landmarks:
                side = 'left_side' if hand.landmark[0].x < 0.5 else 'right_side'
                joints[side].update({
                    'hand_wrist': get_coord(hand.landmark[mp_hands.HandLandmark.WRIST], 'hands', W, H),
                    'index_mcp': get_coord(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], 'hands', W, H),
                    'index_tip': get_coord(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP], 'hands', W, H)
                })

        try:
            metrics['angles']['Neck Flexion'] = calculate_neck_flexion(
                joints['nose'], joints['mid']['shoulder_mid'], joints['mid']['hip_mid'])

            for side in ['left_side', 'right_side']:
                metrics['angles'][f'{side.capitalize()} Shoulder Elevation'] = calculate_angle(
                    joints[side]['hip'], joints[side]['shoulder'], joints[side]['elbow'], 'frontal')
                metrics['angles'][f'{side.capitalize()} Shoulder Flexion'] = calculate_angle(
                    joints[side]['hip'], joints[side]['shoulder'], joints[side]['elbow'], 'sagittal')
                metrics['angles'][f'{side.capitalize()} Elbow Flexion'] = calculate_angle(
                    joints[side]['shoulder'], joints[side]['elbow'], joints[side]['wrist'], 'sagittal')

                if 'hand_wrist' in joints[side]:
                    metrics['angles'][f'{side.capitalize()} Wrist Dorsiflexion'] = calculate_angle(
                        joints[side]['elbow'], joints[side]['hand_wrist'], joints[side].get('index_tip', [0,0,0]), 'sagittal')
                    metrics['angles'][f'{side.capitalize()} Wrist Radial Deviation'] = calculate_angle(
                        joints[side]['index_mcp'], joints[side]['hand_wrist'], joints[side].get('index_tip', [0,0,0]), 'frontal')

            metrics['angles']['Trunk Flexion'] = calculate_trunk_flexion(
                joints['mid']['shoulder_mid'], joints['mid']['hip_mid'], joints['mid']['knee_mid'])

            # 绘制骨骼
            nose = tuple(map(int, joints['nose'][:2]))
            shoulder_mid = tuple(map(int, joints['mid']['shoulder_mid'][:2]))
            hip_mid = tuple(map(int, joints['mid']['hip_mid'][:2]))
            cv2.line(image, nose, shoulder_mid, (255,200,0), 2)
            cv2.line(image, shoulder_mid, hip_mid, (255,200,0), 2)

        except:
            pass

    return image, metrics

# ---------------------- Streamlit UI ----------------------
st.title("📸 人体姿态角度识别")
uploaded_file = st.file_uploader("上传图片", type=["jpg","png"])
threshold = st.slider("风险阈值 (°)", 30,90,60)

if uploaded_file:
    img = Image.open(uploaded_file)
    img_np = np.array(img)
    if img_np.shape[-1] ==4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    processed_img, metrics = process_image(img_np)
    c1,c2 = st.columns(2)
    with c1:
        st.image(processed_img, channels="BGR")
    with c2:
        st.subheader("角度结果")
        for k,v in metrics['angles'].items():
            icon = "⚠️" if v>threshold else "✅"
            st.write(f"{icon} {k}: **{v:.1f}°**")
