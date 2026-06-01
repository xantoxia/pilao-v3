import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------- 姿态识别核心函数 ----------------------
def analyze_pose(image_np):
    # 仅在函数内导入，避免全局初始化报错
    import mediapipe as mp
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    h, w = image_np.shape[:2]
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    metrics = {}
    output_image = image_np.copy()

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        # 关键点坐标提取
        nose = (int(lm[mp_pose.PoseLandmark.NOSE].x * w), int(lm[mp_pose.PoseLandmark.NOSE].y * h))
        l_sh = (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w), int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        r_sh = (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
        l_el = (int(lm[mp_pose.PoseLandmark.LEFT_ELBOW].x * w), int(lm[mp_pose.PoseLandmark.LEFT_ELBOW].y * h))
        r_el = (int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h))
        l_hi = (int(lm[mp_pose.PoseLandmark.LEFT_HIP].x * w), int(lm[mp_pose.PoseLandmark.LEFT_HIP].y * h))
        r_hi = (int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h))

        # 绘制骨骼
        cv2.line(output_image, l_sh, r_sh, (0, 255, 0), 2)
        cv2.line(output_image, l_sh, l_el, (0, 255, 0), 2)
        cv2.line(output_image, r_sh, r_el, (0, 255, 0), 2)
        cv2.line(output_image, l_sh, l_hi, (0, 255, 0), 2)
        cv2.line(output_image, r_sh, r_hi, (0, 255, 0), 2)

        # 计算角度（简化版）
        def calc_angle(a, b, c):
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            ba = a - b
            bc = c - b
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

        metrics["颈部前屈"] = calc_angle(l_hi, l_sh, nose)
        metrics["左肩角度"] = calc_angle(l_hi, l_sh, l_el)
        metrics["右肩角度"] = calc_angle(r_hi, r_sh, r_el)

    pose.close()
    return output_image, metrics

# ---------------------- Streamlit UI ----------------------
st.title("📸 图片人体角度识别")
uploaded_file = st.file_uploader("上传图片", type=["png", "jpg"])
threshold = st.slider("风险阈值 (°)", 30, 90, 60)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    with st.spinner("正在识别姿态..."):
        processed_img, metrics = analyze_pose(img_np)

    col1, col2 = st.columns(2)
    with col1:
        st.image(processed_img, channels="BGR", caption="识别结果")
    with col2:
        st.subheader("角度数据")
        for joint, angle in metrics.items():
            status = "⚠️" if angle > threshold else "✅"
            st.markdown(f"{status} **{joint}**: `{angle:.1f}°`")
