import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from matplotlib import font_manager
import os
from openai import OpenAI
import base64
import requests
import datetime
import io
import pytz

# ---------------------- 1. 基础配置 ----------------------
st.set_page_config(page_title="疲劳评估系统", layout="wide")

# ---------------------- 页面紧凑样式（修复顶部+底部遮挡） ----------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2.5rem !important;
    padding-bottom: 6rem !important;
    max-width: 1000px;
}
h1, h2, h3, h4 {
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}
.stSlider, .stNumberInput, .stSelectbox {
    margin-bottom: 0.3rem;
}
.stExpander {
    padding: 0.2rem 1rem;
}
div[data-testid="column"] {
    padding: 0 0.5rem;
}
.stChatInputContainer {
    position: relative !important;
    bottom: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# GitHub 配置
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_USERNAME = 'xantoxia'
GITHUB_REPO = 'pilao-v2'
GITHUB_BRANCH = 'main'
FILE_PATH = 'fatigue_data.csv'

# 字体配置
font_path = "SourceHanSansCN-Normal.otf"
font_prop = None
if os.path.exists(font_path):
    font_prop = font_manager.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 2. 模型加载 ----------------------
# Load the uploaded file
file_path = 'corrected_fatigue_simulation_data_Chinese.csv'
data = pd.read_csv(file_path, encoding='gbk')

# 1. Features and labels
X = data.drop(columns=["疲劳等级"])
y = data["疲劳等级"]

# Normalize column names to avoid spaces
X.columns = X.columns.str.replace(' ', '_')

# 2. Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Predictions
y_pred = model.predict(X_test)

# 5. Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Feature importance
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Create feature importance plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=ax)
ax.set_title("Feature Importance in Fatigue Classification")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
set_font_properties(ax, font_prop)

# Save model
with open("fatigue_model.pkl", "wb") as f:
    pickle.dump(model, f)

# ---------------------- 3. 图片角度识别模块 ----------------------
def load_pose_models():
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    return mp_pose, mp_hands, pose, hands

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

def draw_landmarks(image, joints):
    colors = {'neck': (255, 200, 0), 'shoulder': (0, 255, 0), 'elbow': (0, 255, 255), 'wrist': (255, 0, 255)}
    nose = tuple(map(int, joints['鼻子'][:2]))
    shoulder_mid = tuple(map(int, joints['mid']['肩膀'][:2]))
    hip_mid = tuple(map(int, joints['mid']['臀部'][:2]))
    cv2.line(image, nose, shoulder_mid, colors['neck'], 2)
    cv2.line(image, shoulder_mid, hip_mid, colors['neck'], 2)
    for side in ['左侧', '右侧']:
        pt1 = tuple(map(int, joints[side]['肩膀'][:2]))
        pt2 = tuple(map(int, joints[side]['肘部'][:2]))
        cv2.line(image, pt1, pt2, colors['shoulder'], 2)
        pt3 = tuple(map(int, joints[side]['肘部'][:2]))
        pt4 = tuple(map(int, joints[side]['手腕'][:2]))
        cv2.line(image, pt3, pt4, colors['elbow'], 2)
        if '食指尖端' in joints[side]:
            pt5 = tuple(map(int, joints[side]['手腕'][:2]))
            pt6 = tuple(map(int, joints[side]['食指尖端'][:2]))
            cv2.line(image, pt5, pt6, colors['wrist'], 2)

def process_image(image):
    mp_pose, mp_hands, pose, hands = load_pose_models()
    H, W, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_result = pose.process(img_rgb)
    hands_result = hands.process(img_rgb)
    metrics = {'angles': {}}

    if pose_result.pose_landmarks:
        def get_pose_pt(landmark):
            return get_coord(pose_result.pose_landmarks.landmark[landmark], 'pose', W, H)
        
        # 提取全身关键点
        left_shoulder = get_pose_pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_pose_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_elbow = get_pose_pt(mp_pose.PoseLandmark.LEFT_ELBOW)
        right_elbow = get_pose_pt(mp_pose.PoseLandmark.RIGHT_ELBOW)
        left_wrist = get_pose_pt(mp_pose.PoseLandmark.LEFT_WRIST)
        right_wrist = get_pose_pt(mp_pose.PoseLandmark.RIGHT_WRIST)
        left_hip = get_pose_pt(mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = get_pose_pt(mp_pose.PoseLandmark.RIGHT_HIP)
        left_knee = get_pose_pt(mp_pose.PoseLandmark.LEFT_KNEE)
        right_knee = get_pose_pt(mp_pose.PoseLandmark.RIGHT_KNEE)
        nose = get_pose_pt(mp_pose.PoseLandmark.NOSE)

        # 躯干中点
        mid_shoulder = [(left_shoulder[i] + right_shoulder[i])/2 for i in range(3)]
        mid_hip = [(left_hip[i] + right_hip[i])/2 for i in range(3)]
        mid_knee = [(left_knee[i] + right_knee[i])/2 for i in range(3)]

        joints = {
            '左侧': {'肩膀': left_shoulder, '肘部': left_elbow, '手腕': left_wrist, '臀部': left_hip, '膝部': left_knee},
            '右侧': {'肩膀': right_shoulder, '肘部': right_elbow, '手腕': right_wrist, '臀部': right_hip, '膝部': right_knee},
            'mid': {'肩膀': mid_shoulder, '臀部': mid_hip, '膝部': mid_knee},
            '鼻子': nose
        }

        # 手部关键点解析
        if hands_result.multi_hand_landmarks:
            for hand in hands_result.multi_hand_landmarks:
                side = '左侧' if hand.landmark[0].x < 0.5 else '右侧'
                wrist = get_coord(hand.landmark[mp_hands.HandLandmark.WRIST], 'hands', W, H)
                mcp = get_coord(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], 'hands', W, H)
                tip = get_coord(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP], 'hands', W, H)
                joints[side].update({'手腕': wrist, '食指中节': mcp, '食指尖端': tip})

        # 颈部、背部角度（沿用原有成熟计算）
        neck = calculate_neck_flexion(nose, mid_shoulder, mid_hip)
        metrics['angles']['颈部前屈'] = max(0, min(90, neck))
        trunk = calculate_trunk_flexion(mid_shoulder, mid_hip, mid_knee)
        metrics['angles']['背部屈曲'] = max(0, min(90, trunk))

        # 左右肢体角度计算
        for side in ['左侧', '右侧']:
            s = np.array(joints[side]['肩膀'][:2])
            e = np.array(joints[side]['肘部'][:2])
            w = np.array(joints[side]['手腕'][:2])

            # ========== 核心修复：肩部上举（适配背面视角） ==========
            # 定义基准：肩膀向下的垂直向量（人体竖直向下）
            vertical_down = np.array([0, 1])
            # 手臂向量：肩膀 → 肘部
            arm_vec = e - s

            # 计算夹角
            if np.linalg.norm(arm_vec) < 1e-6:
                shoulder_angle = 0.0
            else:
                dot_val = np.dot(arm_vec, vertical_down)
                cos_ang = dot_val / (np.linalg.norm(arm_vec) * np.linalg.norm(vertical_down))
                raw_angle = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
                # 关键：背面视角大角度翻转，下垂统一转为小角度
                shoulder_angle = 180 - raw_angle if raw_angle > 90 else raw_angle

            # 限制区间 0~180°
            metrics['angles'][f'{side} 肩部上举'] = round(max(0, min(180, shoulder_angle)), 1)

            # 肩部前伸（复用躯干角度做参考，保持稳定）
            metrics['angles'][f'{side} 肩部前伸'] = metrics['angles']['背部屈曲']

            # 肘部屈伸
            elbow_raw = calculate_angle(joints[side]['肩膀'], joints[side]['肘部'], joints[side]['手腕'], 'sagittal')
            metrics['angles'][f'{side} 肘部屈伸'] = round(max(0, min(180, elbow_raw)), 1)

            # 手腕角度
            if '食指尖端' in joints[side]:
                mcp = joints[side]['食指中节']
                tip = joints[side]['食指尖端']
                wrist_ext = calculate_angle(joints[side]['肘部'], joints[side]['手腕'], tip, 'sagittal')
                metrics['angles'][f'{side} 手腕背伸'] = round(max(0, min(70, wrist_ext)), 1)
                wrist_dev = calculate_angle(mcp, joints[side]['手腕'], tip, 'frontal')
                metrics['angles'][f'{side} 手腕桡偏'] = round(max(-30, min(30, wrist_dev)), 1)

        try:
            draw_landmarks(image, joints)
        except:
            pass

    pose.close()
    hands.close()
    return image, metrics

# ---------------------- 4. 疲劳评估核心模块 ----------------------
def get_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return ""

def get_file_sha(file_path):
    url = f'https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{file_path}'
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()['sha']
    else:
        return None

def save_to_csv(input_data, result, body_fatigue, cognitive_fatigue, emotional_fatigue):
    body_fatigue_score = calculate_score(body_fatigue)
    cognitive_fatigue_score = calculate_score(cognitive_fatigue)
    emotional_fatigue_score = calculate_score(emotional_fatigue)
    tz = pytz.timezone('Asia/Shanghai')
    timestamp = datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
    data = {
        "颈部前屈": int(input_data["颈部前屈"].values[0]), "颈部后仰": int(input_data["颈部后仰"].values[0]),
        "肩部上举范围": int(input_data["肩部上举范围"].values[0]), "肩部前伸范围": int(input_data["肩部前伸范围"].values[0]),
        "肘部屈伸": int(input_data["肘部屈伸"].values[0]), "手腕背伸": int(input_data["手腕背伸"].values[0]),
        "手腕桡偏/尺偏": int(input_data["手腕桡偏/尺偏"].values[0]), "背部屈曲范围": int(input_data["背部屈曲范围"].values[0]),
        "持续时间": int(input_data["持续时间"].values[0]), "重复频率": int(input_data["重复频率"].values[0]),
        "fatigue_result": result, "body_fatigue_score": body_fatigue_score,
        "cognitive_fatigue_score": cognitive_fatigue_score, "emotional_fatigue_score": emotional_fatigue_score,
        "timestamp": timestamp
    }
    df = pd.DataFrame([data])
    if os.path.exists(FILE_PATH):
        existing_content = get_file_content(FILE_PATH)
        if existing_content and existing_content.strip():
            existing_df = pd.read_csv(io.StringIO(existing_content))
        else:
            existing_df = pd.DataFrame(columns=data.keys())
    else:
        existing_df = pd.DataFrame(columns=data.keys())
    updated_df = pd.concat([existing_df, df], ignore_index=True)
    updated_df.to_csv(FILE_PATH, index=False)

def upload_to_github(file_path):
    try:
        sha_value = get_file_sha(file_path)
        with open(file_path, 'rb') as file:
            content = base64.b64encode(file.read()).decode()
        url = f'https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{file_path}'
        commit_message = "Add new fatigue data"
        data = {"message": commit_message, "branch": GITHUB_BRANCH, "content": content}
        if sha_value:
            data["sha"] = sha_value
        headers = {'Authorization': f'token {GITHUB_TOKEN}', 'Accept': 'application/vnd.github.v3+json'}
        requests.put(url, json=data, headers=headers)
    except:
        pass

def calculate_score(answer):
    if answer == '请选择': return 0
    elif answer == '完全没有': return 1
    elif answer == '偶尔': return 2
    elif answer == '经常': return 3
    else: return 4

def fatigue_prediction(input_data):
    global model
    if model is not None:
        try:
            prediction = model.predict(input_data)
            return ["低疲劳状态", "中疲劳状态", "高疲劳状态"][prediction[0]]
        except:
            return "中疲劳状态"
    else:
        return "中疲劳状态"

def call_ark_api(client, messages):
    try:
        ark_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        completion = client.chat.completions.create(model="Pro/deepseek-ai/DeepSeek-V3.2", messages=ark_messages, stream=True)
        for chunk in completion:
            if chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                if hasattr(choice, "delta") and choice.delta.content is not None:
                    yield choice.delta.content
    except Exception as e:
        st.error(f"API 调用错误: {str(e)}")

# ---------------------- 5. 侧边栏：模型性能 + 标准参考 ----------------------
if st.sidebar.checkbox("模型性能"):
    st.subheader("📊 模型评估")
    # 使用 st.columns 创建一列布局
    col1 = st.columns(1)
    # 在第一列中放置内容
    with col1[0]:
        st.markdown("""
        <div style="
            background-color: #F0F2F6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        ">
            <div style="
                font-size: 32px;
                font-weight: bold;
                color: #2E86C1;
            ">
                {:.2f}%
            </div>
            <div style="
                font-size: 16px;
                color: #666;
            ">
                准确性
            </div>
        </div>
        """.format(accuracy * 100), unsafe_allow_html=True)

    # 混淆矩阵
    st.markdown("### 混淆矩阵")
    fig_conf, ax_conf = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax_conf)
    ax_conf.set_xlabel("Predicted")
    ax_conf.set_ylabel("Actual")
    ax_conf.set_title("Confusion Matrix")
    st.pyplot(fig_conf)

    # 特征重要性
    st.markdown("### 特征重要性")
    st.pyplot(fig)

    # 添加一些说明
    st.markdown("""
    <div style="
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 10px;
        color: #2E7D32;
        margin-top: 20px;
    ">
        💡 提示：
        <ul>
            <li>混淆矩阵显示了模型的预测结果与实际标签的对比。对角线上的值表示正确预测的数量。</li>
            <li>特征重要性图展示了每个特征对模型预测的贡献程度。</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
@st.cache_resource
def load_model():
    with open("fatigue_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

if st.sidebar.checkbox("标准参考"):
    st.markdown("""
    <style>
        .header {
            font-size: 24px;
            font-weight: bold;
            color: #2E86C1;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 20px;
            font-weight: bold;
            color: #1A5276;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .sub-section {
            margin-left: 20px;
            margin-bottom: 10px;
        }
        .note {
            font-style: italic;
            color: #666;
            margin-top: 5px;
        }
        .highlight {
            color: #E74C3C;
            font-weight: bold;
        }
        .footer {
            margin-top: 30px;
            font-size: 14px;
            color: #888;
        }
    </style>

    <div class="header">人体各部位动作舒适范围参考指南</div>
    <div class="note">为了帮助您在日常工作或活动中保持健康的姿势，减少肌肉疲劳和关节损伤风险，以下是根据国际人因工程标准（如ISO 11226、ISO 9241等）整理的人体各部位动作舒适范围建议。请参考这些数据，优化您的姿势和工作环境设计。</div>

    <div class="section-title">1. 颈部</div>
    <div class="sub-section">
        - <span class="highlight">前屈（低头）</span>：0°~20°<br>
          <div class="note">（长时间前屈＞20°可能导致颈椎压力累积）</div>
        - <span class="highlight">后仰（抬头）</span>：0°~15°<br>
          <div class="note">（＞15°可能增加颈椎间盘压力，需避免静态保持）</div>
    </div>

    <div class="section-title">2. 肩部</div>
    <div class="sub-section">
        - <span class="highlight">上举（手臂抬高）</span>：0°~90°<br>
          <div class="note">（持续上举＞90°显著增加肩袖损伤风险，动态操作可偶尔达120°但需减少频率）</div>
        - <span class="highlight">前伸（手臂前伸）</span>：0°~30°<br>
          <div class="note">（＞30°易导致肩部肌肉疲劳，重复性任务应控制在15°以内）</div>
    </div>

    <div class="section-title">3. 肘部</div>
    <div class="sub-section">
        - <span class="highlight">屈伸（弯曲/伸直）</span>：60°~120°<br>
          <div class="note">（完全伸展或过度弯曲（如＞120°）会增加肌腱压力，中立位更安全）</div>
    </div>

    <div class="section-title">4. 手腕</div>
    <div class="sub-section">
        - <span class="highlight">背伸（手腕向上）</span>：0°~25°<br>
          <div class="note">（＞25°可能压迫腕管，ISO建议保持中立位附近）</div>
        - <span class="highlight">桡偏/尺偏（左右偏转）</span>：0°~15°<br>
          <div class="note">（超过15°容易造成腕管综合征或肌腱问题，需避免重复性极端偏转）</div>
    </div>

    <div class="section-title">5. 背部（腰椎）</div>
    <div class="sub-section">
        - <span class="highlight">屈曲（弯腰）</span>：0°~20°<br>
          <div class="note">（＞20°显著增加椎间盘压力，需配合髋关节活动以减少负荷）</div>
    </div>

    <div class="section-title">附加建议</div>
    <div class="sub-section">
        - <span class="highlight">动态任务</span>：优先采用中关节活动范围（如肩部上举60°~90°），避免极端姿势。<br>
        - <span class="highlight">静态保持</span>：任何姿势超过2分钟需设计支撑（如肘托、腰靠）。<br>
        - <span class="highlight">人机交互</span>：调整工作站高度、键盘倾斜度等，使关节自然接近中立位。
    </div>

    <div class="section-title">健康建议</div>
    <div class="sub-section">
        - 定期调整姿势，避免长时间保持同一姿势。<br>
        - 使用符合人因工程设计的工具和设备（如可调节桌椅、腕托等）。<br>
        - 结合适当的伸展运动，缓解肌肉疲劳。
    </div>

    <div class="footer">通过遵循以上建议，您可以有效减少肌肉骨骼疾病的风险，提升工作效率和舒适度。</div>
    """, unsafe_allow_html=True)

# ---------------------- 6. 主页面布局 ----------------------
st.markdown("<h1 style='text-align: center;'>疲劳评估系统（一体化版）</h1>", unsafe_allow_html=True)
st.markdown("""该工具依据国际标准ISO 11226（静态工作姿势）、美国国家职业安全健康研究所的《手动材料处理指南》以及OWAS分析与建议等多套国际标准和规范，对工作过程中的疲劳状态进行科学评估，支持「图片识别自动填数」和「手动输入角度数据」两种方式进行疲劳评估。""")

# 初始化会话状态
if 'neck_flexion' not in st.session_state: st.session_state.neck_flexion = 20
if 'shoulder_elevation' not in st.session_state: st.session_state.shoulder_elevation = 60
if 'shoulder_forward' not in st.session_state: st.session_state.shoulder_forward = 120
if 'elbow_flexion' not in st.session_state: st.session_state.elbow_flexion = 120
if 'wrist_extension' not in st.session_state: st.session_state.wrist_extension = 15
if 'wrist_deviation' not in st.session_state: st.session_state.wrist_deviation = 10
if 'back_flexion' not in st.session_state: st.session_state.back_flexion = 20
if "messages" not in st.session_state: st.session_state.messages = []
if "client" not in st.session_state: st.session_state.client = None

# ---------------------- 模块1：图片角度识别 ----------------------
with st.expander("📸 可选：上传图片自动识别角度（点击展开）"):
    uploaded_file = st.file_uploader("上传工作场景图片", type=["jpg", "png"])
    threshold = st.slider("风险阈值(°)", 30, 90, 60)
    if uploaded_file and uploaded_file.type.startswith("image"):
        img = Image.open(uploaded_file)
        img_np = np.array(img)
        if img_np.shape[-1] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        with st.spinner("正在识别..."):
            processed_img, metrics = process_image(img_np)
        col1, col2 = st.columns(2)
        with col1:
            st.image(processed_img, channels="BGR", use_container_width=True)
        with col2:
            st.subheader("识别结果")
            for joint, angle in metrics['angles'].items():
                status = "⚠️" if angle > threshold else "✅"
                st.markdown(f"{status} **{joint}**: `{angle:.1f}°`")
        if 'angles' in metrics:
            angles = metrics['angles']
            st.session_state.neck_flexion = int(angles.get('颈部前屈', 20))
            st.session_state.back_flexion = int(angles.get('背部屈曲', 20))
            st.session_state.shoulder_elevation = int(max(angles.get('左侧 肩部上举', 60), angles.get('右侧 肩部上举', 60)))
            st.session_state.shoulder_forward = int(max(angles.get('左侧 肩部前伸', 120), angles.get('右侧 肩部前伸', 120)))
            st.session_state.elbow_flexion = int(max(angles.get('左侧 肘部屈伸', 120), angles.get('右侧 肘部屈伸', 120)))
            st.session_state.wrist_extension = int(max(angles.get('左侧 手腕背伸', 15), angles.get('右侧 手腕背伸', 15)))
            st.session_state.wrist_deviation = int(max(angles.get('左侧 手腕桡偏', 10), angles.get('右侧 手腕桡偏', 10)))
            st.success("✅ 识别结果已自动填充到下方表单！")

# ---------------------- 模块2：疲劳评估表单 ----------------------
st.subheader("📊 角度参数（支持自动填充/手动修改）")
col1, col2 = st.columns(2)
with col1:
    neck_flexion = st.slider("颈部前屈", 0, 60, st.session_state.neck_flexion)
    neck_extension = st.slider("颈部后仰", 0, 60, 25)
    shoulder_elevation = st.slider("肩部上举范围", 0, 180, st.session_state.shoulder_elevation)
    shoulder_forward = st.slider("肩部前伸范围", 0, 180, st.session_state.shoulder_forward)
with col2:
    elbow_flexion = st.slider("肘部屈伸", 0, 180, st.session_state.elbow_flexion)
    wrist_extension = st.slider("手腕背伸", 0, 60, st.session_state.wrist_extension)
    wrist_deviation = st.slider("手腕桡偏/尺偏", 0, 30, st.session_state.wrist_deviation)
    back_flexion = st.slider("背部屈曲范围", 0, 60, st.session_state.back_flexion)

st.subheader("⏱️ 时间参数")
col3, col4 = st.columns(2)
with col3:
    task_duration = st.number_input("持续时间（秒）", min_value=0, value=5)
with col4:
    movement_frequency = st.number_input("重复频率（每5分钟）", min_value=0, value=35)

st.subheader("😰 主观感受")
col1, col2, col3 = st.columns(3)
with col1:
    body_fatigue = st.selectbox("1. 身体感到无力", ['请选择', '完全没有', '偶尔', '经常', '总是'], index=0)
with col2:
    cognitive_fatigue = st.selectbox("2. 影响睡眠", ['请选择', '完全没有', '偶尔', '经常', '总是'], index=0)
with col3:
    emotional_fatigue = st.selectbox("3. 肌肉酸痛或不适", ['请选择', '完全没有', '偶尔', '经常', '总是'], index=0)

# 评估按钮
if st.button("开始评估", type="primary"):
    if body_fatigue == '请选择' or cognitive_fatigue == '请选择' or emotional_fatigue == '请选择':
        st.warning("请先选择所有主观感受问题的答案！")
    else:
        input_data = pd.DataFrame({
            "颈部前屈": [neck_flexion], "颈部后仰": [neck_extension], "肩部上举范围": [shoulder_elevation],
            "肩部前伸范围": [shoulder_forward], "肘部屈伸": [elbow_flexion], "手腕背伸": [wrist_extension],
            "手腕桡偏/尺偏": [wrist_deviation], "背部屈曲范围": [back_flexion], "持续时间": [task_duration],
            "重复频率": [movement_frequency]
        })
        result = fatigue_prediction(input_data)
        st.success(f"评估结果：{result}")
        save_to_csv(input_data, result, body_fatigue, cognitive_fatigue, emotional_fatigue)
        upload_to_github(FILE_PATH)
        st.session_state.result = result

        # 保存评估记录，用于显示「所有评估记录」表格
        record = input_data.copy()
        record["评估结果"] = result
        record["主观身体感受"] = body_fatigue
        record["主观睡眠影响"] = cognitive_fatigue
        record["主观肌肉酸痛"] = emotional_fatigue
        st.session_state.predictions.append(record)
        
# ---------------------- 显示所有评估记录（加回你原来的功能） ----------------------
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

if st.session_state.predictions:
    st.subheader("📋 所有评估记录")
    prediction_df = pd.concat(st.session_state.predictions, ignore_index=True)
    st.dataframe(prediction_df, use_container_width=True)
    
# ---------------------- 模块3：AI 分析 ----------------------
if st.button("开始 AI 分析"):
    if "result" not in st.session_state:
        st.warning("请先完成疲劳评估！")
    else:
        try:
            API_KEY = st.secrets["API_KEY"]
            client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")
            st.session_state.client = client
            st.session_state.messages = [
                {"role": "system", "content": "你是人因工程专家，依据ISO 11226提供专业建议。"},
                {"role": "user", "content": f"用户目前{body_fatigue}身体无力，{cognitive_fatigue}影响睡眠，{emotional_fatigue}肌肉不适。关节角度：颈部前屈{neck_flexion}°，后仰{neck_extension}°，肩部上举{shoulder_elevation}°，前伸{shoulder_forward}°，肘部{elbow_flexion}°，手腕背伸{wrist_extension}°，桡偏{wrist_deviation}°，背部屈曲{back_flexion}°。请分析风险并给出改善建议。"}
            ]
            with st.spinner("AI分析中..."):
                response = ""
                for partial in call_ark_api(client, st.session_state.messages):
                    response += partial
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown("### 📝 AI 分析结果")
                st.markdown(response)
        except Exception as e:
            st.error(f"API 初始化失败：{e}")

# ---------------------- 永远显示的聊天框（修复后） ----------------------
prompt = st.chat_input("继续咨询人因工程问题：")
if prompt:
    if not st.session_state.client:
        try:
            API_KEY = st.secrets["API_KEY"]
            st.session_state.client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")
        except Exception as e:
            st.error(f"API 初始化失败：{e}")
            st.stop()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("思考中..."):
        full_response = ""
        for chunk in call_ark_api(st.session_state.client, st.session_state.messages):
            full_response += chunk
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# 显示聊天记录
def display_chat_messages():
    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            if msg["role"] != "system":
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

display_chat_messages()
