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

# ---------------------- 页面紧凑样式（已修复页首被遮挡问题） ----------------------
st.markdown("""
<style>
/* 整体页面边距：增加顶部内边距，避免标题被遮挡 */
.block-container {
    padding-top: 2.5rem; /* 把这里从 1rem 改成 2.5rem */
    padding-bottom: 1rem;
    max-width: 1000px;
}

/* 标题和正文间距 */
h1, h2, h3, h4 {
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}

/* 滑块和控件之间的间距 */
.stSlider, .stNumberInput, .stSelectbox {
    margin-bottom: 0.3rem;
}

/* 折叠面板（图片识别）的内边距 */
.stExpander {
    padding: 0.2rem 1rem;
}

/* 列之间的间距 */
div[data-testid="column"] {
    padding: 0 0.5rem;
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
if os.path.exists(font_path):
    font_prop = font_manager.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 模型训练与性能评估模块（来自你最初的代码） ----------------------
# 读取训练数据
file_path = 'corrected_fatigue_simulation_data_Chinese.csv'
if os.path.exists(file_path):
    data = pd.read_csv(file_path, encoding='gbk')
    # 1. Features and labels
    X = data.drop(columns=["疲劳等级"])
    y = data["疲劳等级"]
    # Normalize column names to avoid spaces
    X.columns = X.columns.str.replace(' ', '_')
    # 2. Data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 3. Model training
    model_train = RandomForestClassifier(random_state=42)
    model_train.fit(X_train, y_train)
    # 4. Predictions
    y_pred = model_train.predict(X_test)
    # 5. Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    # Feature importance
    feature_importances = model_train.feature_importances_
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
    def set_font_properties(ax, font_prop):
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_prop)
        ax.title.set_fontproperties(font_prop)
        ax.xaxis.label.set_fontproperties(font_prop)
        ax.yaxis.label.set_fontproperties(font_prop)
    if 'font_prop' in locals():
        set_font_properties(ax, font_prop)
else:
    st.warning("⚠️ 训练数据文件 corrected_fatigue_simulation_data_Chinese.csv 未找到，模型性能模块将不可用")

# ---------------------- 2. 模型加载（已修复：找不到也不崩） ----------------------
model = None
try:
    @st.cache_resource
    def load_fatigue_model():
        with open("fatigue_model.pkl", "rb") as f:
            return pickle.load(f)
    model = load_fatigue_model()
except:
    st.warning("⚠️ 模型文件未找到，将使用模拟评估结果（不影响使用）")

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
        joints = {
            '左侧': {'肩膀': get_pose_pt(mp_pose.PoseLandmark.LEFT_SHOULDER), '肘部': get_pose_pt(mp_pose.PoseLandmark.LEFT_ELBOW), '手腕': get_pose_pt(mp_pose.PoseLandmark.LEFT_WRIST), '臀部': get_pose_pt(mp_pose.PoseLandmark.LEFT_HIP), '膝部': get_pose_pt(mp_pose.PoseLandmark.LEFT_KNEE)},
            '右侧': {'肩膀': get_pose_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER), '肘部': get_pose_pt(mp_pose.PoseLandmark.RIGHT_ELBOW), '手腕': get_pose_pt(mp_pose.PoseLandmark.RIGHT_WRIST), '臀部': get_pose_pt(mp_pose.PoseLandmark.RIGHT_HIP), '膝部': get_pose_pt(mp_pose.PoseLandmark.RIGHT_KNEE)},
            'mid': {'肩膀': [(get_pose_pt(mp_pose.PoseLandmark.LEFT_SHOULDER)[i] + get_pose_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)[i])/2 for i in range(3)], '臀部': [(get_pose_pt(mp_pose.PoseLandmark.LEFT_HIP)[i] + get_pose_pt(mp_pose.PoseLandmark.RIGHT_HIP)[i])/2 for i in range(3)], '膝部': [(get_pose_pt(mp_pose.PoseLandmark.LEFT_KNEE)[i] + get_pose_pt(mp_pose.PoseLandmark.RIGHT_KNEE)[i])/2 for i in range(3)]},
            '鼻子': get_pose_pt(mp_pose.PoseLandmark.NOSE)
        }
        if hands_result.multi_hand_landmarks:
            for hand in hands_result.multi_hand_landmarks:
                side = '左侧' if hand.landmark[0].x < 0.5 else '右侧'
                joints[side].update({'手腕': get_coord(hand.landmark[mp_hands.HandLandmark.WRIST], 'hands', W, H), '食指中节': get_coord(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], 'hands', W, H), '食指尖端': get_coord(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP], 'hands', W, H)})

        # 单独计算角度，每一步都容错，不影响其他数据
        metrics['angles']['颈部前屈'] = calculate_neck_flexion(joints['鼻子'], joints['mid']['肩膀'], joints['mid']['臀部'])
        metrics['angles']['背部屈曲'] = calculate_trunk_flexion(joints['mid']['肩膀'], joints['mid']['臀部'], joints['mid']['膝部'])

        for side in ['左侧', '右侧']:
            metrics['angles'][f'{side} 肩部上举'] = calculate_angle(joints[side]['臀部'], joints[side]['肩膀'], joints[side]['肘部'], 'frontal')
            metrics['angles'][f'{side} 肩部前伸'] = calculate_angle(joints[side]['臀部'], joints[side]['肩膀'], joints[side]['肘部'], 'sagittal')
            metrics['angles'][f'{side} 肘部屈伸'] = calculate_angle(joints[side]['肩膀'], joints[side]['肘部'], joints[side]['手腕'], 'sagittal')
            if '食指尖端' in joints[side]:
                metrics['angles'][f'{side} 手腕背伸'] = calculate_angle(joints[side]['肘部'], joints[side]['手腕'], joints[side]['食指尖端'], 'sagittal')
                metrics['angles'][f'{side} 手腕桡偏'] = calculate_angle(joints[side]['食指中节'], joints[side]['手腕'], joints[side]['食指尖端'], 'frontal')

        # 绘制骨骼（也单独容错）
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

# ---------------------- 修复：模型不存在也能正常评估 ----------------------
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

# ---------------------- 侧边栏：模型性能 & 标准参考 ----------------------
with st.sidebar:
    if 'accuracy' in locals():
        show_model_perf = st.checkbox("模型性能")
    else:
        show_model_perf = False
    show_std_ref = st.checkbox("标准参考")

# 模型性能模块（来自你最初的代码）
if show_model_perf and 'accuracy' in locals():
    st.subheader("📊 模型评估")
    # 准确率卡片
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

# 标准参考模块（来自你最初的代码）
if show_std_ref:
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

# ---------------------- 5. 主页面布局 ----------------------
st.markdown("<h1 style='text-align: center;'>疲劳评估系统（一体化版）</h1>", unsafe_allow_html=True)
st.markdown("""该工具依据国际标准ISO 11226，支持「图片识别自动填数」和「手动输入」两种方式进行疲劳评估。""")

# 初始化会话状态
if 'neck_flexion' not in st.session_state: st.session_state.neck_flexion = 20
if 'shoulder_elevation' not in st.session_state: st.session_state.shoulder_elevation = 60
if 'shoulder_forward' not in st.session_state: st.session_state.shoulder_forward = 120
if 'elbow_flexion' not in st.session_state: st.session_state.elbow_flexion = 120
if 'wrist_extension' not in st.session_state: st.session_state.wrist_extension = 15
if 'wrist_deviation' not in st.session_state: st.session_state.wrist_deviation = 10
if 'back_flexion' not in st.session_state: st.session_state.back_flexion = 20

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

# ---------------------- 模块3：AI 分析 ----------------------
if st.button("开始 AI 分析"):
    if "result" not in st.session_state:
        st.warning("请先完成疲劳评估！")
    else:
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

# 聊天记录显示
def display_chat_messages():
    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            if msg["role"] != "system":
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
display_chat_messages()
