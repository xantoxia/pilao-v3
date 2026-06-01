import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import streamlit as st
from matplotlib import font_manager
import os
from openai import OpenAI
import base64
import requests
import datetime
import io
import pytz


GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]  # 从 Streamlit secrets 中获取 GitHub 令牌
GITHUB_USERNAME = 'xantoxia'  # 替换为你的 GitHub 用户名
GITHUB_REPO = 'blank-app-1'  # 替换为你的 GitHub 仓库名
GITHUB_BRANCH = 'main'  # 要上传的分支
FILE_PATH = 'fatigue_data.csv'  # 文件路径

# 获取文件内容，指定编码为utf-8，避免UnicodeDecodeError
def get_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return ""  # 如果文件不存在，返回空字符串
    except UnicodeDecodeError:
        st.error("文件编码错误，无法解码文件。")
        return None

# 获取文件的 SHA 值
def get_file_sha(file_path):
    url = f'https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{file_path}'
    headers = {'Authorization': f'token {GITHUB_TOKEN}'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        file_info = response.json()
        return file_info['sha']  # 返回SHA值
    else:
        st.warning(f"无法从 GitHub 获取文件: {response.json()}")
        return None

def save_to_csv(input_data, result, body_fatigue, cognitive_fatigue, emotional_fatigue):
    # 计算各问题的得分
    body_fatigue_score = calculate_score(body_fatigue)
    cognitive_fatigue_score = calculate_score(cognitive_fatigue)
    emotional_fatigue_score = calculate_score(emotional_fatigue)
    
    # 获取当前时间戳
    tz = pytz.timezone('Asia/Shanghai')
    timestamp = datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

    # 数据字典，包括评分和其他输入
    data = {
        "颈部前屈": int(input_data["颈部前屈"].values[0]),
        "颈部后仰": int(input_data["颈部后仰"].values[0]),
        "肩部上举范围": int(input_data["肩部上举范围"].values[0]),
        "肩部前伸范围": int(input_data["肩部前伸范围"].values[0]),
        "肘部屈伸": int(input_data["肘部屈伸"].values[0]),
        "手腕背伸": int(input_data["手腕背伸"].values[0]),
        "手腕桡偏/尺偏": int(input_data["手腕桡偏/尺偏"].values[0]),
        "背部屈曲范围": int(input_data["背部屈曲范围"].values[0]),
        "持续时间": int(input_data["持续时间"].values[0]),
        "重复频率": int(input_data["重复频率"].values[0]),
        "fatigue_result": result,
        "body_fatigue_score": body_fatigue_score,  # 添加评分
        "cognitive_fatigue_score": cognitive_fatigue_score,  # 添加评分
        "emotional_fatigue_score": emotional_fatigue_score,  # 添加评分
        "timestamp": timestamp  # 增加时间戳
    }
    df = pd.DataFrame([data])

    # 检查文件是否存在
    if os.path.exists(FILE_PATH):
        existing_content = get_file_content(FILE_PATH)
        
        # 如果文件内容非空，读取数据
        if existing_content and existing_content.strip():
            existing_df = pd.read_csv(io.StringIO(existing_content))
        else:
            # 如果文件为空，初始化空的 DataFrame
            existing_df = pd.DataFrame(columns=['timestamp', '颈部前屈', '颈部后仰', '肩部上举范围', '肩部前伸范围', '肘部屈伸', '手腕背伸', '手腕桡偏/尺偏', '背部屈曲范围', '持续时间', '重复频率', 'fatigue_result', 'body_fatigue_score', 'cognitive_fatigue_score', 'emotional_fatigue_score'])
    else:
        # 文件不存在，初始化空的 DataFrame
        existing_df = pd.DataFrame(columns=['timestamp', '颈部前屈', '颈部后仰', '肩部上举范围', '肩部前伸范围', '肘部屈伸', '手腕背伸', '手腕桡偏/尺偏', '背部屈曲范围', '持续时间', '重复频率', 'fatigue_result', 'body_fatigue_score', 'cognitive_fatigue_score', 'emotional_fatigue_score'])

    # 合并现有的 DataFrame 和新数据
    updated_df = pd.concat([existing_df, df], ignore_index=True)

    # 保存更新后的 DataFrame 到 CSV 文件
    updated_df.to_csv(FILE_PATH, index=False)
# 上传到 GitHub
def upload_to_github(file_path):
    # 获取文件的 SHA 值
    sha_value = get_file_sha(file_path)
    
    # 读取 CSV 文件内容并进行 base64 编码
    with open(file_path, 'rb') as file:
        content = base64.b64encode(file.read()).decode()

    # GitHub API 请求 URL
    url = f'https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPO}/contents/{file_path}'

    # 提交的信息
    commit_message = "Add new fatigue data with timestamp"
    
    data = {
        "message": commit_message,
        "branch": GITHUB_BRANCH,
        "content": content,
    }
    
    # 如果文件已经存在，提供 sha 值
    if sha_value:
        data["sha"] = sha_value

    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
    }

    response = requests.put(url, json=data, headers=headers)

    # 输出详细错误信息
    if response.status_code != 200 and response.status_code != 201:
        st.error(f"Failed to upload CSV file to GitHub: {response.json()}")
        print(f"GitHub API Response: {response.json()}")
def calculate_score(answer):
    if answer == '请选择':
        return 0  # 未选择时，得分为 0
    elif answer == '完全没有':
        return 1
    elif answer == '偶尔':
        return 2
    elif answer == '经常':
        return 3
    else:  # 总是
        return 4
        
font_path = "SourceHanSansCN-Normal.otf"  # 替换为你的上传字体文件名

# 检查字体文件是否存在
if not os.path.exists(font_path):
    st.error(f"Font file not found: {font_path}")
else:
    # 设置字体属性
    font_prop = font_manager.FontProperties(fname=font_path)
    font_name = font_prop.get_name()


    # 创建自定义函数来统一设置字体
    def set_font_properties(ax, font_prop):
        """统一设置坐标轴和标题字体"""
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_prop)
        ax.title.set_fontproperties(font_prop)
        ax.xaxis.label.set_fontproperties(font_prop)
        ax.yaxis.label.set_fontproperties(font_prop)


    # 全局设置字体
    plt.rcParams['font.sans-serif'] = [font_name]
    plt.rcParams['axes.unicode_minus'] = False

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

# 在 Streamlit 中展示
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

model = load_model()
# Streamlit sidebar
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

# 使用 Markdown 居中标题
st.markdown("<h1 style='text-align: center;'>疲劳评估测试系统</h1>", unsafe_allow_html=True)
st.markdown("""该工具依据国际标准ISO 11226（静态工作姿势）、美国国家职业安全健康研究所的《手动材料处理指南》以及OWAS分析与建议等多套国际标准和规范，对工作过程中的疲劳状态进行科学评估。""")

# 初始化存储所有预测记录的列表
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
st.subheader("角度参数")
# Two-column layout for sliders
col1, col2 = st.columns(2)

with col1:
    neck_flexion = st.slider("颈部前屈", 0, 60, 20)
    neck_extension = st.slider("颈部后仰", 0, 60, 25)
    shoulder_elevation = st.slider("肩部上举范围", 0, 180, 60)
    shoulder_forward = st.slider("肩部前伸范围", 0, 180, 120)

with col2:
    elbow_flexion = st.slider("肘部屈伸", 0, 180, 120)
    wrist_extension = st.slider("手腕背伸", 0, 60, 15)
    wrist_deviation = st.slider("手腕桡偏/尺偏", 0, 30, 10)
    back_flexion = st.slider("背部屈曲范围", 0, 60, 20)

# Task parameters
st.subheader("时间参数")
col3, col4 = st.columns(2)
with col3:
    task_duration = st.number_input("持续时间（秒）", min_value=0, value=5)
with col4:
    movement_frequency = st.number_input("重复频率（每5分钟）", min_value=0, value=35)

# 初始化会话状态
if "show_ai_analysis" not in st.session_state:
    st.session_state.show_ai_analysis = False
if "api_key_entered" not in st.session_state:
    st.session_state.api_key_entered = False
if "API_KEY" not in st.session_state:
    st.session_state.API_KEY = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'client' not in st.session_state:
    st.session_state.client = None

    # 定义疲劳评估函数
def fatigue_prediction(input_data):
    prediction = model.predict(input_data)
    return ["低疲劳状态", "中疲劳状态", "高疲劳状态"][prediction[0]]

def call_ark_api(client, messages):
    try:
        ark_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        completion = client.chat.completions.create(
            model="Pro/deepseek-ai/DeepSeek-V3.2",
            messages=ark_messages,
            stream=True
        )
        response = ""
        for chunk in completion:
            # 关键修复：先判断 choices 列表是否为空，再访问索引 0
            if chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                if hasattr(choice, "delta") and hasattr(choice.delta, "content") and choice.delta.content is not None:
                    yield choice.delta.content
            else:
                # 如果 chunk 没有 choices，跳过处理，不返回任何内容
                continue
    except Exception as e:
        st.error(f"调用 Ark API 时出错：{str(e)}")
        yield f"Error: {e}"

# 输入数据表格
input_data = pd.DataFrame({
    "颈部前屈": [neck_flexion],
    "颈部后仰": [neck_extension],
    "肩部上举范围": [shoulder_elevation],
    "肩部前伸范围": [shoulder_forward],
    "肘部屈伸": [elbow_flexion],
    "手腕背伸": [wrist_extension],
    "手腕桡偏/尺偏": [wrist_deviation],
    "背部屈曲范围": [back_flexion],
    "持续时间": [task_duration],
    "重复频率": [movement_frequency],
})
st.subheader("参数信息")
st.write(input_data)

st.subheader("主观感受")
# 使用 columns 来并列显示问题
col1, col2, col3 = st.columns(3)

# 问题1：身体疲劳
with col1:
    body_fatigue = st.selectbox(
        "1. 身体感到无力",
        ['请选择', '完全没有', '偶尔', '经常', '总是'],
        index=0  # 初始状态为未选择（'请选择'）
    )

# 问题2：注意力集中困难
with col2:
    cognitive_fatigue = st.selectbox(
        "2. 影响睡眠",
        ['请选择', '完全没有', '偶尔', '经常', '总是'],
        index=0  # 初始状态为未选择（'请选择'）
    )

# 问题3：情绪疲劳
with col3:
    emotional_fatigue = st.selectbox(
        "3. 肌肉酸痛或不适",
        ['请选择', '完全没有', '偶尔', '经常', '总是'],
        index=0  # 初始状态为未选择（'请选择'）
    )

# 根据选项得分
def calculate_score(answer):
    if answer == '请选择':
        return 0  # 未选择时，得分为 0
    elif answer == '完全没有':
        return 1
    elif answer == '偶尔':
        return 2
    elif answer == '经常':
        return 3
    else:  # 总是
        return 4

if st.button("评估"):
    # 如果用户未选择所有问题，则提示
    if body_fatigue == '请选择' or cognitive_fatigue == '请选择' or emotional_fatigue == '请选择':
        st.warning("请先选择所有问题的答案！")
    else:
        # 计算总得分
        score = calculate_score(body_fatigue) + calculate_score(cognitive_fatigue) + calculate_score(emotional_fatigue)
        # 请确保 fatigue_prediction 函数已定义
        result = fatigue_prediction(input_data)
        st.success(f"评估结果：{result}")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 保存数据到本地 CSV 文件
        save_to_csv(input_data, result, body_fatigue, cognitive_fatigue, emotional_fatigue)
        upload_to_github(FILE_PATH)
        # 保存评估结果到会话状态
        st.session_state.result = result
        record = input_data.copy()
        record["评估"] = result
        st.session_state.predictions.append(record)

        # 重置 AI 分析相关的会话状态
        st.session_state.ai_analysis_result = None
        st.session_state.messages = []
        st.session_state.show_ai_analysis = True
        # 不再要求用户输入API密钥
        st.session_state.api_key_entered = False
        if 'API_KEY' in st.session_state:
            del st.session_state.API_KEY
        if 'client' in st.session_state:
            del st.session_state.client  # 删除旧的 Ark 客户端
            
# 显示所有保存的预测记录
if st.session_state.predictions:
    st.subheader("所有评估记录")
    # 将所有记录合并成一个大DataFrame
    prediction_df = pd.concat(st.session_state.predictions, ignore_index=True)
    st.write(prediction_df)
    
# ====================== 【修复版】AI 分析功能 ======================
if st.button("开始 AI 分析"):
    st.session_state.show_ai_analysis = True

if st.session_state.get("show_ai_analysis", False):
    st.subheader("AI 分析")
    st.info("生成潜在人因危害分析及改善建议：")

    # 直接从 secrets 读取 API Key（无需用户输入）
    try:
        API_KEY = st.secrets["API_KEY"]
        client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")
        st.session_state.client = client
        st.session_state.api_key_entered = True
    except Exception as e:
        st.error(f"API 初始化失败：{str(e)}")
        st.stop()

    # 必须先评估
    if "result" not in st.session_state:
        st.warning("请先点击【评估】按钮进行疲劳评估！")
        st.stop()

    # 只生成一次分析
    if st.session_state.get("ai_analysis_result") is None:
        with st.spinner("正在进行 AI 分析..."):
            try:
                ai_input = f"""
用户目前 {body_fatigue} 身体感到无力，{cognitive_fatigue} 影响睡眠，{emotional_fatigue} 肌肉酸痛或不适。
工作关节角度：
颈部前屈 {neck_flexion}°，颈部后仰 {neck_extension}°，
肩部上举 {shoulder_elevation}°，肩部前伸 {shoulder_forward}°，
肘部屈伸 {elbow_flexion}°，手腕背伸 {wrist_extension}°，
手腕桡偏/尺偏 {wrist_deviation}°，背部屈曲 {back_flexion}°。

请基于人因工程国际标准（ISO 11226）分析潜在风险，并给出优先改善建议。
简洁、专业、有依据。
"""

                messages = [
                    {"role": "system", "content": "你是专业人因工程专家，依据ISO 11226等国际标准提供建议。"},
                    {"role": "user", "content": ai_input}
                ]

                # 调用 AI
                full_response = ""
                for chunk in call_ark_api(st.session_state.client, messages):
                    if chunk:
                        full_response += chunk

                st.session_state.ai_analysis_result = full_response
                st.session_state.messages = messages + [
                    {"role": "assistant", "content": full_response}
                ]

            except Exception as e:
                st.error(f"AI 分析调用失败：{str(e)}")

    # 显示分析结果
    if st.session_state.get("ai_analysis_result"):
        st.markdown("### 📝 分析结果")
        st.success(st.session_state.ai_analysis_result)

# ------------------- 聊天对话 -------------------
if st.session_state.get("api_key_entered") and st.session_state.get("client"):
    prompt = st.chat_input("继续咨询人因工程问题：")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("思考中..."):
            try:
                full_response = ""
                for chunk in call_ark_api(st.session_state.client, st.session_state.messages):
                    full_response += chunk
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"对话出错：{str(e)}")

# 显示聊天记录
def display_chat_messages():
    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            if msg["role"] != "system":
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

display_chat_messages()
