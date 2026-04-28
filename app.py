import streamlit as st
import os
import json
import py3Dmol
import pandas as pd
import time
from stmol import showmol
from pymatgen.core.structure import Structure
from predict_api import SinglePredictor

# --- 1. 页面基本配置 ---
st.set_page_config(
    page_title="HSE 量子晶体预测实验室",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 Session State
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = ""

# --- 2. 核心 CSS 注入 (量子星云背景 + 霓虹卡片 + 对撞对比框) ---
def inject_ui_css():
    # 动态背景：步骤2使用沉浸式星空
    bg_style = """
        background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
        overflow: hidden;
    """ if st.session_state.step == 2 else "background: #f8f9fa;"

    st.markdown(f"""
    <style>
        /* 隐藏默认 UI，保留 Header 以免侧边栏消失 */
        #MainMenu, footer {{ visibility: hidden; }}
        .stApp {{ {bg_style} }}
        
        /* 侧边栏宽度优化 */
        [data-testid="stSidebar"] {{ min-width: 350px !important; }}

        /* 动画：星云呼吸感 */
        @keyframes nebula-breathe {{
            0% {{ opacity: 0.4; transform: scale(1); }}
            50% {{ opacity: 0.7; transform: scale(1.1); }}
            100% {{ opacity: 0.4; transform: scale(1); }}
        }}
        .nebula-bg {{
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            background: 
                radial-gradient(circle at 20% 30%, rgba(79, 172, 254, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 80% 70%, rgba(142, 68, 173, 0.15) 0%, transparent 50%);
            animation: nebula-breathe 8s infinite ease-in-out;
            z-index: -1;
        }}

        /* 预测结果：霓虹发光卡片 */
        .neon-result-box {{
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(15px);
            border-radius: 24px;
            padding: 40px;
            border: 1px solid rgba(0, 242, 254, 0.2);
            box-shadow: 0 0 50px rgba(0, 242, 254, 0.1);
            text-align: right;
            margin-top: 5vh;
        }}
        .neon-value {{
            font-size: 6rem;
            font-weight: 900;
            color: #00f2fe;
            text-shadow: 0 0 20px rgba(0, 242, 254, 0.8);
            line-height: 1.1;
        }}
        .neon-label {{ color: #bdc3c7; letter-spacing: 5px; text-transform: uppercase; font-size: 1rem; }}

        /* 量子对撞对比框 (可视化增强) */
        .collision-box {{
            background: rgba(0, 242, 254, 0.08);
            border: 1px dashed rgba(0, 242, 254, 0.4);
            border-radius: 20px;
            padding: 25px;
            margin-top: 25px;
        }}
        .collision-grid {{ display: flex; justify-content: space-between; align-items: center; margin: 15px 0; }}
        .vs-circle {{
            width: 45px; height: 45px; border-radius: 50%; background: #00f2fe; color: #000;
            display: flex; align-items: center; justify-content: center; font-weight: 900;
            box-shadow: 0 0 20px #00f2fe;
        }}
        .accuracy-bar {{ height: 5px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden; }}
        .accuracy-fill {{ height: 100%; background: linear-gradient(to right, #4facfe, #00ff88); transition: width 2s ease-out; }}
        
        .spotlight {{
            background: radial-gradient(circle at center, rgba(0, 242, 254, 0.2) 0%, transparent 70%);
            border-radius: 50%;
        }}
    </style>
    """, unsafe_allow_html=True)

inject_ui_css()

# --- 3. 数据库与模型加载 ---
MODEL_FILES = {
    "带隙 (Bandgap)": {"model_path": "贝叶斯优化的HSE带隙预测模型.tar", "param_path": "贝叶斯优化的HSE带隙预测模型相关参数.json", "unit": "eV"},
    "晶格常数 (Lattice)": {"model_path": "贝叶斯优化的HSE晶格预测模型.tar", "param_path": "贝叶斯优化的HSE晶格预测模型参数.json", "unit": "Å"}
}

@st.cache_data
def load_target_database():
    try:
        # 读取上传的 predict.csv 文件
        return pd.read_csv("predict.csv")
    except:
        return None

@st.cache_resource
def get_predictors():
    return {k: SinglePredictor(v["model_path"], v["param_path"], "atom_init.json") for k, v in MODEL_FILES.items()}

target_db = load_target_database()
predictors = get_predictors()

# --- 4. 侧边栏：参数科学计数法展示 ---
with st.sidebar:
    st.markdown("## ⚙️ 预测配置中枢")
    target_prop = st.radio("选择预测目标", list(MODEL_FILES.keys()), disabled=(st.session_state.step == 2))
    
    st.markdown("---")
    st.markdown("### 📊 贝叶斯最优参数")
    with open(MODEL_FILES[target_prop]['param_path'], 'r') as f:
        params = json.load(f)
    
    icons = {"lr": "⚡ 学习率", "n_conv": "🔄 卷积层", "atom_fea_len": "🧬 原子特征", "batch_size": "📦 批大小", "h_fea_len": "🧠 隐藏层维度"}
    
    param_html = ""
    for k, v in params.items():
        name = icons.get(k, k)
        # 使用科学计数法格式化显示长浮动数
        disp_v = f"{v:.4g}" if isinstance(v, float) else v
        param_html += f"<div style='display:flex; justify-content:space-between; margin-bottom:8px;'><span>{name}</span><strong>{disp_v}</strong></div>"
    st.markdown(f"<div style='background:rgba(0,0,0,0.05); padding:15px; border-radius:10px;'>{param_html}</div>", unsafe_allow_html=True)

# --- 5. 页面流程渲染 ---

# 【步骤 1：输入与计算】
if st.session_state.step == 1:
    st.markdown("<br><br><h1 style='text-align:center;'>CGCNN 晶体性质预测平台</h1>", unsafe_allow_html=True)
    _, col_mid, _ = st.columns([1, 1.5, 1])
    
    with col_mid:
        uploaded_file = st.file_uploader("上传 POSCAR 文件", type=None)
        if uploaded_file:
            st.session_state.uploaded_file_name = uploaded_file.name
            # 🚨 强制重命名为 POSCAR，解决 pymatgen 读取兼容性
            with open("POSCAR", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ 结构 {uploaded_file.name} 已就绪")
            if st.button("🚀 启动量子推演", use_container_width=True):
                with st.spinner("正在执行图卷积运算..."):
                    res = predictors[target_prop].predict("POSCAR")
                    st.session_state.prediction_result = res
                    st.session_state.step = 2
                    st.rerun()

# 【步骤 2：沉浸式结果 + 对撞机对比】
else:
    st.markdown('<div class="nebula-bg"></div>', unsafe_allow_html=True)
    if st.button("← 返回实验室"):
        st.session_state.step = 1
        st.rerun()

    col_view, col_res = st.columns([1.2, 0.8], gap="large")

    with col_view:
        st.markdown("<h3 style='text-align:center; color:white;'>🔮 晶体空间拓扑态</h3>", unsafe_allow_html=True)
        st.markdown("<div class='spotlight'>", unsafe_allow_html=True)
        # 3D 渲染：背景透明，高亮度原子
        struct = Structure.from_file("POSCAR")
        view = py3Dmol.view(width=800, height=700)
        view.addModel(struct.to(fmt="cif"), 'cif')
        view.setStyle({'sphere': {'colorscheme': 'Jmol', 'scale': 0.35}, 'stick': {'color': '#ffffff', 'radius': 0.15}})
        view.addUnitCell({'color': '#00f2fe'})
        view.zoomTo()
        view.setBackgroundColor('#000000', 0)
        showmol(view, height=700, width=800)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_res:
        name_only = target_prop.split(" ")[0]
        val = st.session_state.prediction_result
        unit = MODEL_FILES[target_prop]["unit"]
        
        # 霓虹结果显示
        st.markdown(f"""
            <div class="neon-result-box">
                <div class="neon-label">PREDICTED {name_only}</div>
                <div class="neon-value">{val:.4f}<span style="font-size:2rem; color:#a29bfe;"> {unit}</span></div>
            </div>
        """, unsafe_allow_html=True)

        # 核心逻辑：自动从 predict.csv 中检索对比
        search_id = st.session_state.uploaded_file_name.split('.')[0].strip()
        if target_db is not None:
            # 搜索 Name 列是否存在匹配
            match = target_db[target_db['Name'].astype(str).str.strip() == search_id]
            if not match.empty:
                # 获取对应的目标值
                db_col = 'Bandgap' if "Bandgap" in target_prop else 'Lattice'
                true_val = match[db_col].values[0]
                
                # 计算置信度 (1 - 相对误差)
                acc = max(0, 100 - (abs(val-true_val)/true_val*100)) if true_val != 0 else 0
                
                # 展示量子对比对撞机
                st.markdown(f"""
                    <div class="collision-box">
                        <div style="color:#00f2fe; font-size:0.9rem; letter-spacing:2px; font-weight:bold;">✦ 发现数据库匹配: {search_id} ✦</div>
                        <div class="collision-grid">
                            <div style="text-align:center;">
                                <small style="color:rgba(255,255,255,0.5);">模型预测</small>
                                <div style="font-size:1.6rem; color:#00f2fe; font-weight:bold;">{val:.4f}</div>
                            </div>
                            <div class="vs-circle">VS</div>
                            <div style="text-align:center;">
                                <small style="color:rgba(255,255,255,0.5);">目标值 (HSE)</small>
                                <div style="font-size:1.6rem; color:#a29bfe; font-weight:bold;">{true_val:.4f}</div>
                            </div>
                        </div>
                        <div class="accuracy-bar"><div class="accuracy-fill" style="width:{acc}%;"></div></div>
                        <div style="display:flex; justify-content:space-between; margin-top:10px; font-size:0.85rem; color:#00ff88;">
                            <span>误差: {abs(val-true_val):.4f}</span>
                            <span>模型置信度: {acc:.1f}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
