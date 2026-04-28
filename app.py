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

# --- 2. 自定义 CSS (增加量子对撞对比框样式) ---
def inject_ui_css():
    # 根据步骤动态切换背景颜色
    bg_style = """
        background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
        overflow: hidden;
    """ if st.session_state.step == 2 else "background: #f8f9fa;"

    st.markdown(f"""
    <style>
        #MainMenu, footer {{ visibility: hidden; }}
        .stApp {{ {bg_style} }}
        [data-testid="stSidebar"] {{ min-width: 320px !important; }}

        /* 结果页面：霓虹发光卡片 */
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

        /* 数据库对比：对撞机样式 */
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
    </style>
    """, unsafe_allow_html=True)

inject_ui_css()

# --- 3. 数据与模型加载 ---
MODEL_FILES = {
    "带隙 (Bandgap)": {"model_path": "贝叶斯优化的HSE带隙预测模型.tar", "param_path": "贝叶斯优化的HSE带隙预测模型相关参数.json", "unit": "eV"},
    "晶格常数 (Lattice)": {"model_path": "贝叶斯优化的HSE晶格预测模型.tar", "param_path": "贝叶斯优化的HSE晶格预测模型参数.json", "unit": "Å"}
}

@st.cache_data
def load_target_database():
    try:
        # 修改点：文件名更新为 predict.csv
        return pd.read_csv("predict.csv")
    except Exception as e:
        return None

@st.cache_resource
def get_predictors():
    return {k: SinglePredictor(v["model_path"], v["param_path"], "atom_init.json") for k, v in MODEL_FILES.items()}

target_db = load_target_database()
predictors = get_predictors()

# --- 4. 侧边栏：参数展示 ---
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
        disp_v = f"{v:.4g}" if isinstance(v, float) else v
        param_html += f"<div style='display:flex; justify-content:space-between; margin-bottom:8px;'><span>{name}</span><strong>{disp_v}</strong></div>"
    st.markdown(f"<div style='background:rgba(0,0,0,0.05); padding:15px; border-radius:10px;'>{param_html}</div>", unsafe_allow_html=True)

# --- 5. 页面逻辑 ---

# 步骤 1：上传区
if st.session_state.step == 1:
    st.markdown("<br><br><h1 style='text-align:center;'>CGCNN 晶体性质预测平台</h1>", unsafe_allow_html=True)
    _, col_mid, _ = st.columns([1, 1.5, 1])
    
    with col_mid:
        uploaded_file = st.file_uploader("上传 POSCAR 文件", type=None)
        if uploaded_file:
            st.session_state.uploaded_file_name = uploaded_file.name
            # 统一命名为 POSCAR 供后端读取
            with open("POSCAR", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ 结构 {uploaded_file.name} 已载入")
            if st.button("🚀 启动量子推演", use_container_width=True):
                with st.spinner("正在执行图卷积运算..."):
                    res = predictors[target_prop].predict("POSCAR")
                    st.session_state.prediction_result = res
                    st.session_state.step = 2
                    st.rerun()

# 步骤 2：沉浸展示 + 对比对比
else:
    if st.button("← 返回首页"):
        st.session_state.step = 1
        st.rerun()

    col_view, col_res = st.columns([1.2, 0.8], gap="large")

    with col_view:
        st.markdown("<h3 style='text-align:center; color:white;'>🔮 晶体空间拓扑态</h3>", unsafe_allow_html=True)
        # 3D 高亮展示
        struct = Structure.from_file("POSCAR")
        view = py3Dmol.view(width=800, height=700)
        view.addModel(struct.to(fmt="cif"), 'cif')
        view.setStyle({'sphere': {'colorscheme': 'Jmol', 'scale': 0.35}, 'stick': {'color': '#ffffff', 'radius': 0.15}})
        view.addUnitCell({'color': '#00f2fe'})
        view.zoomTo()
        view.setBackgroundColor('#000000', 0)
        showmol(view, height=700, width=800)

    with col_res:
        name_only = target_prop.split(" ")[0]
        val = st.session_state.prediction_result
        unit = MODEL_FILES[target_prop]["unit"]
        
        # 核心结果卡片
        st.markdown(f"""
            <div class="neon-result-box">
                <div class="neon-label">PREDICTED {name_only}</div>
                <div class="neon-value">{val:.4f}<span style="font-size:2rem; color:#a29bfe;"> {unit}</span></div>
            </div>
        """, unsafe_allow_html=True)

        # 数据库“对撞”对比逻辑
        search_id = st.session_state.uploaded_file_name.split('.')[0] # 获取文件名（如 POSCAR8）
        if target_db is not None:
            # 匹配 Name 列
            match = target_db[target_db['Name'].astype(str).str.strip() == search_id.strip()]
            if not match.empty:
                # 确定要对比的列名
                db_col = 'Bandgap' if "Bandgap" in target_prop else 'Lattice'
                true_val = match[db_col].values[0]
                # 计算置信度 (1 - 相对误差)
                acc = max(0, 100 - (abs(val-true_val)/true_val*100)) if true_val != 0 else 0
                
                st.markdown(f"""
                    <div class="collision-box">
                        <div style="color:#00f2fe; font-size:0.9rem; letter-spacing:2px; font-weight:bold;">✦ 发现数据库匹配项: {search_id} ✦</div>
                        <div class="collision-grid">
                            <div style="text-align:center;">
                                <small style="color:rgba(255,255,255,0.5);">模型预测</small>
                                <div style="font-size:1.6rem; color:#00f2fe; font-weight:bold;">{val:.4f}</div>
                            </div>
                            <div class="vs-circle">VS</div>
                            <div style="text-align:center;">
                                <small style="color:rgba(255,255,255,0.5);">实验/目标值</small>
                                <div style="font-size:1.6rem; color:#a29bfe; font-weight:bold;">{true_val:.4f}</div>
                            </div>
                        </div>
                        <div class="accuracy-bar"><div class="accuracy-fill" style="width:{acc}%;"></div></div>
                        <div style="display:flex; justify-content:space-between; margin-top:10px; font-size:0.85rem; color:#00ff88;">
                            <span>绝对误差: {abs(val-true_val):.4f}</span>
                            <span>预测精度: {acc:.1f}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
