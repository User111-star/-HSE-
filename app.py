import streamlit as st
import os
import json
import pandas as pd
import py3Dmol
from stmol import showmol
from pymatgen.core.structure import Structure
from predict_api import SinglePredictor

# --- 1. 页面基本配置 ---
st.set_page_config(
    page_title="HSE 晶体性质预测平台",
    layout="wide",  
    initial_sidebar_state="expanded",
    page_icon="🧊"
)

# --- 2. 自定义 CSS 注入 ---
def local_css():
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stApp { background-color: #f8f9fa; }
        .main-title {
            font-size: 3rem; font-weight: 800;
            background: -webkit-linear-gradient(45deg, #2e86c1, #8e44ad);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0rem; text-align: center;
        }
        .sub-title { font-size: 1.2rem; color: #7f8c8d; text-align: center; margin-bottom: 2rem; }
        .result-card {
            background: linear-gradient(135deg, #ffffff 0%, #f1f8ff 100%);
            border-radius: 15px; padding: 30px; box-shadow: 0 10px 20px rgba(0,0,0,0.05);
            text-align: center; border-left: 6px solid #2e86c1; margin-top: 20px;
        }
        .result-value { font-size: 3.5rem; font-weight: 700; color: #2c3e50; margin: 10px 0; }
        .result-label { font-size: 1.2rem; color: #34495e; text-transform: uppercase; letter-spacing: 2px; }
        .param-box { background-color: rgba(255,255,255,0.1); border-radius: 8px; padding: 15px; border-left: 4px solid #8e44ad; margin-bottom: 15px; }
        .stButton>button {
            width: 100%; border-radius: 8px; height: 50px; font-size: 1.2rem; font-weight: 600;
            background: linear-gradient(to right, #3498db, #2980b9); color: white; border: none;
            box-shadow: 0 4px 6px rgba(52, 152, 219, 0.3); transition: all 0.3s ease;
        }
        .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(52, 152, 219, 0.4); color: white; }
        .viewer-container { display: flex; justify-content: center; align-items: center; width: 100%; margin-top: 15px; }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. 配置文件名及路径 ---
MODEL_FILES = {
    "带隙 (Bandgap)": {
        "model_path": "贝叶斯优化的HSE带隙预测模型.tar",
        "param_path": "贝叶斯优化的HSE带隙预测模型相关参数.json",
        "unit": "eV", "icon": "⚡"
    },
    "晶格常数 (Lattice)": {
        "model_path": "贝叶斯优化的HSE晶格预测模型.tar",
        "param_path": "贝叶斯优化的HSE晶格预测模型参数.json",
        "unit": "Å", "icon": "🧊"
    }
}
ATOM_INIT_PATH = "atom_init.json"

# --- 4. 核心加载函数 ---
@st.cache_data
def load_database():
    try:
        df = pd.read_csv("predict.csv")
        # 【关键灵感保留】：纯粹的字符串转换，并去除两端不可见空格
        if 'index_label' in df.columns:
            df['index_label'] = df['index_label'].astype(str).str.strip()
        if 'material' in df.columns:
            df['material'] = df['material'].astype(str).str.strip()
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_resource
def load_all_models():
    bg_info = MODEL_FILES["带隙 (Bandgap)"]
    bg_predictor = SinglePredictor(bg_info["model_path"], bg_info["param_path"], ATOM_INIT_PATH)
    la_info = MODEL_FILES["晶格常数 (Lattice)"]
    la_predictor = SinglePredictor(la_info["model_path"], la_info["param_path"], ATOM_INIT_PATH)
    return {"带隙 (Bandgap)": bg_predictor, "晶格常数 (Lattice)": la_predictor}

def render_crystal_structure(poscar_path):
    struct = Structure.from_file(poscar_path)
    cif_string = struct.to(fmt="cif")
    view = py3Dmol.view(width=600, height=500)
    view.addModel(cif_string, 'cif')
    view.setStyle({'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}, 'stick': {'colorscheme': 'Jmol', 'radius': 0.1}})
    view.addUnitCell()
    view.zoomTo()
    return view

# 提前加载依赖
db_df = load_database()
try:
    with st.spinner("🔄 正在初始化深度学习引擎..."):
        models = load_all_models()
except Exception as e:
    st.error(f"❌ 模型初始化失败！错误详情: {e}")
    st.stop()

# --- 5. 侧边栏 ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2000/2000885.png", width=80)
    st.markdown("## 🎛️ 预测配置中枢")
    st.markdown("---")
    target_property = st.radio("🎯 选择预测目标：", ["带隙 (Bandgap)", "晶格常数 (Lattice)"], index=0)
    st.markdown("---")
    st.markdown("### 🧬 贝叶斯最优超参数")
    try:
        with open(MODEL_FILES[target_property]['param_path'], 'r', encoding='utf-8') as f:
            best_params = json.load(f)
        param_html = ""
        for key, value in best_params.items():
            display_value = f"{value:.4g}" if isinstance(value, float) else value
            param_html += f"<div style='display:flex; justify-content:space-between; margin-bottom:8px;'><span>{key}</span><strong>{display_value}</strong></div>"
        st.markdown(f"<div class='param-box'>{param_html}</div>", unsafe_allow_html=True)
    except: pass

# --- 6. 主界面 ---
st.markdown("<div class='main-title'>CGCNN 晶体性质智能预测平台</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>基于图神经网络与贝叶斯优化的第一性原理精度替代模型</div>", unsafe_allow_html=True)

col_left, col_space, col_right = st.columns([1.2, 0.1, 1])

with col_left:
    st.markdown("### 📥 数据输入区")
    st.info("请上传标准格式的 `POSCAR` 文件 (文件名需包含对应的数字ID，如 '6.POSCAR')。")
    uploaded_file = st.file_uploader("", help="支持 VASP POSCAR 格式文件")

    if uploaded_file:
        temp_poscar_path = "POSCAR"
        with open(temp_poscar_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("✅ 文件上传并解析就绪。")
        st.markdown("<h4 style='text-align: center; margin-top: 20px;'>🧊 晶体结构三维交互预览</h4>", unsafe_allow_html=True)
        try:
            st.markdown("<div class='viewer-container'>", unsafe_allow_html=True)
            view = render_crystal_structure(temp_poscar_path)
            showmol(view, height=500, width=600)
            st.markdown("</div>", unsafe_allow_html=True)
        except: pass

with col_right:
    st.markdown("### 📊 运算结果区")

    if not uploaded_file:
        st.markdown(
            """
            <div style='text-align:center; padding: 50px; background-color:#f1f3f4; border-radius: 10px; color:#9aa0a6; border: 2px dashed #dadce0;'>
                <h4>等待上传 POSCAR 数据...</h4>
                <p>上传文件后，点击下方预测按钮即可获取高精度预测结果。</p>
            </div>
            """, unsafe_allow_html=True
        )

    if uploaded_file:
        if st.button("🚀 启动前向传播预测"):
            with st.spinner("🧠 正在提取图结构特征并进行张量运算..."):
                try:
                    predictor = models[target_property]
                    result_val = predictor.predict(temp_poscar_path)

                    unit = MODEL_FILES[target_property]["unit"]
                    icon = MODEL_FILES[target_property]["icon"]
                    prop_name = target_property.split(" ")[0]

                    # 2. 【核心匹配逻辑】：无 UI 选择框，纯字符串后台匹配
                    error_html = ""
                    if not db_df.empty:
                        # 从文件名中提取基础名称并清理空格（例如 "6.POSCAR" -> "6"）
                        file_base_name = str(os.path.splitext(uploaded_file.name)[0]).strip()
                        
                        # 直接拿字符串和数据库里同样转成了字符串的列进行无缝匹配
                        match_row = db_df[db_df['index_label'] == file_base_name]
                        
                        # 为了极高的容错性，如果没匹配上数字ID，就去匹配 material 列
                        if match_row.empty:
                            match_row = db_df[db_df['material'] == file_base_name]
                        
                        if not match_row.empty:
                            target_col = 'Gap' if "Bandgap" in target_property else 'lattice'
                            true_val = float(match_row.iloc[0][target_col])
                            
                            abs_error = abs(result_val - true_val)
                            rel_error = (abs_error / true_val) * 100 if true_val != 0 else 0
                            
                            error_html = f"""
                            <div style="display: flex; justify-content: space-around; margin-top: 20px; border-top: 2px solid #ecf0f1; padding-top: 20px;">
                                <div>
                                    <div style="font-size: 1rem; color: #7f8c8d; text-transform: uppercase;">📊 数据库真实值</div>
                                    <div style="font-size: 1.8rem; font-weight: 700; color: #2980b9;">{true_val:.4f} <span style="font-size: 1.2rem;">{unit}</span></div>
                                </div>
                                <div>
                                    <div style="font-size: 1rem; color: #7f8c8d; text-transform: uppercase;">📉 预测误差</div>
                                    <div style="font-size: 1.8rem; font-weight: 700; color: #e74c3c;">{abs_error:.4f} <span style="font-size: 1.2rem;">{unit}</span></div>
                                    <div style="font-size: 0.9rem; color: #e74c3c; font-weight:bold;">(相对误差: {rel_error:.2f}%)</div>
                                </div>
                            </div>
                            """

                    # 3. 最终 UI 渲染
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">{icon} 目标性质: {prop_name}</div>
                            <div class="result-value">{result_val:.4f} <span style="font-size: 1.5rem; color:#7f8c8d;">{unit}</span></div>
                            <div style="color: #27ae60; font-weight: 500; margin-bottom: 10px;">✓ 预测成功</div>
                            {error_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"❌ 运算出错: {e}")
