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
            text-align: center; margin-bottom: 0rem;
        }
        .sub-title { font-size: 1.2rem; color: #7f8c8d; text-align: center; margin-bottom: 2rem; }
        .result-card {
            background: linear-gradient(135deg, #ffffff 0%, #f1f8ff 100%);
            border-radius: 15px; padding: 30px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.05);
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
        .viewer-container { display: flex; justify-content: center; align-items: center; width: 100%; margin-top: 15px; }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- 3. 配置文件路径 ---
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
    """加载预测数据库并确保路径正确"""
    db_path = "predict.csv"
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(db_path)
        # 强制将 index_label 转为字符串，方便后续匹配
        if 'index_label' in df.columns:
            df['index_label'] = df['index_label'].astype(str)
        return df
    except:
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

# --- 数据预加载 ---
db_df = load_database()
models = load_all_models()

# --- 5. 侧边栏 ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2000/2000885.png", width=80)
    st.markdown("## 🎛️ 预测配置中枢")
    
    # 数据库状态检查 (调试用)
    if db_df.empty:
        st.error("❌ 未检测到 predict.csv")
    else:
        st.success(f"✅ 数据库已就绪 ({len(db_df)} 条记录)")

    target_property = st.radio("🎯 选择预测目标：", ["带隙 (Bandgap)", "晶格常数 (Lattice)"], index=0)
    st.markdown("---")
    st.markdown("### 🧬 贝叶斯最优超参数")
    try:
        with open(MODEL_FILES[target_property]['param_path'], 'r', encoding='utf-8') as f:
            best_params = json.load(f)
        param_html = "".join([f"<div style='display:flex; justify-content:space-between; margin-bottom:8px;'><span>{k}</span><strong>{v:.4g if isinstance(v, float) else v}</strong></div>" for k, v in best_params.items()])
        st.markdown(f"<div class='param-box'>{param_html}</div>", unsafe_allow_html=True)
    except: pass
    st.caption("🚀 Powered by CGCNN & Bayesian Optimization")

# --- 6. 主界面 ---
st.markdown("<div class='main-title'>CGCNN 晶体性质智能预测平台</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>基于图神经网络与贝叶斯优化的第一性原理精度替代模型</div>", unsafe_allow_html=True)

col_left, _, col_right = st.columns([1.2, 0.1, 1])

with col_left:
    st.markdown("### 📥 数据输入区")
    uploaded_file = st.file_uploader("上传 POSCAR 文件", help="支持 VASP POSCAR 格式")

    selected_material = None
    if uploaded_file:
        temp_path = "POSCAR"
        with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())

        # 精确匹配 ID 逻辑
        if not db_df.empty:
            material_list = db_df['material'].tolist()
            file_id_str = os.path.splitext(uploaded_file.name)[0] # "1.POSCAR" -> "1"
            
            # 在 index_label 列中寻找匹配
            match_row = db_df[db_df['index_label'] == file_id_str]
            default_idx = material_list.index(match_row.iloc[0]['material']) if not match_row.empty else 0
            
            if not match_row.empty:
                st.success(f"🔍 已识别 ID: **{file_id_str}** -> **{match_row.iloc[0]['material']}**")
            
            selected_material = st.selectbox("确认材料名称：", options=material_list, index=default_idx)
        
        st.markdown("<h4 style='text-align: center; margin-top: 20px;'>🧊 晶体结构预览</h4>", unsafe_allow_html=True)
        showmol(render_crystal_structure(temp_path), height=500, width=600)

with col_right:
    st.markdown("### 📊 运算结果区")
    if uploaded_file:
        if st.button("🚀 启动前向传播预测"):
            with st.spinner("计算中..."):
                try:
                    res = models[target_property].predict("POSCAR")
                    unit = MODEL_FILES[target_property]["unit"]
                    prop_name = target_property.split(" ")[0]
                    
                    # 关键的比对 HTML 生成
                    error_html = ""
                    if not db_df.empty and selected_material:
                        col_name = 'Gap' if "Bandgap" in target_property else 'lattice'
                        true_val = db_df[db_df['material'] == selected_material][col_name].values[0]
                        abs_err = abs(res - true_val)
                        rel_err = (abs_err / true_val * 100) if true_val != 0 else 0
                        
                        # 只有这里逻辑触发，才会显示对比
                        error_html = f"""
                        <div style="display: flex; justify-content: space-around; margin-top: 20px; border-top: 2px solid #ecf0f1; padding-top: 20px;">
                            <div>
                                <div style="font-size: 0.9rem; color: #7f8c8d;">📊 数据库真实值</div>
                                <div style="font-size: 1.6rem; font-weight: 700; color: #2980b9;">{true_val:.4f} {unit}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; color: #7f8c8d;">📉 预测误差</div>
                                <div style="font-size: 1.6rem; font-weight: 700; color: #e74c3c;">{abs_err:.4f} {unit}</div>
                                <div style="font-size: 0.8rem; color: #e74c3c;">(相对误差: {rel_err:.2f}%)</div>
                            </div>
                        </div>
                        """
                    
                    st.markdown(f"""
                        <div class="result-card">
                            <div class="result-label">{MODEL_FILES[target_property]['icon']} {prop_name}</div>
                            <div class="result-value">{res:.4f} <span style="font-size: 1.5rem; color:#7f8c8d;">{unit}</span></div>
                            <div style="color: #27ae60; font-weight: 500;">✓ 预测成功</div>
                            {error_html}
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"预测失败: {e}")
    else:
        st.info("请先在左侧上传 POSCAR 文件。")
