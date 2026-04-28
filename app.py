import streamlit as st
import os
import json
import pandas as pd
import py3Dmol
import re
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

# --- 2. 自定义 CSS 注入 (科技感 UI) ---
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
    """加载预测数据库并确保 ID 匹配鲁棒性"""
    try:
        df = pd.read_csv("predict.csv")
        # 统一转为去空格字符串，方便匹配
        if 'index_label' in df.columns:
            df['index_label'] = df['index_label'].astype(str).str.strip()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_resource
def load_all_models():
    """初始化双模型预测器"""
    bg_info = MODEL_FILES["带隙 (Bandgap)"]
    bg_p = SinglePredictor(bg_info["model_path"], bg_info["param_path"], ATOM_INIT_PATH)
    la_info = MODEL_FILES["晶格常数 (Lattice)"]
    la_p = SinglePredictor(la_info["model_path"], la_info["param_path"], ATOM_INIT_PATH)
    return {"带隙 (Bandgap)": bg_p, "晶格常数 (Lattice)": la_p}

def render_crystal(poscar_path):
    """3D 渲染晶体结构"""
    struct = Structure.from_file(poscar_path)
    cif_str = struct.to(fmt="cif")
    view = py3Dmol.view(width=600, height=500)
    view.addModel(cif_str, 'cif')
    view.setStyle({'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}, 'stick': {'colorscheme': 'Jmol', 'radius': 0.1}})
    view.addUnitCell()
    view.zoomTo()
    return view

# 预加载
db_df = load_database()
try:
    with st.spinner("🔄 深度学习引擎装载中..."):
        models = load_all_models()
except Exception as e:
    st.error(f"❌ 模型装载失败: {e}")
    st.stop()

# --- 5. 侧边栏 ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2000/2000885.png", width=80)
    st.markdown("## 🎛️ 预测配置中枢")
    st.markdown("---")
    target_prop = st.radio("🎯 选择预测目标：", ["带隙 (Bandgap)", "晶格常数 (Lattice)"], index=0)
    st.markdown("---")
    st.markdown("### 🧬 贝叶斯最优超参数")
    try:
        with open(MODEL_FILES[target_prop]['param_path'], 'r', encoding='utf-8') as f:
            params = json.load(f)
        p_html = "".join([f"<div style='display:flex; justify-content:space-between; margin-bottom:8px;'><span>{k}</span><strong>{v:.4g if isinstance(v, float) else v}</strong></div>" for k,v in params.items()])
        st.markdown(f"<div class='param-box'>{p_html}</div>", unsafe_allow_html=True)
    except: pass
    st.caption("🚀 Powered by CGCNN & Bayesian Optimization")

# --- 6. 主界面布局 ---
st.markdown("<div class='main-title'>CGCNN 晶体性质智能预测平台</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>基于图神经网络与贝叶斯优化的第一性原理精度替代模型</div>", unsafe_allow_html=True)

col_l, _, col_r = st.columns([1.2, 0.1, 1])

with col_l:
    st.markdown("### 📥 数据输入区")
    st.info("上传 POSCAR 文件（系统将自动根据文件名中的数字匹配数据库 ID）")
    uploaded_file = st.file_uploader("", help="支持 VASP POSCAR 格式")

    if uploaded_file:
        with open("POSCAR", "wb") as f: f.write(uploaded_file.getbuffer())
        st.success(f"✅ 文件 [{uploaded_file.name}] 解析就绪")
        
        st.markdown("<h4 style='text-align: center; margin-top: 20px;'>🧊 晶体结构三维预览</h4>", unsafe_allow_html=True)
        try:
            st.markdown("<div class='viewer-container'>", unsafe_allow_html=True)
            showmol(render_crystal("POSCAR"), height=500, width=600)
            st.markdown("</div>", unsafe_allow_html=True)
        except: st.warning("⚠️ 结构预览生成失败，但不影响预测。")

with col_r:
    st.markdown("### 📊 运算结果区")
    if not uploaded_file:
        st.markdown("<div style='text-align:center; padding: 50px; background-color:#f1f3f4; border-radius: 10px; color:#9aa0a6; border: 2px dashed #dadce0;'><h4>等待数据上传...</h4></div>", unsafe_allow_html=True)
    
    if uploaded_file:
        if st.button("🚀 启动前向传播预测"):
            with st.spinner("计算中..."):
                try:
                    # 1. 预测
                    res = models[target_prop].predict("POSCAR")
                    unit = MODEL_FILES[target_prop]["unit"]
                    icon = MODEL_FILES[target_prop]["icon"]
                    name = target_prop.split(" ")[0]

                    # 2. 全自动后台比对逻辑
                    error_html = ""
                    if not db_df.empty:
                        nums = re.findall(r'\d+', uploaded_file.name)
                        file_id = nums[0] if nums else "UNKNOWN"
                        
                        match = db_df[db_df['index_label'] == file_id]
                        if not match.empty:
                            col = 'Gap' if "Bandgap" in target_prop else 'lattice'
                            try:
                                t_val = float(match.iloc[0][col])
                                abs_err = abs(res - t_val)
                                rel_err = (abs_err / t_val * 100) if t_val != 0 else 0
                                
                                # 注意这里：HTML代码全都顶格写，防止被Markdown解析成代码块
                                error_html = f"""
<div style="display: flex; justify-content: space-around; margin-top: 20px; border-top: 2px solid #ecf0f1; padding-top: 20px;">
    <div>
        <div style="font-size: 1rem; color: #7f8c8d; text-transform: uppercase;">📊 数据库真实值 (ID:{file_id})</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #2980b9;">{t_val:.4f} <span style="font-size: 1.2rem;">{unit}</span></div>
    </div>
    <div>
        <div style="font-size: 1rem; color: #7f8c8d; text-transform: uppercase;">📉 预测误差</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #e74c3c;">{abs_err:.4f} <span style="font-size: 1.2rem;">{unit}</span></div>
        <div style="font-size: 0.9rem; color: #e74c3c; font-weight:bold;">(相对误差: {rel_err:.2f}%)</div>
    </div>
</div>
"""
                            except: 
                                error_html = "<div style='margin-top:15px; color:red;'>❌ 匹配成功但数据库数值格式有误。</div>"
                        else:
                            error_html = f"<div style='margin-top:15px; color:#f39c12;'>⚠️ 未在数据库发现 ID 为 {file_id} 的对比记录。</div>"

                    # 3. UI 渲染 (同样顶格写)
                    st.markdown(f"""
<div class="result-card">
    <div class="result-label">{icon} 目标性质: {name}</div>
    <div class="result-value">{res:.4f} <span style="font-size: 1.5rem; color:#7f8c8d;">{unit}</span></div>
    <div style="color: #27ae60; font-weight: 500; margin-bottom: 10px;">✓ 预测成功</div>
    {error_html}
</div>
""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ 预测出错: {e}")
