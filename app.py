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
    page_title="HSE 晶体性质预测与训练平台",
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
        .viewer-container { display: flex; justify-content: center; align-items: center; width: 100%; margin-top: 15px; }
        .train-card { background: white; border-radius: 10px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 4px solid #8e44ad; margin-bottom: 20px;}
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
    try:
        df = pd.read_csv("predict.csv")
        if 'index_label' in df.columns:
            df['index_label'] = df['index_label'].astype(str).str.strip()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_resource
def load_all_models():
    models = {}
    try:
        bg_info = MODEL_FILES["带隙 (Bandgap)"]
        models["带隙 (Bandgap)"] = SinglePredictor(bg_info["model_path"], bg_info["param_path"], ATOM_INIT_PATH)
    except:
        models["带隙 (Bandgap)"] = None
    try:
        la_info = MODEL_FILES["晶格常数 (Lattice)"]
        models["晶格常数 (Lattice)"] = SinglePredictor(la_info["model_path"], la_info["param_path"], ATOM_INIT_PATH)
    except:
        models["晶格常数 (Lattice)"] = None
    return models

def render_crystal(poscar_path):
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
models = load_all_models()

# --- 5. 侧边栏导航 ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2000/2000885.png", width=80)
    st.markdown("## 🧭 系统导航")
    
    app_mode = st.radio("选择运行模式：", ["🔮 模型预测模式", "⚙️ 模型训练模式"], index=0)
    
    st.markdown("---")
    
    if app_mode == "🔮 模型预测模式":
        st.markdown("### 🎛️ 预测配置")
        target_prop = st.radio("🎯 选择预测目标：", ["带隙 (Bandgap)", "晶格常数 (Lattice)"], index=0)
        
        # 【恢复】：展示详细超参数
        st.markdown("### 🧬 当前模型超参数")
        try:
            with open(MODEL_FILES[target_prop]['param_path'], 'r', encoding='utf-8') as f:
                params = json.load(f)
            p_html = ""
            for k, v in params.items():
                display_v = f"{v:.4g}" if isinstance(v, float) else v
                p_html += f"<div style='display:flex; justify-content:space-between; margin-bottom:8px;'><span>{k}</span><strong>{display_v}</strong></div>"
            st.markdown(f"<div class='param-box'>{p_html}</div>", unsafe_allow_html=True)
        except: st.warning("未找到参数文件")
        
    else:
        st.markdown("### 🎛️ 训练任务选择")
        train_target = st.radio("🎯 选择训练目标：", ["带隙 (Bandgap)", "晶格常数 (Lattice)"], index=0)
        st.success(f"已加载 {train_target} 的预设超参数")

st.markdown("<div class='main-title'>CGCNN 晶体性质智能平台</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>基于图神经网络与贝叶斯优化的第一性原理精度替代模型</div>", unsafe_allow_html=True)

# --- 6. 主界面：模型预测 ---
if app_mode == "🔮 模型预测模式":
    col_l, _, col_r = st.columns([1.2, 0.1, 1])
    with col_l:
        st.markdown("### 📥 数据输入区")
        uploaded_file = st.file_uploader("上传 POSCAR 文件", help="系统将自动识别文件名数字ID")
        if uploaded_file:
            with open("POSCAR", "wb") as f: f.write(uploaded_file.getbuffer())
            st.success(f"✅ 文件 [{uploaded_file.name}] 解析就绪")
            st.markdown("<h4 style='text-align: center; margin-top: 20px;'>🧊 晶体结构三维预览</h4>", unsafe_allow_html=True)
            try:
                st.markdown("<div class='viewer-container'>", unsafe_allow_html=True)
                showmol(render_crystal("POSCAR"), height=500, width=600)
                st.markdown("</div>", unsafe_allow_html=True)
            except: pass

    with col_r:
        st.markdown("### 📊 运算结果区")
        if models[target_prop] is None:
            st.error("❌ 缺少模型权重文件，请先前往训练模式。")
        elif not uploaded_file:
            st.info("等待上传 POSCAR 数据...")
        else:
            if st.button("🚀 启动前向传播预测", use_container_width=True):
                with st.spinner("计算中..."):
                    try:
                        res = models[target_prop].predict("POSCAR")
                        unit, icon = MODEL_FILES[target_prop]["unit"], MODEL_FILES[target_prop]["icon"]
                        error_html = ""
                        if not db_df.empty:
                            nums = re.findall(r'\d+', uploaded_file.name)
                            file_id = nums[0] if nums else "UNKNOWN"
                            match = db_df[db_df['index_label'] == file_id]
                            if not match.empty:
                                col = 'Gap' if "Bandgap" in target_prop else 'lattice'
                                t_val = float(match.iloc[0][col])
                                abs_err = abs(res - t_val)
                                rel_err = (abs_err / t_val * 100) if t_val != 0 else 0
                                # 顶格写防止 Markdown 渲染错误
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
</div>"""
                        st.markdown(f"""
<div class="result-card">
    <div class="result-label">{icon} 目标性质: {target_prop.split(" ")[0]}</div>
    <div class="result-value">{res:.4f} <span style="font-size: 1.5rem; color:#7f8c8d;">{unit}</span></div>
    <div style="color: #27ae60; font-weight: 500; margin-bottom: 10px;">✓ 预测成功</div>
    {error_html}
</div>""", unsafe_allow_html=True)
                    except Exception as e: st.error(f"预测失败: {e}")

# --- 7. 主界面：模型训练 ---
elif app_mode == "⚙️ 模型训练模式":
    col_t1, col_t2 = st.columns([1.5, 1])
    with col_t1:
        st.markdown("<div class='train-card'>", unsafe_allow_html=True)
        st.markdown("### 📁 1. 路径设置")
        dataset_path = st.text_input("数据集路径：", value="./dataset")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='train-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 2. 数据划分比例")
        c1, c2, c3 = st.columns(3)
        train_r = c1.number_input("训练集 (Train)", 0.0, 1.0, 0.8, 0.05)
        val_r = c2.number_input("验证集 (Val)", 0.0, 1.0, 0.1, 0.05)
        test_r = c3.number_input("测试集 (Test)", 0.0, 1.0, 0.1, 0.05)
        
        total_ratio = train_r + val_r + test_r
        if round(total_ratio, 2) != 1.0:
            st.error(f"⚠️ 当前比例总和为 {total_ratio:.2f}，请确保总和等于 1.0")
        else:
            st.success("✅ 比例分配有效")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 开始训练模型", use_container_width=True, disabled=(round(total_ratio, 2) != 1.0)):
            # 这里将在后续接入 main.py 的实际调用
            st.info("任务已启动，正在初始化训练环境...")
            progress_bar = st.progress(0)
            status = st.empty()
            # 模拟进度
            import time
            for i in range(101):
                time.sleep(0.01)
                progress_bar.progress(i)
                status.text(f"训练进度: {i}% (正在同步训练 Loss ...)")
            st.success("🎉 训练任务圆满完成！")

    with col_t2:
        st.markdown("<div class='train-card'>", unsafe_allow_html=True)
        st.markdown("### 🧬 自动加载的超参数")
        try:
            with open(MODEL_FILES[train_target]['param_path'], 'r', encoding='utf-8') as f:
                st.json(json.load(f))
        except: st.error("无法加载对应的 JSON 参数")
        st.markdown("</div>", unsafe_allow_html=True)
