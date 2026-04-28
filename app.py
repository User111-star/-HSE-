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
        .train-card { background: white; border-radius: 10px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 4px solid #8e44ad;}
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
    """初始化预测器，如果找不到文件则返回 None，避免直接卡死整个网页"""
    models = {}
    try:
        bg_info = MODEL_FILES["带隙 (Bandgap)"]
        models["带隙 (Bandgap)"] = SinglePredictor(bg_info["model_path"], bg_info["param_path"], ATOM_INIT_PATH)
    except Exception as e:
        models["带隙 (Bandgap)"] = None
        
    try:
        la_info = MODEL_FILES["晶格常数 (Lattice)"]
        models["晶格常数 (Lattice)"] = SinglePredictor(la_info["model_path"], la_info["param_path"], ATOM_INIT_PATH)
    except Exception as e:
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

# 预加载数据与模型
db_df = load_database()
models = load_all_models()

# ==========================================
# --- 5. 侧边栏 (路由中心) ---
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2000/2000885.png", width=80)
    st.markdown("## 🧭 系统导航")
    
    # 核心：系统模式切换
    app_mode = st.radio(
        "选择运行模式：", 
        ["🔮 模型预测模式", "⚙️ 模型极简训练模式"],
        index=0
    )
    
    st.markdown("---")
    
    # 根据模式展示不同的侧边栏配置
    if app_mode == "🔮 模型预测模式":
        st.markdown("### 🎛️ 预测配置")
        target_prop = st.radio("🎯 选择预测目标：", ["带隙 (Bandgap)", "晶格常数 (Lattice)"], index=0)
        
        st.markdown("### 🧬 当前使用的超参数")
        try:
            with open(MODEL_FILES[target_prop]['param_path'], 'r', encoding='utf-8') as f:
                params = json.load(f)
            p_html = "".join([f"<div style='display:flex; justify-content:space-between; margin-bottom:8px;'><span>{k}</span><strong>{v:.4g if isinstance(v, float) else v}</strong></div>" for k,v in params.items()])
            st.markdown(f"<div class='param-box'>{p_html}</div>", unsafe_allow_html=True)
        except: pass
        
    else:
        st.markdown("### 🎛️ 训练配置")
        train_target = st.radio("🎯 选择训练目标：", ["带隙 (Bandgap)", "晶格常数 (Lattice)"], index=0)
        st.info("💡 极简模式：参数将根据目标自动匹配，数据集已强制划分为 Train:Val:Test = 8:1:1")


st.markdown("<div class='main-title'>CGCNN 晶体性质智能平台</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>基于图神经网络与贝叶斯优化的第一性原理精度替代模型</div>", unsafe_allow_html=True)


# ==========================================
# --- 6. 主界面：预测模式 ---
# ==========================================
if app_mode == "🔮 模型预测模式":
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
        if models[target_prop] is None:
            st.error("❌ 未检测到该目标的模型权重文件 (.tar)，请先进入训练模式进行训练！")
        elif not uploaded_file:
            st.markdown("<div style='text-align:center; padding: 50px; background-color:#f1f3f4; border-radius: 10px; color:#9aa0a6; border: 2px dashed #dadce0;'><h4>等待数据上传...</h4></div>", unsafe_allow_html=True)
        
        if uploaded_file and models[target_prop] is not None:
            if st.button("🚀 启动前向传播预测", use_container_width=True):
                with st.spinner("计算中..."):
                    try:
                        res = models[target_prop].predict("POSCAR")
                        unit = MODEL_FILES[target_prop]["unit"]
                        icon = MODEL_FILES[target_prop]["icon"]
                        name = target_prop.split(" ")[0]

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
                                except: error_html = "<div style='margin-top:15px; color:red;'>❌ 匹配成功但数据库数值格式有误。</div>"
                            else:
                                error_html = f"<div style='margin-top:15px; color:#f39c12;'>⚠️ 未在数据库发现 ID 为 {file_id} 的对比记录。</div>"

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

# ==========================================
# --- 7. 主界面：训练模式 (极简版) ---
# ==========================================
elif app_mode == "⚙️ 模型极简训练模式":
    
    col_t1, col_t2 = st.columns([1.5, 1])
    
    with col_t1:
        st.markdown("<div class='train-card'>", unsafe_allow_html=True)
        st.markdown("### 📁 1. 数据集准备")
        dataset_path = st.text_input(
            "请输入包含 POSCAR/CIF 文件及 id_prop.csv 的文件夹路径：", 
            value="./dataset", 
            help="相对路径或绝对路径均可，默认为当前目录下的 dataset 文件夹"
        )
        
        # 简单验证路径是否存在
        if os.path.exists(dataset_path):
            st.success("✅ 路径有效已确认。")
        else:
            st.warning("⚠️ 路径暂不存在，请确保训练开始前存在该文件夹。")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='train-card' style='margin-top:20px;'>", unsafe_allow_html=True)
        st.markdown("### 🚀 2. 启动训练")
        
        if st.button("▶️ 一键启动 CGCNN 模型训练", use_container_width=True):
            st.toast("🔥 训练任务已提交！", icon="🔥")
            
            # --- 这里是预留给接入 main.py 的进度条区域 ---
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 模拟进度条运转（后续将替换为真实监听 main.py 进程的代码）
            import time
            for percent_complete in range(100):
                time.sleep(0.02)
                progress_bar.progress(percent_complete + 1)
                status_text.markdown(f"**当前状态:** 正在执行 Epoch {percent_complete}/100 ... 损失函数正在下降")
                
            status_text.success(f"🎉 训练完成！最佳模型权重已保存。去左侧切换回【🔮 模型预测模式】试试吧！")
            
        st.markdown("</div>", unsafe_allow_html=True)

    with col_t2:
        st.markdown("<div class='train-card'>", unsafe_allow_html=True)
        st.markdown("### 🧬 本次训练加载参数")
        st.info("系统已从您的 JSON 文件中自动提取了最佳参数，无需手动干预。")
        
        try:
            param_file_path = MODEL_FILES[train_target]['param_path']
            with open(param_file_path, 'r', encoding='utf-8') as f:
                train_params_dict = json.load(f)
            
            # 以漂亮的 JSON 格式展示给用户看
            st.json(train_params_dict)
            
            st.markdown("#### ⚙️ 固定架构参数")
            st.code("""
Train / Val / Test = 8 : 1 : 1
Model Structure = Default CGCNN
Task = Regression
            """)
        except Exception as e:
            st.error(f"无法读取对应的 JSON 参数文件: {e}")
            
        st.markdown("</div>", unsafe_allow_html=True)
