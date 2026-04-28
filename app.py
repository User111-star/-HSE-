import streamlit as st
import os
import json
import pandas as pd
import py3Dmol
import re
import subprocess
import sys
import time
import zipfile
import shutil
from stmol import showmol
from pymatgen.core.structure import Structure
from predict_api import SinglePredictor

# === 画图与数学计算所需的依赖包 ===
import numpy as np   
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- 1. 页面基本配置 ---
st.set_page_config(
    page_title="HSE 晶体性质预测与训练平台",
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
        .train-card { background: white; border-radius: 10px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 4px solid #8e44ad; margin-bottom: 20px;}
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
    try:
        df = pd.read_csv("predict.csv")
        df.columns = df.columns.str.strip()
        if 'index_label' in df.columns:
            df['index_label'] = df['index_label'].astype(str).str.strip()
        return df
    except Exception: 
        return pd.DataFrame()

@st.cache_resource
def load_all_models():
    models = {}
    for key, info in MODEL_FILES.items():
        try:
            models[key] = SinglePredictor(info["model_path"], info["param_path"], ATOM_INIT_PATH)
        except Exception: 
            models[key] = None
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

@st.cache_data
def draw_scatter_plot(csv_path, target_type):
    """二合一画图逻辑：严格根据带隙或晶格调整坐标范围和单位"""
    df_test = pd.read_csv(csv_path, header=None, names=['Index', 'Calculated', 'Predicted'])
    
    # 清洗非数字行
    df_test['Calculated'] = pd.to_numeric(df_test['Calculated'], errors='coerce')
    df_test['Predicted'] = pd.to_numeric(df_test['Predicted'], errors='coerce')
    df_test = df_test.dropna()

    y_true = df_test['Calculated'].values
    y_pred = df_test['Predicted'].values

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    z = np.abs(y_true - y_pred) / np.sqrt(2)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect('equal', adjustable='box')
    scatter = ax.scatter(y_true, y_pred, c=z, cmap='viridis', s=30, alpha=0.8, edgecolor='none')

    min_val, max_val = np.min(y_true), np.max(y_true)

    if "带隙" in target_type:
        x_y_min, x_y_max = -0.1, max_val + 0.2
        unit_str = " eV"
        ax.set_xlabel(r'Calculated $E_{g\_HSE}$ (eV)', fontsize=16)
        ax.set_ylabel(r'Predicted $E_{g\_HSE}$ (eV)', fontsize=16)
        text_x = 0.40
    else:  # 晶格常数
        x_y_min, x_y_max = min_val - 0.1, max_val + 0.2
        unit_str = ""
        ax.set_xlabel(r'Calculated a_HSE (Å)', fontsize=16)
        ax.set_ylabel(r'Predicted a_HSE (Å)', fontsize=16)
        text_x = 0.50

    ax.plot([x_y_min, x_y_max], [x_y_min, x_y_max], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([x_y_min, x_y_max])
    ax.set_ylim([x_y_min, x_y_max])
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(scatter, cax=cax)
    scatter.set_clim(vmin=0, vmax=np.max(z))

    textstr = '\n'.join((r'$R^2=%.3f$' % r2, r'MAE$=%.3f$%s' % (mae, unit_str), r'RMSE$=%.3f$%s' % (rmse, unit_str)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none')
    ax.text(text_x, 0.05, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='bottom', bbox=props)
    
    fig.tight_layout()
    return fig

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
        target_prop = st.radio("🎯 选择预测目标：", list(MODEL_FILES.keys()), index=0)
        st.markdown("### 🧬 当前模型超参数")
        try:
            with open(MODEL_FILES[target_prop]['param_path'], 'r', encoding='utf-8') as f:
                params = json.load(f)
            p_html = "".join([f"<div style='display:flex; justify-content:space-between; margin-bottom:8px;'><span>{k}</span><strong>{v:.4g if isinstance(v, float) else v}</strong></div>" for k,v in params.items()])
            st.markdown(f"<div class='param-box'>{p_html}</div>", unsafe_allow_html=True)
        except Exception: st.warning("未找到参数文件")
    else:
        train_target = st.radio("🎯 选择训练任务：", list(MODEL_FILES.keys()), index=0)
        st.success(f"已加载【{train_target}】的超参数")

st.markdown("<div class='main-title'>CGCNN 晶体性质智能平台</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>基于图神经网络与贝叶斯优化的第一性原理精度替代模型</div>", unsafe_allow_html=True)

# --- 6. 预测模式界面 ---
if app_mode == "🔮 模型预测模式":
    col_l, _, col_r = st.columns([1.2, 0.1, 1])
    with col_l:
        st.markdown("### 📥 数据输入区")
        uploaded_file = st.file_uploader("上传 POSCAR 文件")
        if uploaded_file:
            with open("POSCAR", "wb") as f: f.write(uploaded_file.getbuffer())
            showmol(render_crystal("POSCAR"), height=500, width=600)

    with col_r:
        if uploaded_file and models[target_prop]:
            if st.button("🚀 启动预测", use_container_width=True):
                res = models[target_prop].predict("POSCAR")
                unit = MODEL_FILES[target_prop]["unit"]
                st.markdown(f'<div class="result-card"><h3>预测结果</h3><h1 style="color:#2e86c1;">{res:.4f} {unit}</h1></div>', unsafe_allow_html=True)

# --- 7. 训练模式界面 ---
elif app_mode == "⚙️ 模型训练模式":
    col_t1, col_t2 = st.columns([1.5, 1])
    with col_t1:
        st.markdown("<div class='train-card'>### 📁 1. 上传数据集")
        uploaded_dataset = st.file_uploader("上传数据集 ZIP", type="zip")
        st.markdown("### 📊 2. 划分比例")
        c1, c2, c3 = st.columns(3)
        train_r = c1.number_input("训练集", 0.0, 1.0, 0.8)
        val_r = c2.number_input("验证集", 0.0, 1.0, 0.1)
        test_r = c3.number_input("测试集", 0.0, 1.0, 0.1)
        valid = round(train_r + val_r + test_r, 2) == 1.0
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 启动模型训练", use_container_width=True, disabled=not (valid and uploaded_dataset)):
            run_id = str(int(time.time()))
            temp_root, data_dir = f"temp_run_{run_id}", f"temp_dataset_{run_id}"
            os.makedirs(temp_root, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)
            
            try:
                zip_path = os.path.join(temp_root, "data.zip")
                with open(zip_path, "wb") as f: f.write(uploaded_dataset.getbuffer())
                with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(temp_root)
                
                real_folder = None
                for r, d, files in os.walk(temp_root):
                    if any(f.endswith('.csv') for f in files) and any("POSCAR" in f for f in files):
                        real_folder = r
                        target_csv_name = [f for f in files if f.endswith('.csv')][0]
                        break
                
                for item in os.listdir(real_folder):
                    shutil.copy2(os.path.join(real_folder, item), os.path.join(data_dir, "id_prop.csv" if item == target_csv_name else item))

                with open(MODEL_FILES[train_target]['param_path'], 'r', encoding='utf-8') as f:
                    params = json.load(f)

                # === 核心逻辑修改：根据你的 Excel 截图精准传参 ===
                # 第3列是带隙 -> Index 2, 第5列是晶格 -> Index 4
                target_idx = 2 if "Bandgap" in train_target else 4

                cmd = [
                    sys.executable, "main.py", data_dir,
                    "--epochs", str(params.get("epochs", 50) or 50),
                    "--batch-size", str(params.get("batch_size", 256) or 256),
                    "--lr", str(params.get("lr", 0.01) or 0.01),
                    "--train-ratio", str(train_r),
                    "--val-ratio", str(val_r),
                    "--test-ratio", str(test_r),
                    "--n-conv", str(params.get("n_conv", 3) or 3),
                    "--atom-fea-len", str(params.get("atom_fea_len", 64) or 64),
                    "--print-freq", "1",
                    "--target", str(target_idx)  # 这里传入正确的 2 或 4
                ]
                
                progress_bar = st.progress(0)
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                
                for line in process.stdout:
                    full_match = re.search(r"Epoch:\s*\[(\d+)\]\[(\d+)/(\d+)\]", line)
                    if full_match:
                        prog = min((int(full_match.group(1)) + int(full_match.group(2))/int(full_match.group(3))) / int(params.get("epochs", 50)), 1.0)
                        progress_bar.progress(prog)
                    st.code(line.strip())
                
                process.wait()
                if process.returncode == 0:
                    st.success("🎉 训练圆满完成！")
                    result_csv = [f for f in os.listdir(".") if "test_results" in f and run_id in f][0]
                    st.pyplot(draw_scatter_plot(result_csv, train_target))
                    for f in os.listdir("."):
                        if run_id in f: st.download_button(f"📥 下载 {f}", open(f, "rb"), file_name=f)
            finally:
                shutil.rmtree(temp_root, ignore_errors=True)
                shutil.rmtree(data_dir, ignore_errors=True)

    with col_t2:
        st.markdown(f"<div class='train-card'>### 🧬 {train_target} 超参数预览")
        try:
            with open(MODEL_FILES[train_target]['param_path'], 'r', encoding='utf-8') as f:
                st.json(json.load(f))
        except Exception: st.warning("加载失败")
        st.markdown("</div>", unsafe_allow_html=True)
