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

# --- 3. 路径配置 ---
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

# --- 4. 辅助函数 ---
@st.cache_data
def load_database():
    try:
        df = pd.read_csv("predict.csv")
        if 'index_label' in df.columns:
            df['index_label'] = df['index_label'].astype(str).str.strip()
        return df
    except: return pd.DataFrame()

@st.cache_resource
def load_all_models():
    models = {}
    for key, info in MODEL_FILES.items():
        try:
            models[key] = SinglePredictor(info["model_path"], info["param_path"], ATOM_INIT_PATH)
        except: models[key] = None
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

db_df = load_database()
models = load_all_models()

# --- 5. 侧边栏 ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2000/2000885.png", width=80)
    st.markdown("## 🧭 系统导航")
    app_mode = st.radio("选择运行模式：", ["🔮 模型预测模式", "⚙️ 模型训练模式"], index=0)
    st.markdown("---")
    
    if app_mode == "🔮 模型预测模式":
        target_prop = st.radio("🎯 选择预测目标：", list(MODEL_FILES.keys()), index=0)
        st.markdown("### 🧬 当前使用的超参数")
        try:
            with open(MODEL_FILES[target_prop]['param_path'], 'r', encoding='utf-8') as f:
                params = json.load(f)
            p_html = "".join([f"<div style='display:flex; justify-content:space-between; margin-bottom:8px;'><span>{k}</span><strong>{v:.4g if isinstance(v, float) else v}</strong></div>" for k,v in params.items()])
            st.markdown(f"<div class='param-box'>{p_html}</div>", unsafe_allow_html=True)
        except: pass
    else:
        train_target = st.radio("🎯 选择训练任务：", list(MODEL_FILES.keys()), index=0)

st.markdown("<div class='main-title'>CGCNN 晶体性质智能平台</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>基于图神经网络与贝叶斯优化的第一性原理精度替代模型</div>", unsafe_allow_html=True)

# --- 6. 预测模式界面 ---
if app_mode == "🔮 模型预测模式":
    col_l, _, col_r = st.columns([1.2, 0.1, 1])
    with col_l:
        st.markdown("### 📥 数据输入区")
        uploaded_file = st.file_uploader("上传 POSCAR 文件", help="支持 VASP POSCAR 格式")
        if uploaded_file:
            with open("POSCAR", "wb") as f: f.write(uploaded_file.getbuffer())
            st.success(f"✅ 文件 [{uploaded_file.name}] 已就绪")
            st.markdown("<h4 style='text-align: center; margin-top: 20px;'>🧊 晶体结构预览</h4>", unsafe_allow_html=True)
            try:
                st.markdown("<div class='viewer-container'>", unsafe_allow_html=True)
                showmol(render_crystal("POSCAR"), height=500, width=600)
                st.markdown("</div>", unsafe_allow_html=True)
            except: pass

    with col_r:
        st.markdown("### 📊 运算结果区")
        if models[target_prop] is None:
            st.error(f"❌ 未检测到 {target_prop} 的模型权重，请先前往训练模式。")
        elif not uploaded_file:
            st.info("等待上传数据...")
        else:
            if st.button("🚀 启动预测", use_container_width=True):
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
                                abs_err, rel_err = abs(res - t_val), (abs(res - t_val)/t_val*100 if t_val != 0 else 0)
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
                    except Exception as e: st.error(f"❌ 预测出错: {e}")

# --- 7. 训练模式界面 ---
elif app_mode == "⚙️ 模型训练模式":
    col_t1, col_t2 = st.columns([1.5, 1])
    with col_t1:
        st.markdown("<div class='train-card'>", unsafe_allow_html=True)
        st.markdown("### 📁 1. 数据集准备")
        
        # 核心更新：使用文件上传器接受 ZIP
        uploaded_dataset = st.file_uploader("上传数据集压缩包 (.zip)", type="zip", help="压缩包内需包含所有 POSCAR 文件及对应的 id_prop.csv")
        
        st.markdown("### 📊 2. 数据划分比例")
        c1, c2, c3 = st.columns(3)
        train_r = c1.number_input("训练集", 0.0, 1.0, 0.8, 0.01)
        val_r = c2.number_input("验证集", 0.0, 1.0, 0.1, 0.01)
        test_r = c3.number_input("测试集", 0.0, 1.0, 0.1, 0.01)
        
        valid = round(train_r + val_r + test_r, 2) == 1.0
        if not valid: st.error(f"⚠️ 比例总和必须为 1.0 (当前: {train_r+val_r+test_r:.2f})")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 启动 CGCNN 训练任务", use_container_width=True, disabled=not (valid and uploaded_dataset)):
            st.info("📦 正在初始化云端临时环境，解压数据集...")
            
            # --- 自动解压与过渡文件夹构建机制 ---
            temp_extract_dir = "temp_extract_dir_cgcnn"
            final_data_dir = "temp_dataset"  # 纯净的无斜杠目录名，防止 main.py 保存权重时路径冲突
            
            # 确保启动前清理残留
            for d in [temp_extract_dir, final_data_dir]:
                if os.path.exists(d): shutil.rmtree(d, ignore_errors=True)
                
            os.makedirs(temp_extract_dir, exist_ok=True)
            os.makedirs(final_data_dir, exist_ok=True)
            
            try:
                # 1. 保存压缩包并解压
                zip_path = os.path.join(temp_extract_dir, "upload_data.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_dataset.getbuffer())
                    
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(temp_extract_dir)
                    
                # 2. 智能寻路：寻找包含 id_prop.csv 的真实数据根目录
                target_folder = None
                for root, dirs, files in os.walk(temp_extract_dir):
                    if "id_prop.csv" in files:
                        target_folder = root
                        break
                        
                if target_folder is None:
                    st.error("❌ 解压失败：在压缩包内未找到 `id_prop.csv` 文件，请检查格式！")
                    shutil.rmtree(temp_extract_dir, ignore_errors=True)
                    shutil.rmtree(final_data_dir, ignore_errors=True)
                    st.stop()
                    
                # 3. 将真实数据迁移到平铺的 final_data_dir 目录
                for item in os.listdir(target_folder):
                    s = os.path.join(target_folder, item)
                    d = os.path.join(final_data_dir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d)
                    else:
                        shutil.copy2(s, d)
                        
                st.success("✅ 数据集解压并挂载成功！正在调起训练进程...")
                
                # 4. 组装 main.py 的命令并调用
                with open(MODEL_FILES[train_target]['param_path'], 'r') as f:
                    params = json.load(f)
                
                cmd = [
                    sys.executable, "main.py", final_data_dir,
                    "--epochs", str(params.get("epochs", 50)),
                    "--batch-size", str(params.get("batch_size", 256)),
                    "--lr", str(params.get("lr", 0.01)),
                    "--train-ratio", str(train_r),
                    "--val-ratio", str(val_r),
                    "--test-ratio", str(test_r),
                    "--atom-fea-len", str(params.get("atom_fea_len", 64)),
                    "--n-conv", str(params.get("n_conv", 3)),
                    "--h-fea-len", str(params.get("h_fea_len", 128)),
                    "--n-h", str(params.get("n_h", 1))
                ]
                
                progress_bar = st.progress(0)
                log_area = st.empty()
                max_epochs = params.get("epochs", 50)
                
                # 异步执行，实时捕获日志
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                
                for line in process.stdout:
                    # 抓取控制台输出里的 Epoch: [数字]
                    epoch_match = re.search(r"Epoch: \[(\d+)\]", line)
                    if epoch_match:
                        current_epoch = int(epoch_match.group(1))
                        progress_bar.progress(min(current_epoch / max_epochs, 1.0))
                    
                    log_area.code(f"日志流: {line.strip()}")
                
                process.wait()
                
                if process.returncode == 0:
                    st.success("🎉 训练圆满完成！最佳权重已更新。")
                else:
                    st.error("❌ 训练异常终止，请查看上方日志。")
                    
            except Exception as e:
                st.error(f"启动失败: {e}")
            finally:
                # 5. 阅后即焚：无论成功失败，销毁所有上百兆的临时数据集文件！
                log_area.code(f"日志流: 训练任务结束，正在清理临时虚拟文件夹...")
                time.sleep(1)
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
                shutil.rmtree(final_data_dir, ignore_errors=True)
                log_area.code(f"日志流: 清理完成，系统环境已恢复洁净。")

    with col_t2:
        st.markdown("<div class='train-card'>", unsafe_allow_html=True)
        st.markdown("### 🧬 自动加载的超参数")
        try:
            with open(MODEL_FILES[train_target]['param_path'], 'r') as f:
                st.json(json.load(f))
        except: st.warning("未找到参数文件")
        st.markdown("</div>", unsafe_allow_html=True)
