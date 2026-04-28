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

# === 新增：画图所需的依赖包 ===
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
        # 暴力清洗列名空格，防止 KeyError
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

# === 新增：二合一的散点图绘制函数 ===
@st.cache_data
def draw_scatter_plot(csv_path, target_type):
    """根据测试结果 CSV 绘制散点图并返回 fig 对象"""
    # 1. 读取数据
    df_test = pd.read_csv(csv_path, header=None, names=['Index', 'Calculated', 'Predicted'])
    y_true = df_test['Calculated'].values
    y_pred = df_test['Predicted'].values

    # 2. 计算误差
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    z = np.abs(y_true - y_pred) / np.sqrt(2)

    # 3. 初始化画布
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect('equal', adjustable='box')
    scatter = ax.scatter(y_true, y_pred, c=z, cmap='viridis', s=30, alpha=0.8, edgecolor='none')

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    # 4. 根据目标类型动态切换 UI 设置
    if "带隙" in target_type:
        x_y_min = -0.1
        x_y_max = max_val + 0.2
        unit_str = " eV"
        ax.set_xlabel(r'Calculated $E_{g\_HSE}$ (eV)', fontsize=16)
        ax.set_ylabel(r'Predicted $E_{g\_HSE}$ (eV)', fontsize=16)
        text_x = 0.40
    else:  # 晶格常数
        x_y_min = min_val - 0.1
        x_y_max = max_val + 0.2
        unit_str = ""
        ax.set_xlabel(r'Calculated a_HSE (Å)', fontsize=16)
        ax.set_ylabel(r'Predicted a_HSE (Å)', fontsize=16)
        text_x = 0.50

    # 5. 画对角线和设置刻度
    ax.plot([x_y_min, x_y_max], [x_y_min, x_y_max], 'k--', lw=1, alpha=0.5)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(MaxNLocator(integer=False, nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=6))
    ax.set_xlim([x_y_min, x_y_max])
    ax.set_ylim([x_y_min, x_y_max])

    # 6. 颜色条设置
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    scatter.set_clim(vmin=0, vmax=np.max(z))
    cbar.locator = MaxNLocator(nbins=6)
    cbar.update_ticks()

    # 7. 文本框
    textstr = '\n'.join((
        r'$R^2=%.3f$' % (r2,),
        r'MAE$=%.3f$%s' % (mae, unit_str),
        r'RMSE$=%.3f$%s' % (rmse, unit_str)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none')
    ax.text(text_x, 0.05, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='bottom', bbox=props)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
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
        st.markdown("### 🎯 预测配置")
        target_prop = st.radio("选择预测目标：", list(MODEL_FILES.keys()), index=0)
        
        st.markdown("### 🧬 当前模型超参数")
        try:
            with open(MODEL_FILES[target_prop]['param_path'], 'r', encoding='utf-8') as f:
                params = json.load(f)
            p_html = ""
            for k, v in params.items():
                display_v = f"{v:.4g}" if isinstance(v, float) else v
                p_html += f"<div style='display:flex; justify-content:space-between; margin-bottom:8px;'><span>{k}</span><strong>{display_v}</strong></div>"
            st.markdown(f"<div class='param-box'>{p_html}</div>", unsafe_allow_html=True)
        except Exception: 
            st.warning("未找到参数文件")
            
    else:
        st.markdown("### 🎯 训练任务选择")
        train_target = st.radio("选择训练任务：", list(MODEL_FILES.keys()), index=0)
        st.success(f"已挂载【{train_target}】的预设超参数")

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
            except Exception: 
                pass

    with col_r:
        st.markdown("### 📊 运算结果区")
        if models[target_prop] is None:
            st.error(f"❌ 未检测到权重文件，请先前往训练模式。")
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
                            # 终极修复：正则提取纯数字 ID
                            nums = re.findall(r'\d+', uploaded_file.name)
                            file_id = nums[0] if nums else "UNKNOWN"
                            match = db_df[db_df['index_label'] == file_id]
                            if not match.empty:
                                # 灵活匹配列名大小写
                                col = 'Gap' if "Bandgap" in target_prop else 'lattice'
                                try:
                                    true_val = float(match.iloc[0][col])
                                    abs_err = abs(res - true_val)
                                    rel_err = (abs_err / true_val * 100) if true_val != 0 else 0
                                    error_html = f"""
<div style="display: flex; justify-content: space-around; margin-top: 20px; border-top: 2px solid #ecf0f1; padding-top: 20px;">
    <div>
        <div style="font-size: 1rem; color: #7f8c8d; text-transform: uppercase;">📊 数据库真实值 (ID:{file_id})</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #2980b9;">{true_val:.4f} <span style="font-size: 1.2rem;">{unit}</span></div>
    </div>
    <div>
        <div style="font-size: 1rem; color: #7f8c8d; text-transform: uppercase;">📉 预测误差</div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #e74c3c;">{abs_err:.4f} <span style="font-size: 1.2rem;">{unit}</span></div>
        <div style="font-size: 0.9rem; color: #e74c3c; font-weight:bold;">(相对误差: {rel_err:.2f}%)</div>
    </div>
</div>"""
                                except Exception: 
                                    error_html = "<div style='margin-top:15px; color:red;'>❌ 匹配成功但数据库数值格式有误。</div>"
                        
                        st.markdown(f"""
<div class="result-card">
    <div class="result-label">{icon} 目标性质: {target_prop.split(" ")[0]}</div>
    <div class="result-value">{res:.4f} <span style="font-size: 1.5rem; color:#7f8c8d;">{unit}</span></div>
    <div style="color: #27ae60; font-weight: 500; margin-bottom: 10px;">✓ 预测成功</div>
    {error_html}
</div>""", unsafe_allow_html=True)
                    except Exception as e: 
                        st.error(f"❌ 预测出错: {e}")

# --- 7. 训练模式界面 ---
elif app_mode == "⚙️ 模型训练模式":
    col_t1, col_t2 = st.columns([1.5, 1])
    with col_t1:
        st.markdown("<div class='train-card'>", unsafe_allow_html=True)
        st.markdown("### 📁 1. 上传数据集")
        uploaded_dataset = st.file_uploader("上传数据集 ZIP 压缩包", type="zip")
        
        st.markdown("### 📊 2. 自定义划分比例")
        c1, c2, c3 = st.columns(3)
        train_r = c1.number_input("训练集", 0.0, 1.0, 0.8, 0.05)
        val_r = c2.number_input("验证集", 0.0, 1.0, 0.1, 0.05)
        test_r = c3.number_input("测试集", 0.0, 1.0, 0.1, 0.05)
        
        valid = round(train_r + val_r + test_r, 2) == 1.0
        if not valid: 
            st.error(f"⚠️ 比例总和必须为 1.0 (当前: {train_r+val_r+test_r:.2f})")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 启动模型训练", use_container_width=True, disabled=not (valid and uploaded_dataset)):
            # --- 自动解压与过渡文件夹构建机制 ---
            # 引入时间戳，每次生成独一无二的文件夹名称
            run_id = str(int(time.time()))
            temp_root = f"temp_run_{run_id}"
            data_dir = f"temp_dataset_{run_id}"
            
            # 因为是全新名字，大概率不存在，但加上 exist_ok 更保险
            os.makedirs(temp_root, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)
            
            try:
                # 1. 解压
                zip_path = os.path.join(temp_root, "data.zip")
                with open(zip_path, "wb") as f: 
                    f.write(uploaded_dataset.getbuffer())
                with zipfile.ZipFile(zip_path, 'r') as z: 
                    z.extractall(temp_root)
                
                # 2. 智能寻路：查找包含 POSCAR 和 CSV 的目录
                real_folder = None
                target_csv = None
                for r, d, files in os.walk(temp_root):
                    csv_files = [f for f in files if f.endswith('.csv')]
                    has_poscar = any("POSCAR" in f for f in files)
                    
                    if csv_files and has_poscar:
                        real_folder = r
                        target_csv = csv_files[0]  
                        break
                
                if not real_folder:
                    st.error("❌ ZIP 包内未同时找到 `CSV表格` 和 `POSCAR` 文件，请确保它们放在同一个文件夹内！")
                    st.stop()
                
                # 3. 挂载数据到干净目录，并【强制重命名】CSV 喂给 main.py
                for item in os.listdir(real_folder):
                    src_path = os.path.join(real_folder, item)
                    
                    if item == target_csv:
                        # 核心修复：把压缩包里的 csv 强行改名为 main.py 认识的 id_prop.csv
                        dst_path = os.path.join(data_dir, "id_prop.csv")
                    else:
                        dst_path = os.path.join(data_dir, item)
                        
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path)
                    else:
                        shutil.copy2(src_path, dst_path)
                
                st.success(f"✅ 数据集挂载成功！(已自动将 {target_csv} 识别为训练标签)。开始运行...")
                
                # 4. 读取 JSON 参数 (严格绑定 train_target 并加上 encoding='utf-8')
                with open(MODEL_FILES[train_target]['param_path'], 'r', encoding='utf-8') as f:
                    params = json.load(f)
                
                # 5. 调用子进程
                # 使用 or 操作符防止 JSON 中出现 null 导致 params.get 返回 None
                cmd = [
                    sys.executable, "main.py", data_dir,
                    # 基础参数
                    "--epochs", str(params.get("epochs", 50) or 50),
                    "--batch-size", str(params.get("batch_size", 256) or 256),
                    "--lr", str(params.get("lr", 0.01) or 0.01),
                    "--train-ratio", str(train_r),
                    "--val-ratio", str(val_r),
                    "--test-ratio", str(test_r),
                    
                    # 补全遗漏的 JSON 核心超参数
                    "--n-conv", str(params.get("n_conv", 3) or 3),
                    "--atom-fea-len", str(params.get("atom_fea_len", 64) or 64),
                    "--optim", str(params.get("optim", "SGD") or "SGD"),
                    "--weight-decay", str(params.get("weight_decay", 0) or 0),
                    
                    # 【核心修复】：强行设置打印频率为 1，让它每个 batch 都输出日志，保证进度条平滑
                    "--print-freq", "1",
                    "--target", "0" 
                ]
                
                progress_bar = st.progress(0)
                status = st.empty()
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                
                total_epochs = int(params.get("epochs", 50) or 50)
                
                for line in process.stdout:
                    # 【核心修复】：解析更细腻的进度 (轮数 + 当前批次/总批次)
                    full_match = re.search(r"Epoch:\s*\[(\d+)\]\[(\d+)/(\d+)\]", line)
                    if full_match:
                        curr_epoch = int(full_match.group(1))
                        curr_batch = int(full_match.group(2))
                        total_batch = int(full_match.group(3))
                        
                        # 计算包含小数的细致进度比例
                        fractional_epoch = curr_epoch + (curr_batch / total_batch)
                        prog = min(fractional_epoch / total_epochs, 1.0)
                        progress_bar.progress(prog)
                    else:
                        # 兼容处理测试阶段等只有整数 Epoch 的情况
                        epoch_match = re.search(r"Epoch: \[(\d+)\]", line)
                        if epoch_match:
                            curr = int(epoch_match.group(1))
                            progress_bar.progress(min(curr / total_epochs, 1.0))
                            
                    status.code(f"实时日志: {line.strip()}")
                
                process.wait()

                # === 新增：自动画图并提供精准下载 ===
                if process.returncode == 0:
                    st.success("🎉 训练圆满完成！新模型已就绪。")
                    
                    st.markdown("### 📈 模型测试集表现")
                    result_csv = None 
                    
                    st.markdown("### 📥 下载训练成果")
                    st.info("⚠️ 请及时下载！如果网页休眠或刷新，这些文件可能会被系统重置清除。")
                    
                    for file_name in os.listdir("."):
                        if ("model_best" in file_name or "test_results" in file_name) and run_id in file_name:
                            if "test_results" in file_name:
                                result_csv = file_name
                                
                            with open(file_name, "rb") as f:
                                file_bytes = f.read()
                                st.download_button(
                                    label=f"💾 下载 {file_name}",
                                    data=file_bytes,
                                    file_name=file_name,
                                    mime="application/octet-stream"
                                )
                    
                    if result_csv:
                        with st.spinner("正在绘制线性拟合图..."):
                            try:
                                fig = draw_scatter_plot(result_csv, train_target)
                                st.pyplot(fig)
                                plt.close(fig) # 释放内存
                            except Exception as e:
                                st.warning(f"绘图失败: {e}")
                else:
                    st.error("❌ 训练异常终止，请检查日志。")
                    
            finally:
                # 销毁临时数据
                shutil.rmtree(temp_root, ignore_errors=True)
                shutil.rmtree(data_dir, ignore_errors=True)

    with col_t2:
        st.markdown("<div class='train-card'>", unsafe_allow_html=True)
        # 动态切换标题和展示内容
        st.markdown(f"### 🧬 {train_target.split(' ')[0]} 超参数预览")
        try:
            with open(MODEL_FILES[train_target]['param_path'], 'r', encoding='utf-8') as f:
                st.json(json.load(f))
        except Exception: 
            st.warning("参数文件读取失败")
        st.markdown("</div>", unsafe_allow_html=True)
