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
import random
from stmol import showmol
from pymatgen.core.structure import Structure
from predict_api import SinglePredictor

# === 画图所需的依赖包 ===
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
    """根据测试结果 CSV 绘制散点图并返回 fig 对象"""
    df_test = pd.read_csv(csv_path, header=None, names=['Index', 'Calculated', 'Predicted'])
    
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

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    if "带隙" in target_type:
        x_y_min = -0.1
        x_y_max = max_val + 0.2
        unit_str = " eV"
        ax.set_xlabel(r'Calculated $E_{g\_HSE}$ (eV)', fontsize=16)
        ax.set_ylabel(r'Predicted $E_{g\_HSE}$ (eV)', fontsize=16)
        text_x = 0.40
    else:  
        x_y_min = min_val - 0.1
        x_y_max = max_val + 0.2
        unit_str = ""
        ax.set_xlabel(r'Calculated a_HSE (Å)', fontsize=16)
        ax.set_ylabel(r'Predicted a_HSE (Å)', fontsize=16)
        text_x = 0.50

    ax.plot([x_y_min, x_y_max], [x_y_min, x_y_max], 'k--', lw=1, alpha=0.5)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(MaxNLocator(integer=False, nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(integer=False, nbins=6))
    ax.set_xlim([x_y_min, x_y_max])
    ax.set_ylim([x_y_min, x_y_max])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    scatter.set_clim(vmin=0, vmax=np.max(z))
    cbar.locator = MaxNLocator(nbins=6)
    cbar.update_ticks()

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
    
    pred_mode = st.radio("切换预测模式：", ["📄 单文件预测", "📦 批量文件预测 (ZIP 压缩包)"], horizontal=True)
    st.markdown("---")

    # ========================== 单文件预测逻辑 ==========================
    if pred_mode == "📄 单文件预测":
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
                                nums = re.findall(r'\d+', uploaded_file.name)
                                file_id = nums[0] if nums else "UNKNOWN"
                                match = db_df[db_df['index_label'] == file_id]
                                if not match.empty:
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

    # ========================== 批量预测逻辑 ==========================
    elif pred_mode == "📦 批量文件预测 (ZIP 压缩包)":
        st.markdown("### 📥 批量数据输入区")
        uploaded_zip = st.file_uploader("上传包含多个 POSCAR 和 (可选) 真值表 CSV 的 ZIP 压缩包", type="zip")
        
        max_predict_num = st.number_input("期望预测的样本数量 (将在剔除真值 <0 后随机抽取)", min_value=1, value=50, step=1)
        
        if uploaded_zip and models[target_prop]:
            if st.button("🚀 启动批量预测", use_container_width=True):
                run_id = str(int(time.time()))
                batch_temp = f"batch_temp_{run_id}"
                os.makedirs(batch_temp, exist_ok=True)
                
                try:
                    with st.spinner("正在解压并智能比对数据..."):
                        zip_path = os.path.join(batch_temp, "upload.zip")
                        with open(zip_path, "wb") as f:
                            f.write(uploaded_zip.getbuffer())
                        with zipfile.ZipFile(zip_path, 'r') as z:
                            z.extractall(batch_temp)
                    
                    df_batch = None
                    for r, d, files in os.walk(batch_temp):
                        csv_files = [f for f in files if f.endswith('.csv')]
                        if csv_files:
                            try:
                                df_batch = pd.read_csv(os.path.join(r, csv_files[0]))
                            except: pass
                            break
                    
                    poscar_files = []
                    for r, d, files in os.walk(batch_temp):
                        for f in files:
                            if "POSCAR" in f:
                                poscar_files.append(os.path.join(r, f))
                                
                    if not poscar_files:
                        st.error("❌ ZIP 包内未找到任何 POSCAR 文件！")
                    else:
                        target_idx = 2 if "Bandgap" in target_prop else 4
                        
                        # === 🚀 核心优化：构建真值表字典进行 O(1) 极速哈希查找 ===
                        true_val_dict = {}
                        if df_batch is not None:
                            try:
                                # 强制转数字，并剥离首尾空格
                                df_batch['target'] = pd.to_numeric(df_batch.iloc[:, target_idx], errors='coerce')
                                for _, row in df_batch.iterrows():
                                    k = str(row.iloc[0]).strip()
                                    v = row['target']
                                    if pd.notna(v):
                                        true_val_dict[k] = v
                            except: pass

                        valid_data_info = []
                        for p_path in poscar_files:
                            filename = os.path.basename(p_path)
                            nums = re.findall(r'\d+', filename)
                            if not nums:
                                parent_dir = os.path.basename(os.path.dirname(p_path))
                                nums = re.findall(r'\d+', parent_dir)
                            file_id = nums[0] if nums else "UNKNOWN"
                            
                            # O(1) 极速提取真值
                            true_val = true_val_dict.get(file_id, np.nan)
                            
                            if pd.notna(true_val) and true_val < 0:
                                continue
                                
                            valid_data_info.append((p_path, file_id, true_val))
                        # =======================================================

                        total_valid_count = len(valid_data_info)
                        
                        if total_valid_count == 0:
                            st.error("⚠️ 数据过滤完毕后，未能找到任何有效的预测样本（可能所有数值均 < 0 或未能成功匹配表格），预测已终止。")
                        else:
                            if total_valid_count > max_predict_num:
                                valid_data_info = random.sample(valid_data_info, max_predict_num)
                                st.success(f"✅ 剔除异常数据（<0）后，数据库真实剩余 **{total_valid_count}** 个有效样本。已从中随机抽取 **{max_predict_num}** 个进行预测。")
                            else:
                                st.success(f"✅ 剔除异常数据后，真实剩余 **{total_valid_count}** 个有效样本 (未超过您设定的阈值)，即将全量预测。")
                                
                            prog_bar = st.progress(0)
                            status_text = st.empty()
                            results = []
                            
                            for i, (p_path, file_id, true_val) in enumerate(valid_data_info):
                                status_text.code(f"正在预测 ({i+1}/{len(valid_data_info)}): 晶体ID {file_id}")
                                try:
                                    pred_val = models[target_prop].predict(p_path)
                                    results.append([file_id, true_val, pred_val])
                                except Exception as e:
                                    pass 
                                
                                prog_bar.progress((i + 1) / len(valid_data_info))
                            
                            status_text.success(f"🎉 批量预测完成！共生成 {len(results)} 条数据结果。")
                            
                            res_df = pd.DataFrame(results, columns=["Index", "Calculated", "Predicted"])
                            out_csv_name = f"batch_predict_results_{run_id}.csv"
                            res_df.to_csv(out_csv_name, index=False, header=False)
                            
                            st.markdown("---")
                            
                            col_a, col_b = st.columns([1, 1.5])
                            with col_a:
                                st.markdown("### 📥 获取预测报告")
                                st.info("完整的编号、真实值及预测值对照表已生成，请点击下载保存。")
                                with open(out_csv_name, "rb") as f:
                                    st.download_button(
                                        label=f"💾 下载 {out_csv_name}",
                                        data=f.read(),
                                        file_name=out_csv_name,
                                        mime="text/csv"
                                    )
                            
                            with col_b:
                                valid_rows = res_df.dropna()
                                if len(valid_rows) >= 2:
                                    st.markdown("### 📈 批量预测散点拟合")
                                    with st.spinner("正在绘制可视化图表..."):
                                        fig = draw_scatter_plot(out_csv_name, target_prop)
                                        st.pyplot(fig)
                                        plt.close(fig)
                                else:
                                    st.warning("⚠️ CSV 内匹配到的有效真实值不足（或未提供参考表），无法绘制真实/预测对比散点图。")
                
                finally:
                    shutil.rmtree(batch_temp, ignore_errors=True)

# --- 7. 训练模式界面 ---
elif app_mode == "⚙️ 模型训练模式":
    col_t1, col_t2 = st.columns([1.5, 1])
    with col_t1:
        st.markdown("<div class='train-card'>", unsafe_allow_html=True)
        st.markdown("### 📁 1. 上传数据集")
        uploaded_dataset = st.file_uploader("上传数据集 ZIP 压缩包", type="zip")
        
        st.markdown("### 📊 2. 自定义划分比例与抽样")
        c1, c2, c3 = st.columns(3)
        train_r = c1.number_input("训练集", 0.0, 1.0, 0.8, 0.05)
        val_r = c2.number_input("验证集", 0.0, 1.0, 0.1, 0.05)
        test_r = c3.number_input("测试集", 0.0, 1.0, 0.1, 0.05)
        
        max_train_num = st.number_input("期望用于训练的总样本数量 (将在剔除真值 <0 后随机抽取)", min_value=10, value=500, step=10)
        
        valid = round(train_r + val_r + test_r, 2) == 1.0
        if not valid: 
            st.error(f"⚠️ 比例总和必须为 1.0 (当前: {train_r+val_r+test_r:.2f})")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚀 启动模型训练", use_container_width=True, disabled=not (valid and uploaded_dataset)):
            run_id = str(int(time.time()))
            temp_root = f"temp_run_{run_id}"
            data_dir = f"temp_dataset_{run_id}"
            
            os.makedirs(temp_root, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)
            
            try:
                zip_path = os.path.join(temp_root, "data.zip")
                with open(zip_path, "wb") as f: 
                    f.write(uploaded_dataset.getbuffer())
                with zipfile.ZipFile(zip_path, 'r') as z: 
                    z.extractall(temp_root)
                
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
                
                target_idx = 2 if "Bandgap" in train_target else 4
                target_csv_path = os.path.join(real_folder, target_csv)
                
                df_train_raw = pd.read_csv(target_csv_path, header=None)
                numeric_targets = pd.to_numeric(df_train_raw.iloc[:, target_idx], errors='coerce')
                valid_mask = (numeric_targets >= 0) & (numeric_targets.notna())
                df_train_valid = df_train_raw[valid_mask]
                
                total_valid_count = len(df_train_valid)
                if total_valid_count == 0:
                    st.error("⚠️ 数据过滤完毕后，未能找到任何有效的训练样本（可能所有数值均 < 0 或表格格式不匹配），训练终止。")
                    st.stop()
                    
                if total_valid_count > max_train_num:
                    df_train_sampled = df_train_valid.sample(n=max_train_num, random_state=int(time.time()))
                    st.success(f"✅ 剔除异常数据（<0）后，真实剩余 **{total_valid_count}** 个有效样本。已从中随机抽取 **{max_train_num}** 个进行训练。")
                else:
                    df_train_sampled = df_train_valid
                    st.success(f"✅ 剔除异常数据后，真实剩余 **{total_valid_count}** 个有效样本 (未超过您设定的阈值)，将全量用于训练。")
                
                df_train_sampled.to_csv(os.path.join(data_dir, "id_prop.csv"), index=False, header=False)
                
                for item in os.listdir(real_folder):
                    if item == target_csv:
                        continue 
                    src_path = os.path.join(real_folder, item)
                    dst_path = os.path.join(data_dir, item)
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path)
                    else:
                        shutil.copy2(src_path, dst_path)
                
                # === 替换的文案 ===
                st.info("🎯 数据精洗完成，开始执行深度图神经网络训练...")
                
                with open(MODEL_FILES[train_target]['param_path'], 'r', encoding='utf-8') as f:
                    params = json.load(f)
                
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
                    "--optim", str(params.get("optim", "SGD") or "SGD"),
                    "--weight-decay", str(params.get("weight_decay", 0) or 0),
                    "--print-freq", "1",
                    "--target", str(target_idx) 
                ]
                
                progress_bar = st.progress(0)
                status = st.empty()
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                
                total_epochs = int(params.get("epochs", 50) or 50)
                
                for line in process.stdout:
                    full_match = re.search(r"Epoch:\s*\[(\d+)\]\[(\d+)/(\d+)\]", line)
                    if full_match:
                        curr_epoch = int(full_match.group(1))
                        curr_batch = int(full_match.group(2))
                        total_batch = int(full_match.group(3))
                        fractional_epoch = curr_epoch + (curr_batch / total_batch)
                        prog = min(fractional_epoch / total_epochs, 1.0)
                        progress_bar.progress(prog)
                        status.code(f"实时日志: {line.strip()}")
                    else:
                        epoch_match = re.search(r"Epoch: \[(\d+)\]", line)
                        if epoch_match:
                            curr = int(epoch_match.group(1))
                            progress_bar.progress(min(curr / total_epochs, 1.0))
                            status.code(f"实时日志: {line.strip()}")
                        else:
                            test_match = re.search(r"Test:\s*\[(\d+)/(\d+)\]", line)
                            if test_match:
                                status.code(f"正在进行验证/测试评估 ({test_match.group(1)}/{test_match.group(2)})...")
                            else:
                                status.code(f"实时日志: {line.strip()}")
                
                process.wait()
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
                shutil.rmtree(temp_root, ignore_errors=True)
                shutil.rmtree(data_dir, ignore_errors=True)

    with col_t2:
        st.markdown("<div class='train-card'>", unsafe_allow_html=True)
        st.markdown(f"### 🧬 {train_target.split(' ')[0]} 超参数预览")
        try:
            with open(MODEL_FILES[train_target]['param_path'], 'r', encoding='utf-8') as f:
                st.json(json.load(f))
        except Exception: 
            st.warning("参数文件读取失败")
        st.markdown("</div>", unsafe_allow_html=True)
