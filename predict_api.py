import os
import json
import torch
import numpy as np
from pymatgen.core.structure import Structure
# 确保你的文件夹里有 cgcnn 这个包
from cgcnn.model import CrystalGraphConvNet
from cgcnn.data import AtomCustomJSONInitializer, GaussianDistance


class SinglePredictor:
    def __init__(self, model_path, param_path, atom_init_path):
        """
        初始化预测器
        :param model_path: 模型权重文件 (.tar)
        :param param_path: 最佳超参数文件 (.json)
        :param atom_init_path: 原子初始化文件路径 (cgcnn/atom_init.json)
        """
        # 1. 解析最佳超参数 JSON 文件
        try:
            with open(param_path, 'r', encoding='utf-8') as f:
                best_params = json.load(f)
        except Exception as e:
            raise RuntimeError(f"无法读取参数文件 {param_path}。错误: {e}")

        # 提取参数
        atom_fea_len = best_params.get('atom_fea_len', 64)
        n_conv = best_params.get('n_conv', 3)
        h_fea_len = best_params.get('h_fea_len', 128)
        n_h = best_params.get('n_h', 1)

        # 2. 准备数据预处理工具 (使用传入的 atom_init.json 路径)
        self.ari = AtomCustomJSONInitializer(atom_init_path)
        self.gdf = GaussianDistance(dmin=0, dmax=8, step=0.2)
        self.max_num_nbr = 12
        self.radius = 8

        # 3. 构建模型结构
        self.model = CrystalGraphConvNet(
            orig_atom_fea_len=92,
            nbr_fea_len=41,
            atom_fea_len=atom_fea_len,
            n_conv=n_conv,
            h_fea_len=h_fea_len,
            n_h=n_h,
            classification=False
        )

        # 4. 加载模型权重
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"无法加载模型文件 {model_path}。错误: {e}")

        # 5. 记录归一化参数
        self.normalizer_mean = checkpoint['normalizer']['mean']
        self.normalizer_std = checkpoint['normalizer']['std']

    def process_single_poscar(self, poscar_path):
        """将 POSCAR 转换为模型需要的张量"""
        crystal = Structure.from_file(poscar_path)
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)

        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []

        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) + [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))

        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        crystal_atom_idx = [torch.LongTensor(np.arange(atom_fea.shape[0]))]

        return atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx

    def predict(self, poscar_path):
        """执行预测并进行反归一化"""
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = self.process_single_poscar(poscar_path)
        with torch.no_grad():
            output = self.model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            pred_value = output.item() * self.normalizer_std.item() + self.normalizer_mean.item()
        return pred_value