# -*- coding: utf-8 -*-
"""
多模态融合模型注意力回溯脚本 (V4.3 - 最终修正版)
==================================================================================
本脚本现在直接从用户验证过可以工作的训练脚本中导入数据加载函数，
以确保路径生成逻辑100%一致，从根本上解决文件找不到的问题。

V4.3更新：修正了autocast的调用错误 (device_type=DEVICE.type -> device_type=DEVICE)

核心功能:
1.  加载最终的融合模型和其依赖的单模态模型。
2.  调用训练脚本的函数来获取验证数据集。
3.  提取和分析与最终分类决策相关的切片注意力分数。
4.  生成一个覆盖全验证集的汇总表和聚合注意力图。

前置要求:
- 项目结构符合规范。
- 训练脚本 'amyloid+tau_vit_2d5_cross_attention_fusion.py' 存在于 'src/architectures/' 目录。
- pip install timm torch pandas scikit-learn SimpleITK tqdm matplotlib
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import math
import importlib.util  # 用于从文件导入模块

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import SimpleITK as sitk
import timm
import matplotlib.pyplot as plt

# ======================================================================
# --- [动态路径与模块导入区] ---
# ======================================================================

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd()

# --- 动态导入训练脚本中的数据创建函数 ---
TRAINING_SCRIPT_FILENAME = 'amyloid+tau_vit_2d5_cross_attention_fusion.py'
TRAINING_SCRIPT_PATH = PROJECT_ROOT / "src" / "architectures" / TRAINING_SCRIPT_FILENAME

print(f"--- 正在从 '{TRAINING_SCRIPT_PATH}' 动态加载函数...")
if not TRAINING_SCRIPT_PATH.exists():
    raise FileNotFoundError(f"错误：找不到必需的训练脚本: {TRAINING_SCRIPT_PATH}")

spec = importlib.util.spec_from_file_location("training_pipeline", TRAINING_SCRIPT_PATH)
training_pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_pipeline)

create_multimodal_data_list = training_pipeline.create_multimodal_data_list


# --- 导入结束 ---


# ======================================================================
# --- [模型定义区 - 从训练脚本复制，保持一致] ---
# ======================================================================

class PretrainedVisionTransformer2_5D(training_pipeline.PretrainedVisionTransformer2_5D):
    # 重写extract_feature_vector以返回注意力
    def extract_feature_vector(self, x, return_attention=False):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        slice_features = self.vit_2d(x).view(B, D, self.embed_dim)
        slice_features = slice_features.permute(1, 0, 2)
        slice_features = self.slice_pos_embed(slice_features)

        attn_output, slice_attn_weights = self.agg_attention(slice_features, slice_features, slice_features)

        attn_output = self.agg_norm(attn_output + slice_features)
        final_feature = self.agg_linear(attn_output)
        final_feature = final_feature[0]

        if return_attention:
            return final_feature, slice_attn_weights
        return final_feature


class FusionModel(training_pipeline.FusionModel):
    def __init__(self, amyl_model_path, tau_model_path, model_name='vit_tiny_patch16_224'):
        super(training_pipeline.FusionModel, self).__init__()
        self.amyl_extractor = PretrainedVisionTransformer2_5D(model_name=model_name)
        self.tau_extractor = PretrainedVisionTransformer2_5D(model_name=model_name)

        if not os.path.exists(amyl_model_path):
            raise FileNotFoundError(f"Amyloid model not found at: {amyl_model_path}")
        if not os.path.exists(tau_model_path):
            raise FileNotFoundError(f"Tau model not found at: {tau_model_path}")

        print(f"--- 正在加载 Amyloid 主干网络权重: {amyl_model_path}")
        self.amyl_extractor.load_state_dict(torch.load(amyl_model_path))
        print(f"--- 正在加载 Tau 主干网络权重: {tau_model_path}")
        self.tau_extractor.load_state_dict(torch.load(tau_model_path))

        print("--- 正在冻结两个主干网络的所有参数... ---")
        for param in self.amyl_extractor.parameters():
            param.requires_grad = False
        for param in self.tau_extractor.parameters():
            param.requires_grad = False

        embed_dim = self.amyl_extractor.embed_dim
        num_heads = self.amyl_extractor.agg_attention.num_heads
        self.fusion_module = training_pipeline.CrossAttentionFusion(embed_dim, num_heads)
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 1)
        )

    def forward(self, x_amyl, x_tau, return_slice_attention=False):
        self.amyl_extractor.eval()
        self.tau_extractor.eval()

        with torch.no_grad():
            features_amyl, slice_attn_amyl = self.amyl_extractor.extract_feature_vector(x_amyl, return_attention=True)
            features_tau, slice_attn_tau = self.tau_extractor.extract_feature_vector(x_tau, return_attention=True)

        fused_features = self.fusion_module(features_amyl, features_tau)
        output = self.classification_head(fused_features).squeeze(-1)

        if return_slice_attention:
            return output, slice_attn_amyl, slice_attn_tau
        return output


# ======================================================================
# --- [数据加载区 - 同样可复用] ---
# ======================================================================
class FusionPetDatasetEval(Dataset):
    def __init__(self, dataframe, target_size, transform_func=None):
        self.df = dataframe
        self.transform = transform_func
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath_amyl = row['filepath_amyl']
        filepath_tau = row['filepath_tau']
        label = row['label']

        try:
            volume_amyl = np.load(filepath_amyl).astype(np.float32)
            volume_tau = np.load(filepath_tau).astype(np.float32)
        except FileNotFoundError as e:
            print(f"!! 警告: 找不到文件 {e}，将跳过此样本。")
            return torch.empty(0), torch.empty(0), -1, ""

        if self.transform:
            volume_amyl = self.transform(volume_amyl, self.target_size)
            volume_tau = self.transform(volume_tau, self.target_size)

        label = torch.tensor(label, dtype=torch.float32)
        return volume_amyl, volume_tau, label, Path(filepath_amyl).stem


def fusion_collate_fn_eval(batch):
    batch = list(filter(lambda x: x[2] != -1, batch))
    if not batch: return torch.empty(0), torch.empty(0), torch.empty(0), []
    amyl_vols, tau_vols, labels, fnames = zip(*batch)
    return torch.utils.data.dataloader.default_collate(amyl_vols), \
        torch.utils.data.dataloader.default_collate(tau_vols), \
        torch.utils.data.dataloader.default_collate(labels), \
        fnames


# ======================================================================
# --- [主执行区] ---
# ======================================================================
def analyze_attention():
    # --- [配置区] ---
    MODEL_NAME = 'vit_tiny_patch16_224'
    TARGET_SIZE = (64, 224, 224)
    BATCH_SIZE = 1

    MODELS_ROOT = PROJECT_ROOT / "models"
    REPORTS_ROOT = PROJECT_ROOT / "reports"
    REPORTS_ROOT.mkdir(exist_ok=True)

    UNIMODAL_MODELS_DIR = MODELS_ROOT / "pet_unimodal"
    MULTIMODAL_MODELS_DIR = MODELS_ROOT / "pet_multimodal"

    MODEL_TO_EVALUATE = MULTIMODAL_MODELS_DIR / 'amyloid+tau_vit_2d5_cross_attention_fusion_best_model.pth'
    AMYL_MODEL_PATH = UNIMODAL_MODELS_DIR / 'vit_2d5_amyloid_best.pth'
    TAU_MODEL_PATH = UNIMODAL_MODELS_DIR / 'vit_2d5_tau_best.pth'
    # --- 配置区结束 ---

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- 开始执行注意力回溯分析 (V4.3) ---")
    print(f"--- 使用设备: {DEVICE} ---")

    # 1. 准备验证数据 (直接调用工作函数)
    print("\n--- [步骤 1/3] 调用训练脚本中的函数准备数据 ---")
    _, val_df = create_multimodal_data_list()
    if val_df is None or val_df.empty:
        print("!! 致命错误: 从训练脚本获取数据失败或验证集为空。")
        return

    val_dataset = FusionPetDatasetEval(val_df, target_size=TARGET_SIZE,
                                       transform_func=training_pipeline.resize_and_normalize)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=fusion_collate_fn_eval)
    print(f"--- 验证数据加载器创建完成，包含 {len(val_df)} 条记录。")

    # 2. 加载模型
    print(f"\n--- [步骤 2/3] 加载最终融合模型 '{MODEL_TO_EVALUATE.name}' ---")
    try:
        model = FusionModel(AMYL_MODEL_PATH, TAU_MODEL_PATH, model_name=MODEL_NAME).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_TO_EVALUATE, map_location=DEVICE), strict=False)
    except FileNotFoundError as e:
        print(f"!! 致命错误: 找不到模型文件 '{e}'。请检查路径配置。")
        return
    model.eval()

    # 3. 聚合分析
    print(f"\n--- [步骤 3/3] 开始对整个验证集进行聚合分析 ---")

    results_list = []
    normal_slice_attn_amyl, abnormal_slice_attn_amyl = [], []
    normal_slice_attn_tau, abnormal_slice_attn_tau = [], []

    with torch.no_grad():
        for amyl_vol, tau_vol, labels, fnames in tqdm(val_loader, desc="正在分析验证集"):
            if amyl_vol.numel() == 0 or tau_vol.numel() == 0: continue
            amyl_vol, tau_vol = amyl_vol.to(DEVICE), tau_vol.to(DEVICE)

            # 【最终修正】直接使用DEVICE变量，而不是DEVICE.type
            with autocast(device_type=DEVICE):
                output, slice_attn_amyl, slice_attn_tau = model(amyl_vol, tau_vol, return_slice_attention=True)

            prob = torch.sigmoid(output).item()
            label = int(labels.item())

            slice_scores_amyl = slice_attn_amyl[0, 0, :].cpu().numpy()
            slice_scores_tau = slice_attn_tau[0, 0, :].cpu().numpy()

            results_list.append({
                "FileID": fnames[0],
                "TrueLabel": "Abnormal" if label == 1 else "Normal",
                "PredictedProb": f"{prob:.4f}",
                "PredictedLabel": "Abnormal" if prob > 0.5 else "Normal",
                "Correct": (prob > 0.5) == (label == 1),
                "TopSlice_Amyl": np.argmax(slice_scores_amyl),
                "TopSlice_Tau": np.argmax(slice_scores_tau)
            })

            if label == 1:
                abnormal_slice_attn_amyl.append(slice_scores_amyl)
                abnormal_slice_attn_tau.append(slice_scores_tau)
            else:
                normal_slice_attn_amyl.append(slice_scores_amyl)
                normal_slice_attn_tau.append(slice_scores_tau)

    # --- 4. 打印并保存汇总表 ---
    print("\n\n======================================================================================")
    print(f"--- 切片注意力分析汇总表 (全验证集, N={len(val_df)}) ---")
    results_df = pd.DataFrame(results_list)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results_df.to_string())
    csv_report_path = REPORTS_ROOT / 'attention_analysis_summary.csv'
    results_df.to_csv(csv_report_path, index=False)
    print(f"\n--- 汇总表已保存到: {csv_report_path} ---")

    # --- 5. 绘制并保存聚合注意力图 ---
    print("\n\n======================================================================================")
    print(f"--- 平均切片注意力对比图 (全验证集, N={len(val_df)}) ---")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey=True, dpi=150)
    fig.suptitle(f'Average Slice Attention Patterns (Full Validation Set, N={len(val_df)})', fontsize=16, weight='bold')
    indices = np.arange(TARGET_SIZE[0])

    def plot_avg_attention(ax, data, title, color):
        if not data:
            ax.text(0.5, 0.5, 'No samples in this category', ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=14)
            return
        mean_scores = np.mean(data, axis=0)
        std_scores = np.std(data, axis=0)
        ax.bar(indices, mean_scores, yerr=std_scores, capsize=3, color=color, alpha=0.7, ecolor='gray',
               label='Mean Attention')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f'Slice Index (0=Bottom -> {TARGET_SIZE[0] - 1}=Top)')
        ax.set_ylabel('Average Attention Score')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xlim(-1, TARGET_SIZE[0])
        ax.legend()

    plot_avg_attention(axes[0, 0], normal_slice_attn_amyl,
                       f'Amyloid PET - Normal Samples (N={len(normal_slice_attn_amyl)})', 'green')
    plot_avg_attention(axes[0, 1], abnormal_slice_attn_amyl,
                       f'Amyloid PET - Abnormal Samples (N={len(abnormal_slice_attn_amyl)})', 'red')
    plot_avg_attention(axes[1, 0], normal_slice_attn_tau, f'Tau PET - Normal Samples (N={len(normal_slice_attn_tau)})',
                       'blue')
    plot_avg_attention(axes[1, 1], abnormal_slice_attn_tau,
                       f'Tau PET - Abnormal Samples (N={len(abnormal_slice_attn_tau)})', 'orange')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure_report_path = REPORTS_ROOT / 'attention_analysis_plot.png'
    plt.savefig(figure_report_path)
    print(f"\n--- 注意力图已保存到: {figure_report_path} ---")
    plt.show()


if __name__ == '__main__':
    analyze_attention()