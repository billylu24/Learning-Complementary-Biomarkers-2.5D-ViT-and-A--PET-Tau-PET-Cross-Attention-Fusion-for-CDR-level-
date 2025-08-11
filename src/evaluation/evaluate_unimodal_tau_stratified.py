# -*- coding: utf-8 -*-
"""
增强版独立模型评估脚本 (Tau PET) - 支持按疾病阶段分层分析
======================================================================
本脚本用于对已训练的Tau PET模型进行深入、分阶段的评估。

核心功能:
1.  **标准评估**: 输出总体的分类报告、混淆矩阵和AUC分数。
2.  **分层注意力分析**:
    - 根据CDR评分将样本自动分为三组:
        1. Normal (CDR = 0)
        2. MCI (CDR = 0.5)
        3. Dementia (CDR >= 1.0)
    - 为这三组生成按切片平均的注意力对比图。
3.  **新增: 经典机器学习健全性检查 (Sanity Check)**:
    - 通过简单的逻辑回归模型验证顶部切片区域的特征是否真的比
      内侧颞叶区域更具区分度，以确认ViT模型的发现并非算法偶然。

前置要求:
- 将训练好的模型文件、CSV文件与此脚本放在同一目录。
- pip install timm torch pandas scikit-learn SimpleITK tqdm matplotlib
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer  # <-- 修复BUG：新增imputer
import SimpleITK as sitk
import timm
import matplotlib.pyplot as plt
import collections


# ======================================================================
# --- [全局定义区 - 与训练脚本保持一致] ---
# ======================================================================

class PetCdrDataset(Dataset):
    """自定义数据集，现在也返回CDR分数用于分层分析。"""

    def __init__(self, dataframe, target_size, transform_func=None):
        self.df = dataframe
        self.transform = transform_func
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row['filepath']
        label = row['label']
        cdr_score = row['CDGLOBAL']  # <-- 新增：获取CDR分数

        try:
            volume = np.load(filepath).astype(np.float32)
        except FileNotFoundError:
            # 返回一个独特的标识符来过滤掉坏数据
            return torch.empty(0), -1, -1, torch.empty(0)

        if self.transform:
            # 注意：健全性检查需要原始数据，所以这里返回变换前和变换后的
            original_volume = torch.from_numpy(volume).clone()
            transformed_volume = self.transform(volume, self.target_size)
            return transformed_volume, label, cdr_score, original_volume
        else:
            volume_tensor = torch.from_numpy(volume)
            return volume_tensor, label, cdr_score, volume_tensor


def resize_and_normalize(volume_np, target_size):
    min_val, max_val = np.min(volume_np), np.max(volume_np)
    if max_val > min_val:
        volume_np = (volume_np - min_val) / (max_val - min_val)
    sitk_image = sitk.GetImageFromArray(volume_np)
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    new_spacing = [
        original_size[i] * original_spacing[i] / target_size[i] for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize((target_size[2], target_size[1], target_size[0]))
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkLinear)
    resized_sitk_image = resampler.Execute(sitk_image)
    resized_volume = sitk.GetArrayFromImage(resized_sitk_image)
    return torch.from_numpy(resized_volume).unsqueeze(0)


def collate_fn_stratified(batch):
    """新的collate_fn，处理包含CDR分数的批次。"""
    batch = list(filter(lambda x: x[1] != -1, batch))
    if not batch: return torch.empty(0), torch.empty(0), torch.empty(0), []

    transformed_volumes, labels, cdr_scores, original_volumes = zip(*batch)

    transformed_volumes = torch.utils.data.dataloader.default_collate(transformed_volumes)
    labels = torch.utils.data.dataloader.default_collate(labels)
    cdr_scores = torch.utils.data.dataloader.default_collate(cdr_scores)

    return transformed_volumes, labels, cdr_scores, original_volumes


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PretrainedVisionTransformer2_5D(nn.Module):
    def __init__(self, model_name='vit_tiny_patch16_224', num_classes=1, dropout=0.1):
        super().__init__()
        self.vit_2d = timm.create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 to get features
        self.embed_dim = self.vit_2d.embed_dim
        num_heads = self.vit_2d.blocks[-1].attn.num_heads

        original_patch_embed = self.vit_2d.patch_embed.proj
        new_patch_embed = nn.Conv2d(1, self.embed_dim, kernel_size=original_patch_embed.kernel_size,
                                    stride=original_patch_embed.stride, padding=original_patch_embed.padding)
        with torch.no_grad():
            new_patch_embed.weight.copy_(original_patch_embed.weight.mean(1, keepdim=True))
            if new_patch_embed.bias is not None:
                new_patch_embed.bias.copy_(original_patch_embed.bias)
        self.vit_2d.patch_embed.proj = new_patch_embed

        self.slice_pos_embed = PositionalEncoding(self.embed_dim, dropout)
        self.agg_attention = nn.MultiheadAttention(self.embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.agg_norm = nn.LayerNorm(self.embed_dim)
        self.agg_linear = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, x, return_attention=False):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        slice_features = self.vit_2d(x).view(B, D, self.embed_dim)
        slice_features = slice_features.permute(1, 0, 2)
        slice_features = self.slice_pos_embed(slice_features)

        query = slice_features[0:1]

        attn_output, attn_weights = self.agg_attention(query, slice_features, slice_features)

        attn_output = self.agg_norm(attn_output + query)
        final_feature = self.agg_linear(attn_output).squeeze(0)

        output = self.head(final_feature).squeeze(-1)

        if return_attention:
            return output, attn_weights.squeeze(1)
        return output


# ======================================================================
# --- [新增功能: 经典机器学习健全性检查] ---
# ======================================================================
def perform_roi_sanity_check(dataset):
    """
    使用经典机器学习方法验证ViT模型的发现。
    该函数完全独立于ViT模型，直接在原始图像数据上操作。
    它比较了两个关键ROI的特征在区分Normal和MCI时的预测能力。
    """
    print("\n--- [步骤 4/4] 执行经典机器学习健全性检查 (ROI Sanity Check) ---")

    # 定义ROI的切片范围
    ROI_TOP_SLICES = slice(54, 64)  # 顶部皮层区域
    ROI_MTL_SLICES = slice(30, 40)  # 内侧颞叶区域 (理论早期区域)

    features_top = []
    features_mtl = []
    labels = []

    print("--- 正在从原始图像中提取ROI特征... ---")
    for i in tqdm(range(len(dataset)), desc="提取ROI特征"):
        # 我们只关心Normal (CDR=0) 和 MCI (CDR=0.5) 的对比
        _, _, cdr_score, original_volume = dataset[i]
        if cdr_score.item() > 0.5:
            continue

        if original_volume.numel() == 0:
            continue

        # 提取特征：ROI内的平均信号值
        # 修复BUG：检查切片是否为空，防止计算mean()时产生NaN
        top_slice = original_volume[ROI_TOP_SLICES, :, :]
        mtl_slice = original_volume[ROI_MTL_SLICES, :, :]

        feature_top = top_slice.mean().item() if top_slice.numel() > 0 else np.nan
        feature_mtl = mtl_slice.mean().item() if mtl_slice.numel() > 0 else np.nan

        features_top.append(feature_top)
        features_mtl.append(feature_mtl)
        labels.append(1 if cdr_score.item() == 0.5 else 0)

    if len(labels) < 2 or len(set(labels)) < 2:
        print("!! 健全性检查中止：没有足够的Normal和MCI样本进行有意义的比较。")
        return

    X_top = np.array(features_top).reshape(-1, 1)
    X_mtl = np.array(features_mtl).reshape(-1, 1)
    y = np.array(labels)

    # 修复BUG: 使用Imputer处理由空切片产生的NaN值
    imputer = SimpleImputer(strategy='mean')
    X_top_imputed = imputer.fit_transform(X_top)
    X_mtl_imputed = imputer.fit_transform(X_mtl)

    # 初始化逻辑回归模型
    lr_top = LogisticRegression(random_state=42, class_weight='balanced')
    lr_mtl = LogisticRegression(random_state=42, class_weight='balanced')

    print("--- 正在训练和评估简单的逻辑回归模型... ---")
    lr_top.fit(X_top_imputed, y)
    y_pred_prob_top = lr_top.predict_proba(X_top_imputed)[:, 1]
    auc_top = roc_auc_score(y, y_pred_prob_top)

    lr_mtl.fit(X_mtl_imputed, y)
    y_pred_prob_mtl = lr_mtl.predict_proba(X_mtl_imputed)[:, 1]
    auc_mtl = roc_auc_score(y, y_pred_prob_mtl)

    print("\n--- ROI健全性检查结果 (Normal vs MCI) ---")
    print("===================================================================")
    print(f"目的: 验证哪个区域的特征对早期诊断(MCI)更有预测价值。")
    print(f"方法: 使用逻辑回归和单一ROI特征进行分类。")
    print("-------------------------------------------------------------------")
    print(f"使用 [顶部皮层 ROI] 特征的AUC: {auc_top:.4f}")
    print(f"使用 [内侧颞叶 ROI] 特征的AUC: {auc_mtl:.4f}")
    print("===================================================================")

    if auc_top > auc_mtl + 0.05:
        print("\n[结论]: 顶部皮层ROI的预测能力显著优于内侧颞叶ROI。")
        print("这为ViT模型的发现提供了强有力的、独立于算法的证据。")
    elif auc_mtl > auc_top + 0.05:
        print("\n[结论]: 内侧颞叶ROI的预测能力显著优于顶部皮层ROI。")
        print("这与经典理论更吻合，提示ViT模型的行为可能需要进一步探究。")
    else:
        print("\n[结论]: 两个区域的预测能力相当。")
        print("这表明两者都包含早期诊断信息，ViT模型可能因为其他因素偏好顶部区域。")


# ======================================================================
# --- [主执行区] ---
# ======================================================================
def evaluate_stratified():
    # --- 配置区 ---
    MODEL_TO_EVALUATE = 'tau_adni_pretrained_vit_tiny_best.pth'
    MODEL_NAME = 'vit_tiny_patch16_224'
    TARGET_SIZE = (64, 224, 224)
    BATCH_SIZE = 4
    PET_CSV_PATH = 'All_Subjects_UCBERKELEY_TAUPVC_6MM_08Aug2025.csv'
    CDR_CSV_PATH = 'All_Subjects_CDR_08Aug2025.csv'
    NPY_BASE_DIR = Path(r"E:\Preprocessed_PET_NPY\tau_pet")
    OUTPUT_FIGURE_NAME = 'tau_attention_stratified_comparison.pdf'
    # --- 配置区结束 ---

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- 开始执行分层模型评估 (Tau PET) ---")
    print(f"--- 使用设备: {device} ---")

    # --- 1. 准备包含CDR分数的验证数据 ---
    print("\n--- [步骤 1/4] 准备验证数据加载器 (包含CDR分数) ---")
    try:
        pet_df = pd.read_csv(PET_CSV_PATH)
        cdr_df = pd.read_csv(CDR_CSV_PATH)
    except FileNotFoundError as e:
        print(f"!! 致命错误: 找不到CSV文件: {e}。")
        return

    pet_df = pet_df[['PTID', 'VISCODE2', 'LONIUID']].copy()
    cdr_df = cdr_df[['PTID', 'VISCODE2', 'CDGLOBAL']].copy()
    cdr_df.dropna(subset=['CDGLOBAL'], inplace=True)
    cdr_df['CDGLOBAL'] = pd.to_numeric(cdr_df['CDGLOBAL'], errors='coerce')
    cdr_df.dropna(subset=['CDGLOBAL'], inplace=True)

    merged_df = pd.merge(pet_df, cdr_df, on=['PTID', 'VISCODE2'], how='inner')
    merged_df['label'] = (merged_df['CDGLOBAL'] > 0).astype(int)
    merged_df['filepath'] = merged_df['LONIUID'].apply(lambda x: str(NPY_BASE_DIR / f"I{x}.npy"))
    merged_df = merged_df[merged_df['filepath'].apply(os.path.exists)]

    if merged_df.empty:
        print("\n!! 致命错误: 数据准备失败。")
        return

    final_df = merged_df[['PTID', 'filepath', 'label', 'CDGLOBAL']].copy()
    patient_ids = final_df['PTID'].unique()
    _, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    val_df = final_df[final_df['PTID'].isin(val_ids)]

    # 使用不进行transform的数据集进行ROI分析
    val_dataset_for_roi = PetCdrDataset(val_df, target_size=TARGET_SIZE, transform_func=None)

    val_dataset = PetCdrDataset(val_df, target_size=TARGET_SIZE, transform_func=resize_and_normalize)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
                            collate_fn=collate_fn_stratified)
    print(f"--- 验证数据加载器创建完成，包含 {len(val_df)} 条记录。")

    # --- 2. 加载模型并进行标准评估 ---
    print(f"\n--- [步骤 2/4] 加载模型 '{MODEL_TO_EVALUATE}' 并进行标准评估 ---")
    model = PretrainedVisionTransformer2_5D(model_name=MODEL_NAME).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_TO_EVALUATE))
    except FileNotFoundError:
        print(f"!! 致命错误: 找不到模型文件 '{MODEL_TO_EVALUATE}'。")
        return
    model.eval()

    attention_groups = collections.defaultdict(list)

    with torch.no_grad():
        for inputs, labels, cdr_scores, _ in tqdm(val_loader, desc="评估并提取注意力图"):
            if inputs.numel() == 0: continue
            inputs = inputs.to(device)
            with autocast(device_type=device):
                outputs, attn_weights = model(inputs, return_attention=True)

            for i in range(inputs.shape[0]):
                cdr = cdr_scores[i].item()
                attn_map = attn_weights[i].detach().cpu().numpy()
                if cdr == 0:
                    attention_groups['Normal (CDR=0)'].append(attn_map)
                elif cdr == 0.5:
                    attention_groups['MCI (CDR=0.5)'].append(attn_map)
                elif cdr >= 1.0:
                    attention_groups['Dementia (CDR>=1.0)'].append(attn_map)

    # --- 3. 生成分层注意力对比图 ---
    print(f"\n--- [步骤 3/4] 生成分层注意力对比图 ---")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(22, 12))
    indices = np.arange(TARGET_SIZE[0])
    num_groups = len(attention_groups)
    bar_width = 0.8 / num_groups if num_groups > 0 else 0.8
    colors = {'Normal (CDR=0)': 'green', 'MCI (CDR=0.5)': 'orange', 'Dementia (CDR>=1.0)': 'firebrick'}

    print("--- 注意力分析分组统计 ---")
    for name, maps in attention_groups.items():
        print(f"- {name}: {len(maps)} 个样本")
    print("------------------------")

    for i, (name, maps) in enumerate(attention_groups.items()):
        if not maps: continue
        mean_scores = np.mean(maps, axis=0)
        std_scores = np.std(maps, axis=0)
        position = indices - (bar_width * (num_groups - 1) / 2) + (i * bar_width)
        ax.bar(position, mean_scores, bar_width, yerr=std_scores,
               capsize=3, color=colors.get(name, 'gray'), alpha=0.8,
               label=f'{name} (N={len(maps)})')

    ax.set_title('Average Attention Score per Slice (Stratified by Disease Stage)', fontsize=20, weight='bold')
    ax.set_xlabel('Slice Index (0=Bottom -> 63=Top)', fontsize=16)
    ax.set_ylabel('Average Attention Score', fontsize=16)
    ax.set_xticks(indices[::2])
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=14)
    ax.set_xlim(-1, TARGET_SIZE[0])
    fig.tight_layout(pad=2.0)
    fig.savefig(OUTPUT_FIGURE_NAME, bbox_inches='tight', dpi=300)
    print(f"\n--- 分层注意力对比图已成功保存为 '{OUTPUT_FIGURE_NAME}' ---")

    # --- 4. 执行健全性检查 ---
    perform_roi_sanity_check(val_dataset_for_roi)


if __name__ == '__main__':
    evaluate_stratified()
