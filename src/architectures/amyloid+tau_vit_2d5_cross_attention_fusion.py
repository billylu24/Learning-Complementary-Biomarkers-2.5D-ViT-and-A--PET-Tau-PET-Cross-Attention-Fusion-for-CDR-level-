# -*- coding: utf-8 -*-
"""
全自动多模态(Amyloid+Tau)影像分类与模型融合训练流水线 (V3.2 - 路径修正版)
==================================================================================
本脚本修正了文件路径，使其能够根据当前文件位置动态定位项目中的其他文件。

核心策略: 交叉注意力引导的特征层融合 (Cross-Attention Guided Feature-level Fusion)
- 使用可训练的交叉注意力模块对两种模态的特征进行智能融合。
- 两个单模态2.5D ViT模型作为冻结的特征提取主干。
- 仅训练新的交叉注意力模块和最终的分类头。

前置要求:
- 项目结构符合规范。
- 预训练模型 'vit_2d5_amyloid_best.pth' 和 'vit_2d5_tau_best.pth' 已存在于 'models/pet_unimodal/' 目录。
- pip install timm torch pandas scikit-learn SimpleITK tqdm
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
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import SimpleITK as sitk
import timm

# ======================================================================
# --- [全局定义区] ---
# ======================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EarlyStopping:
    """早停机制，用于在验证损失不再改善时停止训练。"""

    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience, self.verbose, self.delta, self.path = patience, verbose, delta, path
        self.counter, self.best_score, self.early_stop, self.val_loss_min = 0, None, False, np.Inf

    def __call__(self, val_loss, model_state_dict):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_state_dict)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_state_dict)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_state_dict):
        if self.verbose: print(
            f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model_state_dict, self.path)
        self.val_loss_min = val_loss


class FusionPetDataset(Dataset):
    """为多模态融合定制的数据集，一次加载两种影像。"""

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
            return torch.empty(0), torch.empty(0), -1

        if self.transform:
            volume_amyl = self.transform(volume_amyl, self.target_size)
            volume_tau = self.transform(volume_tau, self.target_size)

        label = torch.tensor(label, dtype=torch.float32)
        return volume_amyl, volume_tau, label


def resize_and_normalize(volume_np, target_size):
    """使用SimpleITK进行重采样和归一化"""
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


def fusion_collate_fn(batch):
    """自定义的collate_fn，用于过滤掉加载失败的样本。"""
    batch = list(filter(lambda x: x[2] != -1, batch))
    if not batch: return torch.empty(0), torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)


class PositionalEncoding(nn.Module):
    """为序列添加位置信息"""

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
    """【主干网络】2.5D ViT模型结构"""

    def __init__(self, model_name='vit_tiny_patch16_224', num_classes=1, dropout=0.1):
        super().__init__()
        self.vit_2d = timm.create_model(model_name, pretrained=True)
        self.embed_dim = self.vit_2d.embed_dim
        num_heads = self.vit_2d.blocks[-1].attn.num_heads

        original_patch_embed = self.vit_2d.patch_embed.proj
        new_patch_embed = nn.Conv2d(1, self.embed_dim, kernel_size=original_patch_embed.kernel_size,
                                    stride=original_patch_embed.stride, padding=original_patch_embed.padding)
        with torch.no_grad():
            new_patch_embed.weight.copy_(original_patch_embed.weight.mean(1, keepdim=True))
            new_patch_embed.bias.copy_(original_patch_embed.bias)
        self.vit_2d.patch_embed.proj = new_patch_embed
        self.vit_2d.head = nn.Identity()

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

    def forward(self, x):
        final_feature = self.extract_feature_vector(x)
        output = self.head(final_feature).squeeze(-1)
        return output

    def extract_feature_vector(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        slice_features = self.vit_2d(x).view(B, D, self.embed_dim)
        slice_features = slice_features.permute(1, 0, 2)
        slice_features = self.slice_pos_embed(slice_features)

        attn_output, _ = self.agg_attention(slice_features, slice_features, slice_features)

        attn_output = self.agg_norm(attn_output + slice_features)
        final_feature = self.agg_linear(attn_output)
        final_feature = final_feature[0]

        return final_feature


class CrossAttentionFusion(nn.Module):
    """【新增】交叉注意力融合模块"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn_amyl_to_tau = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_tau_to_amyl = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim * 2)
        )
        self.norm3 = nn.LayerNorm(embed_dim * 2)

    def forward(self, features_amyl, features_tau):
        amyl_q = features_amyl.unsqueeze(1)
        tau_kv = features_tau.unsqueeze(1)
        tau_q = features_tau.unsqueeze(1)
        amyl_kv = features_amyl.unsqueeze(1)

        amyl_context, _ = self.cross_attn_amyl_to_tau(query=amyl_q, key=tau_kv, value=tau_kv)
        amyl_fused = self.norm1(amyl_q + amyl_context)

        tau_context, _ = self.cross_attn_tau_to_amyl(query=tau_q, key=amyl_kv, value=amyl_kv)
        tau_fused = self.norm2(tau_q + tau_context)

        fused_vector = torch.cat((amyl_fused, tau_fused), dim=-1).squeeze(1)
        fused_vector = self.norm3(fused_vector + self.ffn(fused_vector))
        return fused_vector


class FusionModel(nn.Module):
    """【V3.2 融合模型】"""

    def __init__(self, amyl_model_path, tau_model_path, model_name='vit_tiny_patch16_224'):
        super().__init__()
        self.amyl_extractor = PretrainedVisionTransformer2_5D(model_name=model_name)
        self.tau_extractor = PretrainedVisionTransformer2_5D(model_name=model_name)

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

        self.fusion_module = CrossAttentionFusion(embed_dim, num_heads)
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 1)
        )

    def forward(self, x_amyl, x_tau):
        self.amyl_extractor.eval()
        self.tau_extractor.eval()

        with torch.no_grad():
            features_amyl = self.amyl_extractor.extract_feature_vector(x_amyl)
            features_tau = self.tau_extractor.extract_feature_vector(x_tau)

        fused_features = self.fusion_module(features_amyl, features_tau)
        return self.classification_head(fused_features).squeeze(-1)


# ======================================================================
# --- [步骤一: 创建配对数据清单] ---
# ======================================================================
def create_multimodal_data_list():
    # --- 动态路径配置区 ---
    DATASET_ROOT = PROJECT_ROOT / "dataset"
    CSV_DIR = DATASET_ROOT / "csv"
    NPY_ROOT = DATASET_ROOT / "Preprocessed_PET_NPY"

    amyl_csv_path = CSV_DIR / 'All_Subjects_UCBERKELEY_AMY_6MM_10Aug2025.csv'
    tau_csv_path = CSV_DIR / 'All_Subjects_UCBERKELEY_TAUPVC_6MM_10Aug2025.csv'
    cdr_csv_path = CSV_DIR / 'All_Subjects_CDR_10Aug2025.csv'
    amyl_npy_dir = NPY_ROOT / "amyloid_pet"
    tau_npy_dir = NPY_ROOT / "tau_pet"
    # --- 配置区结束 ---

    print("--- [步骤 1/3] 开始创建多模态配对数据清单 ---")
    try:
        amyl_df = pd.read_csv(amyl_csv_path)[['PTID', 'VISCODE2', 'LONIUID']]
        tau_df = pd.read_csv(tau_csv_path)[['PTID', 'VISCODE2', 'LONIUID']]
        cdr_df = pd.read_csv(cdr_csv_path)[['PTID', 'VISCODE2', 'CDGLOBAL']]
    except FileNotFoundError as e:
        print(f"!! 致命错误: 找不到CSV文件: {e}。")
        return None, None

    paired_df = pd.merge(amyl_df, tau_df, on=['PTID', 'VISCODE2'], suffixes=('_amyl', '_tau'))
    cdr_df.dropna(subset=['CDGLOBAL'], inplace=True)
    cdr_df['CDGLOBAL'] = pd.to_numeric(cdr_df['CDGLOBAL'], errors='coerce')
    cdr_df.dropna(subset=['CDGLOBAL'], inplace=True)
    merged_df = pd.merge(paired_df, cdr_df, on=['PTID', 'VISCODE2'], how='inner')

    merged_df['label'] = (merged_df['CDGLOBAL'] > 0).astype(int)
    merged_df['filepath_amyl'] = merged_df['LONIUID_amyl'].apply(lambda x: str(amyl_npy_dir / f"I{x}.npy"))
    merged_df['filepath_tau'] = merged_df['LONIUID_tau'].apply(lambda x: str(tau_npy_dir / f"I{x}.npy"))

    merged_df = merged_df[merged_df['filepath_amyl'].apply(os.path.exists)]
    merged_df = merged_df[merged_df['filepath_tau'].apply(os.path.exists)]

    if merged_df.empty:
        print("\n!! 致命错误: 找不到任何配对的Amyl和Tau扫描记录。")
        return None, None

    final_df = merged_df[['PTID', 'filepath_amyl', 'filepath_tau', 'label']].copy()
    patient_ids = final_df['PTID'].unique()
    train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    train_df = final_df[final_df['PTID'].isin(train_ids)].drop(columns=['PTID'])
    val_df = final_df[final_df['PTID'].isin(val_ids)].drop(columns=['PTID'])

    print(f"--- 数据清单创建成功！总共找到 {len(final_df)} 条配对扫描记录。")
    print(f"--- 训练集: {len(train_df)} 条 | 验证集: {len(val_df)} 条。")
    return train_df, val_df


# ======================================================================
# --- [步骤二: 训练融合模型] ---
# ======================================================================
def train_fusion_model(train_df, val_df):
    # --- 配置区 ---
    TARGET_SIZE = (64, 224, 224)
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    MODEL_NAME = 'vit_tiny_patch16_224'
    EARLY_STOP_PATIENCE = 10

    # 【路径已更新】使用动态路径构建模型文件路径
    MODELS_ROOT = PROJECT_ROOT / "models"
    UNIMODAL_MODELS_DIR = MODELS_ROOT / "pet_unimodal"
    MULTIMODAL_MODELS_DIR = MODELS_ROOT / "pet_multimodal"
    MULTIMODAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    AMYL_MODEL_PATH = UNIMODAL_MODELS_DIR / 'vit_2d5_amyloid_best.pth'
    TAU_MODEL_PATH = UNIMODAL_MODELS_DIR / 'vit_2d5_tau_best.pth'
    BEST_FUSION_MODEL_PATH = MULTIMODAL_MODELS_DIR / 'amyloid+tau_vit_2d5_cross_attention_fusion_best_model.pth'
    # --- 配置区结束 ---

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n--- [步骤 2/3] 开始执行交叉注意力融合模型训练 ---")
    print(f"--- 将使用设备: {DEVICE} ---")

    train_dataset = FusionPetDataset(train_df, target_size=TARGET_SIZE, transform_func=resize_and_normalize)
    val_dataset = FusionPetDataset(val_df, target_size=TARGET_SIZE, transform_func=resize_and_normalize)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
                              collate_fn=fusion_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
                            collate_fn=fusion_collate_fn)

    model = FusionModel(AMYL_MODEL_PATH, TAU_MODEL_PATH, model_name=MODEL_NAME).to(DEVICE)
    trainable_params = list(model.fusion_module.parameters()) + list(model.classification_head.parameters())

    print("\n--- 融合模型架构 ---")
    print(f"将只训练以下可训练的模块:")
    print("1. CrossAttentionFusion Module")
    print("2. Classification Head")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.05)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, verbose=True, path=BEST_FUSION_MODEL_PATH)

    for epoch in range(EPOCHS):
        model.fusion_module.train()
        model.classification_head.train()

        train_correct, total_train_samples = 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [训练中]", leave=False)
        for amyl_vol, tau_vol, labels in progress_bar:
            if amyl_vol.numel() == 0: continue
            amyl_vol, tau_vol, labels = amyl_vol.to(DEVICE), tau_vol.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE):
                outputs = model(amyl_vol, tau_vol)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            preds = torch.sigmoid(outputs)
            train_correct += (torch.round(preds) == labels).sum().item()
            total_train_samples += labels.size(0)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        model.fusion_module.eval()
        model.classification_head.eval()
        val_correct, total_val_samples, total_val_loss = 0, 0, 0
        with torch.no_grad():
            for amyl_vol, tau_vol, labels in val_loader:
                if amyl_vol.numel() == 0: continue
                amyl_vol, tau_vol, labels = amyl_vol.to(DEVICE), tau_vol.to(DEVICE), labels.to(DEVICE)
                with autocast(device_type=DEVICE):
                    outputs = model(amyl_vol, tau_vol)
                    loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                preds = torch.sigmoid(outputs)
                val_correct += (torch.round(preds) == labels).sum().item()
                total_val_samples += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        train_acc = train_correct / total_train_samples if total_train_samples > 0 else 0
        val_acc = val_correct / total_val_samples if total_val_samples > 0 else 0
        print(
            f"Epoch {epoch + 1}/{EPOCHS} -> Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {avg_val_loss:.4f}")

        trainable_state_dict = {
            **{'fusion_module.' + k: v for k, v in model.fusion_module.state_dict().items()},
            **{'classification_head.' + k: v for k, v in model.classification_head.state_dict().items()}
        }
        early_stopping(avg_val_loss, trainable_state_dict)

        if early_stopping.early_stop:
            print("早停机制触发，训练提前结束。")
            break

    return BEST_FUSION_MODEL_PATH, val_loader, DEVICE, MODEL_NAME, AMYL_MODEL_PATH, TAU_MODEL_PATH


# ======================================================================
# --- [步骤三: 最终评估] ---
# ======================================================================
def final_evaluation(model_path, val_loader, device, model_name, amyl_model_path, tau_model_path):
    print("\n--- [步骤 3/3] 开始最终评估 ---")

    model = FusionModel(amyl_model_path, tau_model_path, model_name=model_name).to(device)
    try:
        # strict=False 允许只加载部分权重
        model.load_state_dict(torch.load(model_path), strict=False)
        print(f"--- 成功加载最佳融合模块和分类头: '{model_path}' ---")
    except FileNotFoundError:
        print(f"!! 错误: 找不到模型文件 '{model_path}'。")
        return

    model.eval()
    all_labels, all_preds_probs = [], []
    with torch.no_grad():
        for amyl_vol, tau_vol, labels in tqdm(val_loader, desc="正在验证集上评估"):
            if amyl_vol.numel() == 0: continue
            amyl_vol, tau_vol, labels = amyl_vol.to(device), tau_vol.to(device), labels.to(device)
            with autocast(device_type=device):
                outputs = model(amyl_vol, tau_vol)
            probs = torch.sigmoid(outputs)
            all_labels.extend(labels.cpu().numpy())
            all_preds_probs.extend(probs.cpu().numpy().flatten())

    if not all_labels:
        print("!! 验证集中没有有效数据，无法生成评估报告。")
        return

    all_preds_binary = np.round(all_preds_probs)

    print("\n=================================================")
    print("--- 最终融合模型评估报告 (V3.2 - 交叉注意力版) ---")
    print("=================================================\n")
    print("--- 1. 分类报告 ---")
    print(classification_report(all_labels, all_preds_binary, target_names=['CDR=0 (正常)', 'CDR>0 (异常)']))
    print("--- 2. 混淆矩阵 ---")
    cm = confusion_matrix(all_labels, all_preds_binary)
    print("                 预测为正常   预测为异常")
    print(f"真实为正常 (TN, FP): {cm[0]}")
    print(f"真实为异常 (FN, TP): {cm[1]}\n")
    try:
        auc_score = roc_auc_score(all_labels, all_preds_probs)
        print("--- 3. ROC AUC 分数 ---")
        print(f"AUC: {auc_score:.4f}\n")
    except ValueError:
        print("--- 3. ROC AUC 分数 ---")
        print("无法计算AUC，因为验证集中只包含一个类别。\n")
    print("=================================================")


# ======================================================================
# --- [主执行区] ---
# ======================================================================
if __name__ == '__main__':
    train_dataframe, val_dataframe = create_multimodal_data_list()

    if train_dataframe is not None and val_dataframe is not None:
        best_model_path, validation_loader, device_in_use, final_model_name, final_amyl_path, final_tau_path = train_fusion_model(
            train_dataframe, val_dataframe
        )
        if best_model_path and os.path.exists(best_model_path):
            final_evaluation(best_model_path, validation_loader, device_in_use, final_model_name, final_amyl_path,
                             final_tau_path)
        else:
            print(f"\n!! 训练未生成最佳模型文件或文件不存在于 '{best_model_path}'，跳过最终评估。")
    else:
        print("\n!! 流水线因数据清单创建失败而终止。!!")
