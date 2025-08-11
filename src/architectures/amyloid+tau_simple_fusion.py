# -*- coding: utf-8 -*-
"""
全自动多模态(Amyloid+Tau)影像分类与模型融合训练流水线 (V1.1 - 路径修正版)
==================================================================================
本脚本修正了文件路径，使其能够根据当前文件位置动态定位项目中的其他文件。
【算法保持不变】: 核心策略仍然是2.5D ViT特征拼接融合。

核心策略: 特征层融合 (Feature-level Fusion)
- 使用已在单模态上训练好的两个2.5D ViT模型作为特征提取器。
- 冻结这两个模型的权重。
- 将它们提取的特征进行拼接。
- 训练一个新的、小型的分类头来对融合后的特征进行分类。

前置要求:
- 项目结构符合规范 (例如，dataset, models等目录在项目根目录下)。
- 预训练模型 'adni_pretrained_vit_tiny_best.pth' 和 'tau_adni_pretrained_vit_tiny_best.pth' 已存在于 'models/pet_unimodal/' 目录。
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

# 【已修正】获取项目根目录，以便构建绝对路径
# 假设此脚本位于: <PROJECT_ROOT>/src/architectures/your_script_name.py
# .parents[2] 会定位到 <PROJECT_ROOT>
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EarlyStopping:
    # --- 算法未变 ---
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience, self.verbose, self.delta, self.path = patience, verbose, delta, path
        self.counter, self.best_score, self.early_stop, self.val_loss_min = 0, None, False, np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose: print(
            f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        # 【注意】这里保存的是传入的模块的state_dict，符合原逻辑
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class FusionPetDataset(Dataset):
    # --- 算法未变 ---
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
    # --- 算法未变 ---
    min_val, max_val = np.min(volume_np), np.max(volume_np)
    if max_val > min_val:
        volume_np = (volume_np - min_val) / (max_val - min_val)
    sitk_image = sitk.GetImageFromArray(volume_np)
    original_spacing, original_size = sitk_image.GetSpacing(), sitk_image.GetSize()
    new_spacing = [orig_sz * orig_sp / targ_sz for orig_sz, orig_sp, targ_sz in
                   zip(original_size, original_spacing, target_size)]
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
    # --- 算法未变 ---
    batch = list(filter(lambda x: x[2] != -1, batch))
    if not batch: return torch.empty(0), torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)


class FeatureExtractor(nn.Module):
    # --- 算法未变 ---
    def __init__(self, model_name='vit_tiny_patch16_224'):
        super().__init__()
        self.vit_2d = timm.create_model(model_name, pretrained=True)
        self.embed_dim = self.vit_2d.embed_dim
        original_patch_embed = self.vit_2d.patch_embed.proj
        new_patch_embed = nn.Conv2d(1, self.embed_dim, kernel_size=original_patch_embed.kernel_size,
                                    stride=original_patch_embed.stride, padding=original_patch_embed.padding)
        with torch.no_grad():
            new_patch_embed.weight.copy_(original_patch_embed.weight.mean(1, keepdim=True))
            new_patch_embed.bias.copy_(original_patch_embed.bias)
        self.vit_2d.patch_embed.proj = new_patch_embed
        self.vit_2d.head = nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        slice_features = self.vit_2d(x)
        return slice_features.view(B, D, -1).mean(dim=1)


class FusionModel(nn.Module):
    # --- 算法未变 ---
    def __init__(self, amyl_model_path, tau_model_path, model_name='vit_tiny_patch16_224'):
        super().__init__()
        self.amyl_extractor = FeatureExtractor(model_name)
        self.tau_extractor = FeatureExtractor(model_name)

        # 【已修正】加载路径现在由外部传入
        print(f"--- 正在加载 Amyloid 主干网络权重: {amyl_model_path}")
        self.amyl_extractor.load_state_dict(torch.load(amyl_model_path), strict=False)
        print(f"--- 正在加载 Tau 主干网络权重: {tau_model_path}")
        self.tau_extractor.load_state_dict(torch.load(tau_model_path), strict=False)

        for param in self.amyl_extractor.parameters():
            param.requires_grad = False
        for param in self.tau_extractor.parameters():
            param.requires_grad = False

        embed_dim = self.amyl_extractor.embed_dim
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x_amyl, x_tau):
        self.amyl_extractor.eval()
        self.tau_extractor.eval()
        with torch.no_grad():
            features_amyl = self.amyl_extractor(x_amyl)
            features_tau = self.tau_extractor(x_tau)
        fused_features = torch.cat((features_amyl, features_tau), dim=1)
        return self.fusion_head(fused_features).squeeze(-1)


# ======================================================================
# --- [步骤一: 创建配对数据清单] ---
# ======================================================================
def create_multimodal_data_list():
    # --- 【已修正】配置区使用动态路径 ---
    DATASET_ROOT = PROJECT_ROOT / "dataset"
    CSV_DIR = DATASET_ROOT / "csv"
    NPY_ROOT = DATASET_ROOT / "Preprocessed_PET_NPY"

    # 使用原始脚本中的文件名
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
        print(f"!! 致命错误: 找不到CSV文件: {e}。请检查项目结构和文件名。")
        return None, None

    # --- 后续数据处理逻辑未变 ---
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
    # --- 配置区 (算法相关超参数未变) ---
    TARGET_SIZE = (64, 224, 224)
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    MODEL_NAME = 'vit_tiny_patch16_224'
    EARLY_STOP_PATIENCE = 10

    # --- 【已修正】模型路径配置区使用动态路径 ---
    MODELS_ROOT = PROJECT_ROOT / "models"
    UNIMODAL_MODELS_DIR = MODELS_ROOT / "pet_unimodal"
    MULTIMODAL_MODELS_DIR = MODELS_ROOT / "pet_multimodal"
    MULTIMODAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    # 使用原始脚本中的文件名
    AMYL_MODEL_PATH = UNIMODAL_MODELS_DIR / 'vit_2d5_amyloid_best.pth'
    TAU_MODEL_PATH = UNIMODAL_MODELS_DIR / 'vit_2d5_tau_best.pth'
    BEST_FUSION_MODEL_PATH = MULTIMODAL_MODELS_DIR / 'simple_fusion_best.pth'
    # --- 配置区结束 ---

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n--- [步骤 2/3] 开始执行融合模型训练 ---")
    print(f"--- 将使用设备: {DEVICE} ---")

    train_dataset = FusionPetDataset(train_df, target_size=TARGET_SIZE, transform_func=resize_and_normalize)
    val_dataset = FusionPetDataset(val_df, target_size=TARGET_SIZE, transform_func=resize_and_normalize)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
                              collate_fn=fusion_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
                            collate_fn=fusion_collate_fn)

    # 【已修正】将动态路径传入模型
    model = FusionModel(AMYL_MODEL_PATH, TAU_MODEL_PATH, model_name=MODEL_NAME).to(DEVICE)
    print("--- 融合模型架构 ---")
    print(f"将只训练以下分类头:")
    print(model.fusion_head)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.fusion_head.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scaler = GradScaler()
    # 【已修正】将动态路径传入EarlyStopping
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, verbose=True, path=BEST_FUSION_MODEL_PATH)

    # --- 训练循环逻辑未变 ---
    for epoch in range(EPOCHS):
        model.fusion_head.train()
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

        model.fusion_head.eval()
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

        early_stopping(avg_val_loss, model.fusion_head)
        if early_stopping.early_stop:
            print("早停机制触发，训练提前结束。")
            break

    # 【已修正】返回动态路径以供评估函数使用
    return BEST_FUSION_MODEL_PATH, val_loader, DEVICE, MODEL_NAME, AMYL_MODEL_PATH, TAU_MODEL_PATH


# ======================================================================
# --- [步骤三: 最终评估] ---
# ======================================================================
# 【已修正】函数签名，接收动态路径
def final_evaluation(model_path, val_loader, device, model_name, amyl_model_path, tau_model_path):
    print("\n--- [步骤 3/3] 开始最终评估 ---")

    # 【已修正】使用传入的动态路径加载模型
    model = FusionModel(amyl_model_path, tau_model_path, model_name=model_name).to(device)
    try:
        model.fusion_head.load_state_dict(torch.load(model_path))
        print(f"--- 成功加载最佳融合头: '{model_path}' ---")
    except FileNotFoundError:
        print(f"!! 错误: 找不到模型文件 '{model_path}'。")
        return

    # --- 评估逻辑未变 ---
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
    print("---      最终融合模型评估报告 (Amyloid+Tau)      ---")
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
        # 【已修正】适配新的函数返回值
        best_model_path, validation_loader, device_in_use, final_model_name, final_amyl_path, final_tau_path = train_fusion_model(
            train_dataframe, val_dataframe
        )
        # 【已修正】增加文件存在性检查，并适配新的函数参数
        if best_model_path and os.path.exists(best_model_path):
            final_evaluation(best_model_path, validation_loader, device_in_use, final_model_name, final_amyl_path,
                             final_tau_path)
        else:
            print(f"\n!! 训练未生成最佳模型文件或文件不存在于 '{best_model_path}'，跳过最终评估。")
    else:
        print("\n!! 流水线因数据清单创建失败而终止。!!")