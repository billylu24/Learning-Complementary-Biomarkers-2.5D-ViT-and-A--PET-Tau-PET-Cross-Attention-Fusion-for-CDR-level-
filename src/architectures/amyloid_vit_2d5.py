# -*- coding: utf-8 -*-
"""
全自动PET影像分类与2.5D Vision Transformer训练流水线 (V14.1 - 路径修正版)
================================================================================
本脚本修正了文件路径，使其能够根据当前文件位置动态定位项目中的其他文件。
【算法保持不变】: 核心策略仍然是2.5D ViT训练，包含早停、AMP等机制。

核心策略:
- 使用一个预训练的Vision Transformer (ViT)作为2D切片特征提取器。
- 对所有切片的特征进行序列化，并使用一个注意力机制进行聚合。
- 训练整个模型以进行端到端的3D影像分类。

前置要求:
- 项目结构符合规范 (例如，dataset, models等目录在项目根目录下)。
- pip install timm torch pandas scikit-learn SimpleITK tqdm matplotlib
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    """【算法未变】早停机制，用于在验证损失不再改善时停止训练。"""

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
            f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class PetCdrDataset(Dataset):
    """【算法未变】自定义PyTorch数据集，用于加载和预处理PET影像。"""

    def __init__(self, dataframe, target_size, transform_func=None):
        self.df = dataframe
        self.transform = transform_func
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filepath = self.df.iloc[idx]['filepath']
        label = self.df.iloc[idx]['label']
        try:
            volume = np.load(filepath).astype(np.float32)
        except FileNotFoundError:
            print(f"!! 警告: 找不到文件 {filepath}，将跳过此样本。")
            return torch.empty(0), torch.empty(0)

        if self.transform:
            volume = self.transform(volume, self.target_size)
        label = torch.tensor(label, dtype=torch.float32)
        return volume, label


def resize_and_normalize(volume_np, target_size):
    """【算法未变】使用SimpleITK进行重采样和归一化"""
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


def collate_fn(batch):
    """【算法未变】自定义的collate_fn，过滤加载失败的样本。"""
    batch = list(filter(lambda x: x[0].numel() > 0, batch))
    if not batch: return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)


class PositionalEncoding(nn.Module):
    """【算法未变】为序列添加位置信息"""

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
    """【算法未变】使用预训练ViT的2.5D模型。"""

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
            nn.Linear(self.embed_dim, self.embed_dim * 4), nn.GELU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
        )
        self.head = nn.Sequential(nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, num_classes))

    def forward(self, x, return_attention=False):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        slice_features = self.vit_2d(x).view(B, D, self.embed_dim).permute(1, 0, 2)
        slice_features = self.slice_pos_embed(slice_features)
        attn_output, attn_weights = self.agg_attention(slice_features, slice_features, slice_features)
        attn_output = self.agg_norm(attn_output + slice_features)
        final_feature = self.agg_linear(attn_output)[0]
        output = self.head(final_feature).squeeze(-1)
        if return_attention: return output, attn_weights
        return output


# ======================================================================
# --- [步骤一: 创建数据清单] ---
# ======================================================================
def create_adni_data_list():
    """加载并处理ADNI的CSV文件，以创建用于训练的精确数据清单。"""
    # --- 【已修正】配置区使用动态路径 ---
    DATASET_ROOT = PROJECT_ROOT / "dataset"
    CSV_DIR = DATASET_ROOT / "csv"
    NPY_ROOT = DATASET_ROOT / "Preprocessed_PET_NPY"

    pet_csv_path = CSV_DIR / 'All_Subjects_UCBERKELEY_AMY_6MM_08Aug2025.csv'
    cdr_csv_path = CSV_DIR / 'All_Subjects_CDR_08Aug2025.csv'
    npy_base_dir = NPY_ROOT / "amyloid_pet"
    # --- 配置区结束 ---

    print("--- [步骤 1/3] 开始创建ADNI数据清单 ---")
    try:
        pet_df = pd.read_csv(pet_csv_path)
        cdr_df = pd.read_csv(cdr_csv_path)
    except FileNotFoundError as e:
        print(f"!! 致命错误: 找不到CSV文件: {e}。请检查项目结构和文件名。")
        return None, None

    # --- 后续数据处理逻辑未变 ---
    pet_df = pet_df[['PTID', 'VISCODE2', 'LONIUID']].copy()
    cdr_df = cdr_df[['PTID', 'VISCODE2', 'CDGLOBAL']].copy()
    cdr_df.dropna(subset=['CDGLOBAL'], inplace=True)
    cdr_df['CDGLOBAL'] = pd.to_numeric(cdr_df['CDGLOBAL'], errors='coerce')
    cdr_df.dropna(subset=['CDGLOBAL'], inplace=True)
    merged_df = pd.merge(pet_df, cdr_df, on=['PTID', 'VISCODE2'], how='inner')
    merged_df['label'] = (merged_df['CDGLOBAL'] > 0).astype(int)
    merged_df['filepath'] = merged_df['LONIUID'].apply(lambda x: str(npy_base_dir / f"I{x}.npy"))
    initial_count = len(merged_df)
    merged_df = merged_df[merged_df['filepath'].apply(os.path.exists)]
    print(f"--- 文件有效性检查: 移除了 {initial_count - len(merged_df)} 个找不到对应.npy文件的记录。")

    if merged_df.empty:
        print("\n!! 致命错误: 数据准备失败，没有任何匹配项。")
        return None, None

    final_df = merged_df[['PTID', 'filepath', 'label']].copy()
    patient_ids = final_df['PTID'].unique()
    train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    train_df = final_df[final_df['PTID'].isin(train_ids)].drop(columns=['PTID'])
    val_df = final_df[final_df['PTID'].isin(val_ids)].drop(columns=['PTID'])

    print(f"--- 数据清单创建成功！总共 {len(final_df)} 条有效扫描记录。")
    print(f"--- 训练集: {len(train_df)} 条 | 验证集: {len(val_df)} 条。")
    return train_df, val_df


# ======================================================================
# --- [步骤二: 训练模型] ---
# ======================================================================
def train_model(train_df, val_df):
    """接收准备好的数据，构建并训练模型。"""
    # --- 配置区 (算法相关超参数未变) ---
    TARGET_SIZE = (64, 224, 224)
    BATCH_SIZE = 4
    EPOCHS = 30
    LEARNING_RATE = 1e-5
    PRETRAINED_MODEL_NAME = 'vit_tiny_patch16_224'
    EARLY_STOP_PATIENCE = 5

    # --- 【已修正】模型路径配置区使用动态路径 ---
    MODELS_ROOT = PROJECT_ROOT / "models"
    UNIMODAL_MODELS_DIR = MODELS_ROOT / "pet_unimodal"
    UNIMODAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    BEST_MODEL_PATH = UNIMODAL_MODELS_DIR / 'adni_pretrained_vit_tiny_best.pth'
    # --- 配置区结束 ---

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n--- [步骤 2/3] 开始执行模型训练 ---")
    print(f"--- 将使用设备: {DEVICE} ---")

    train_dataset = PetCdrDataset(train_df, target_size=TARGET_SIZE, transform_func=resize_and_normalize)
    val_dataset = PetCdrDataset(val_df, target_size=TARGET_SIZE, transform_func=resize_and_normalize)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
                            collate_fn=collate_fn)
    print(f"数据加载器创建完成: {len(train_loader)} 批次训练, {len(val_loader)} 批次验证")

    model = PretrainedVisionTransformer2_5D(model_name=PRETRAINED_MODEL_NAME).to(DEVICE)
    print("--- 模型架构 ---")
    print(f"使用预训练模型: {PRETRAINED_MODEL_NAME}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scaler = GradScaler()

    # 【已修正】初始化早停机制，传入动态路径
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, verbose=True, path=BEST_MODEL_PATH)

    # --- 训练循环逻辑未变 ---
    for epoch in range(EPOCHS):
        model.train()
        train_correct, total_train_samples = 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [训练中]", leave=False)
        for inputs, labels in progress_bar:
            if inputs.numel() == 0: continue
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            preds = torch.sigmoid(outputs)
            train_correct += (torch.round(preds) == labels).sum().item()
            total_train_samples += labels.size(0)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_correct, total_val_samples, total_val_loss = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                if inputs.numel() == 0: continue
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                with autocast(device_type=DEVICE):
                    outputs = model(inputs)
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

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("早停机制触发，训练提前结束。")
            break

    # 【已修正】返回动态路径以供评估函数使用
    return BEST_MODEL_PATH, val_loader, DEVICE, PRETRAINED_MODEL_NAME


# ======================================================================
# --- [步骤三: 最终评估] ---
# ======================================================================
def final_evaluation(model_path, val_loader, device, model_name):
    """【逻辑未变】加载最佳模型并在验证集上进行详细评估。"""
    print("\n--- [步骤 3/3] 开始最终评估 ---")
    model = PretrainedVisionTransformer2_5D(model_name=model_name).to(device)
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"--- 成功加载最佳模型: '{model_path}' ---")
    except FileNotFoundError:
        print(f"!! 错误: 找不到模型文件 '{model_path}'。")
        return

    model.eval()
    all_labels, all_preds_probs = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="正在验证集上评估"):
            if inputs.numel() == 0: continue
            inputs, labels_cpu = inputs.to(device), labels  # Keep labels on CPU
            with autocast(device_type=device):
                outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_labels.extend(labels_cpu.numpy())
            all_preds_probs.extend(probs.cpu().numpy().flatten())

    if not all_labels:
        print("!! 验证集中没有有效数据，无法生成评估报告。")
        return

    all_preds_binary = np.round(all_preds_probs)
    print("\n=================================================")
    print("---            最终模型评估报告            ---")
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

    print("\n--- 4. 注意力机制可视化 (前4个样本) ---")
    try:
        inputs, labels = next(iter(val_loader))
        inputs = inputs.to(device)
        with torch.no_grad():
            with autocast(device_type=device):
                _, attn_weights = model(inputs, return_attention=True)
        for i in range(min(4, inputs.shape[0])):
            # 注意力权重形状为 (num_heads, target_len, source_len), 我们需要对 head 取平均
            # 在自注意力中, target_len=source_len=D (切片数)
            slice_importance = attn_weights.mean(dim=0)[i].mean(dim=0).cpu().numpy()
            top5_indices = np.argsort(slice_importance)[-5:][::-1]
            print(f"\n[样本 {i + 1}] 真实标签: {'异常' if labels[i] == 1 else '正常'}")
            print(f"  -> 模型认为最重要的5个切片索引是: {top5_indices}")
    except Exception as e:
        print(f"!! 生成注意力图时发生错误: {e}")


# ======================================================================
# --- [主执行区] ---
# ======================================================================
if __name__ == '__main__':
    train_dataframe, val_dataframe = create_adni_data_list()

    if train_dataframe is not None and val_dataframe is not None:
        # 【已修正】适配新的函数返回值
        best_model_path, validation_loader, device_in_use, final_model_name = train_model(
            train_dataframe, val_dataframe
        )

        # 【已修正】增加文件存在性检查
        if best_model_path and os.path.exists(best_model_path):
            final_evaluation(best_model_path, validation_loader, device_in_use, final_model_name)
        else:
            print(f"\n!! 训练未生成最佳模型文件或文件不存在于 '{best_model_path}'，跳过最终评估。")
    else:
        print("\n!! 流水线因数据清单创建失败而终止。!!")