# -*- coding: utf-8 -*-
"""
全自动Tau PET影像分类与3D-CNN训练流水线 (ADNI增强版)
============================================================
本脚本是一个“一键式”的完整解决方案，用于在Tau PET数据上训练和评估3D-CNN模型。

它会自动执行以下流程:
1.  **数据清单生成 (步骤一)**:
    - 加载ADNI官方的Tau PET影像信息CSV和临床CDR评分CSV。
    - **通过病人和访问代码(VISCODE)进行精确匹配**。
    - 使用一个病人的所有有效扫描数据。
    - 根据CDR评分生成真实的标签 (0为正常, 1为异常)。
    - **按病人ID划分训练集/验证集**，以严格防止数据泄露。
    - 将处理好的数据清单直接传入训练步骤。

2.  **模型训练 (步骤二)**:
    - 接收上一步准备好的训练和验证数据。
    - 构建、训练并验证一个3D卷积神经网络。
    - 保存性能最佳的模型。

3.  **最终评估 (步骤三)**:
    - 加载最佳模型，在验证集上进行最终评估。
    - **以文字形式输出准确率、召回率、AUC、混淆矩阵等详细指标**。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import SimpleITK as sitk


# ======================================================================
# --- [全局定义区] ---
# ======================================================================

class PetCdrDataset(Dataset):
    """自定义PyTorch数据集，用于加载和预处理PET影像。"""

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


def collate_fn(batch):
    """
    自定义的collate_fn，用于过滤掉在__getitem__中加载失败的样本。
    必须定义在全局作用域，以供多进程DataLoader使用。
    """
    batch = list(filter(lambda x: x[0].numel() > 0, batch))
    if not batch: return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)


# ======================================================================
# --- [步骤一: 创建数据清单] ---
# ======================================================================
def create_adni_data_list():
    """
    加载并处理ADNI的CSV文件，以创建用于训练的精确数据清单。
    该函数会处理病人多次访问的问题，并生成真实的标签。
    """
    # --- 配置区 (步骤一) ---
    pet_csv_path = 'All_Subjects_UCBERKELEY_TAUPVC_6MM_08Aug2025.csv'  # 【重要更新】
    cdr_csv_path = 'All_Subjects_CDR_08Aug2025.csv'
    npy_base_dir = Path(r"E:\Preprocessed_PET_NPY\tau_pet")  # 【重要更新】
    # --- 配置区结束 ---

    print("--- [步骤 1/3] 开始创建ADNI Tau PET数据清单 ---")

    # 1. 加载数据
    try:
        pet_df = pd.read_csv(pet_csv_path)
        cdr_df = pd.read_csv(cdr_csv_path)
    except FileNotFoundError as e:
        print(f"!! 致命错误: 找不到CSV文件: {e}。请确保文件与脚本在同一目录。")
        return None, None

    # 2. 数据预处理
    pet_df = pet_df[['PTID', 'VISCODE2', 'LONIUID']].copy()
    cdr_df = cdr_df[['PTID', 'VISCODE2', 'CDGLOBAL']].copy()
    cdr_df = cdr_df.dropna(subset=['CDGLOBAL'])
    cdr_df['CDGLOBAL'] = pd.to_numeric(cdr_df['CDGLOBAL'], errors='coerce')
    cdr_df = cdr_df.dropna(subset=['CDGLOBAL'])

    # 3. 通过病人和访问代码进行精确匹配
    merged_df = pd.merge(
        pet_df,
        cdr_df,
        on=['PTID', 'VISCODE2'],
        how='inner'
    )

    # 4. 创建最终的数据列
    merged_df['label'] = (merged_df['CDGLOBAL'] > 0).astype(int)
    merged_df['filepath'] = merged_df['LONIUID'].apply(lambda x: str(npy_base_dir / f"I{x}.npy"))

    # 5. 检查文件是否存在
    initial_count = len(merged_df)
    merged_df = merged_df[merged_df['filepath'].apply(lambda x: os.path.exists(x))]
    print(f"--- 文件有效性检查: 移除了 {initial_count - len(merged_df)} 个找不到对应.npy文件的记录。")

    # 6. 健壮性检查
    if merged_df.empty:
        print("\n!! 致命错误: 数据准备失败，没有任何匹配项。")
        return None, None

    # 7. 按病人ID划分训练集和验证集
    final_df = merged_df[['PTID', 'filepath', 'label']].copy()
    patient_ids = final_df['PTID'].unique()
    train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)

    train_df = final_df[final_df['PTID'].isin(train_ids)].drop(columns=['PTID'])
    val_df = final_df[final_df['PTID'].isin(val_ids)].drop(columns=['PTID'])

    print(f"--- 数据清单创建成功！---")
    print(f"--- 总共 {len(final_df)} 条有效扫描记录。")
    print(f"--- 训练集: {len(train_df)} 条记录 | 验证集: {len(val_df)} 条记录。")

    return train_df, val_df


# ======================================================================
# --- [步骤二: 训练3D-CNN模型] ---
# ======================================================================
def train_3d_cnn_model(train_df, val_df):
    """接收准备好的数据，构建并训练3D-CNN模型。"""

    # --- 配置区 (步骤二) ---
    TARGET_SIZE = (64, 128, 128)
    BATCH_SIZE = 4
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    BEST_MODEL_NAME = 'tau_adni_3d_cnn_best.pth'  # 【重要更新】
    # --- 配置区结束 ---

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n--- [步骤 2/3] 开始执行模型训练 ---")
    print(f"--- 将使用设备: {DEVICE} ---")

    # --- 创建DataLoader ---
    train_dataset = PetCdrDataset(train_df, target_size=TARGET_SIZE, transform_func=resize_and_normalize)
    val_dataset = PetCdrDataset(val_df, target_size=TARGET_SIZE, transform_func=resize_and_normalize)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
                            collate_fn=collate_fn)
    print(f"数据加载器创建完成: {len(train_loader)} 批次训练, {len(val_loader)} 批次验证")

    # --- 构建3D-CNN模型 ---
    class Pet3DCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool3d(2), nn.BatchNorm3d(64),
                nn.Conv3d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool3d(2), nn.BatchNorm3d(128),
                nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                nn.Linear(128, 256), nn.ReLU(),
                nn.Dropout(0.4), nn.Linear(256, 1)  # 【重要更新】移除Sigmoid
            )

        def forward(self, x): return self.model(x)

    model = Pet3DCNN().to(DEVICE)
    print(model)

    # --- 训练与验证循环 ---
    criterion = nn.BCEWithLogitsLoss()  # 【重要更新】使用更稳定的Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_correct, total_train_samples = 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [训练中]", leave=False)
        for inputs, labels in progress_bar:
            if inputs.numel() == 0: continue
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.sigmoid(outputs)  # 计算准确率时应用sigmoid
            train_correct += (torch.round(preds) == labels).sum().item()
            total_train_samples += labels.size(0)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_correct, total_val_samples, total_val_loss = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                if inputs.numel() == 0: continue
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).view(-1, 1)
                outputs = model(inputs)
                total_val_loss += criterion(outputs, labels).item()

                preds = torch.sigmoid(outputs)  # 计算准确率时应用sigmoid
                val_correct += (torch.round(preds) == labels).sum().item()
                total_val_samples += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        train_acc = train_correct / total_train_samples if total_train_samples > 0 else 0
        val_acc = val_correct / total_val_samples if total_val_samples > 0 else 0

        print(
            f"Epoch {epoch + 1}/{EPOCHS} -> Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_NAME)
            print(f"  -> 模型已保存至 '{BEST_MODEL_NAME}' (验证损失降低至: {best_val_loss:.4f})")

    return BEST_MODEL_NAME, val_loader, DEVICE


# ======================================================================
# --- [步骤三: 最终评估] ---
# ======================================================================
def final_evaluation(model_path, val_loader, device):
    """加载最佳模型并在验证集上进行详细评估。"""
    print("\n--- [步骤 3/3] 开始最终评估 ---")

    # --- 加载模型 ---
    class Pet3DCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool3d(2), nn.BatchNorm3d(64),
                nn.Conv3d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool3d(2), nn.BatchNorm3d(128),
                nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                nn.Linear(128, 256), nn.ReLU(),
                nn.Dropout(0.4), nn.Linear(256, 1)  # 【重要更新】移除Sigmoid
            )

        def forward(self, x): return self.model(x)

    model = Pet3DCNN().to(device)
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"--- 成功加载最佳模型: '{model_path}' ---")
    except FileNotFoundError:
        print(f"!! 错误: 找不到模型文件 '{model_path}'。无法进行最终评估。")
        return

    model.eval()

    # --- 收集所有预测和标签 ---
    all_labels = []
    all_preds_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="正在验证集上评估"):
            if inputs.numel() == 0: continue
            inputs = inputs.to(device)
            outputs = model(inputs)

            probs = torch.sigmoid(outputs)  # 将logits转换为概率
            all_labels.extend(labels.cpu().numpy())
            all_preds_probs.extend(probs.cpu().numpy().flatten())

    if not all_labels:
        print("!! 验证集中没有有效数据，无法生成评估报告。")
        return

    # 将概率转换为0或1的预测
    all_preds_binary = np.round(all_preds_probs)

    # --- 计算并打印评估指标 ---
    print("\n=================================================")
    print("---            最终模型评估报告 (Tau PET)            ---")
    print("=================================================\n")

    # 1. 分类报告 (精确率, 召回率, F1分数)
    print("--- 1. 分类报告 ---")
    report = classification_report(all_labels, all_preds_binary, target_names=['CDR=0 (正常)', 'CDR>0 (异常)'])
    print(report)

    # 2. 混淆矩阵
    print("--- 2. 混淆矩阵 ---")
    cm = confusion_matrix(all_labels, all_preds_binary)
    print("                 预测为正常   预测为异常")
    print(f"真实为正常 (TN, FP): {cm[0]}")
    print(f"真实为异常 (FN, TP): {cm[1]}\n")

    # 3. AUC
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
    # 确保在Windows上使用多进程时代码的安全性

    # 步骤一: 创建数据清单
    train_dataframe, val_dataframe = create_adni_data_list()

    # 如果数据准备成功，则继续执行步骤二
    if train_dataframe is not None and val_dataframe is not None:
        # 步骤二: 训练模型并获取必要信息
        best_model_path, validation_loader, device_in_use = train_3d_cnn_model(train_dataframe, val_dataframe)

        # 步骤三: 使用训练好的模型进行最终评估
        if best_model_path:
            final_evaluation(best_model_path, validation_loader, device_in_use)
    else:
        print("\n!! 流水线因数据清单创建失败而终止。!!")
