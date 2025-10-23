好的，这是移除了 "如何使用 (示例)" 和 "引用" 部分的 README。

---

# Learning Complementary Biomarkers: 2.5D ViT and Aβ-PET/Tau-PET Cross-Attention Fusion for CDR-level Alzheimer’s Classification

> This project implements the models described in the paper "Learning Complementary Biomarkers: 2.5D ViT and Cross-Attention Fusion of Amyloid-PET and Tau-PET for CDR-level Alzheimer’s Classification".
>
> The goal of this project is to classify patient-level Alzheimer's status (CDR>0 vs CDR=0) using complementary **Amyloid-PET (Aβ-PET)** and **Tau-PET** signals.

## Introduction

Aβ-PET and Tau-PET capture complementary biological information in Alzheimer's disease. This project aims to efficiently fuse these two modalities to classify cognitive impairment (CDR>0 vs CDR=0).

The project addresses three main challenges:
1.  **Data Efficiency:** Handling the modest sample size of paired scan data.
2.  **Leakage Prevention:** Controlling data leakage by using strict **patient-level** splits.
3.  **Interpretability:** Understanding which slices and which biomarkers drive the model's predictions.

## Core Features

**2.5D ViT Model:** A data-efficient 2.5D Vision Transformer (ViT) that encodes 2D axial slices and applies attention across the depth dimension[cite: 2, 11].
**Cross-Attention Fusion:** A lightweight fusion module that allows Aβ and Tau features to "query" each other[cite: 3, 12, 74].
**Interpretability:** The resulting slice-attention maps align with known biological patterns of Aβ (cortical) and Tau (medial-temporal)[cite: 4, 110, 115].

## Model Architecture

### 1. Unimodal 2.5D ViT Encoder

* [cite_start]Each PET volume (Aβ or Tau) is treated as a sequence of axial slices along the depth axis[cite: 61].
* [cite_start]A pre-trained 2D ViT backbone encodes each individual 2D slice[cite: 59].
* [cite_start]A multi-head self-attention block aggregates all slice features across the depth dimension to produce a single global representation[cite: 62].

### 2. Multimodal Fusion

[cite_start]During fusion training, the unimodal encoders remain **frozen**[cite: 68, 79, 93]. Two fusion strategies were compared:

* **Baseline: Feature-concatenation**
    * [cite_start]The global feature vectors from Aβ and Tau are simply concatenated and passed to a small MLP for classification[cite: 69, 70].
* **Our Method: Cross-attention**
    * [cite_start]Before concatenation, the Aβ and Tau global feature vectors are processed by a cross-attention block, allowing the features to interact and query each other[cite: 74, 76, 78].

## Dataset & Preprocessing

* [cite_start]**Data Source:** ADNI [cite: 24]
* [cite_start]**Data Modalities:** Amyloid-PET (Aβ) and Tau-PET [cite: 24]
* [cite_start]**Labels:** Binary label based on global CDR score (y=1 if CDR>0, else 0) [cite: 25]
* [cite_start]**Data Split:** Strict patient-level 80/20 stratified split [cite: 26, 83, 87]
* **Preprocessing:**
    * [cite_start]**Normalization:** Per-scan min-max normalization to `[0, 1]` [cite: 31, 39]
    * [cite_start]**Resampling:** Using `SimpleITK` with linear interpolation [cite: 32, 39]
    * [cite_start]**Target Grid (2.5D-ViT):** `(64, 224, 224)` (i.e., 64 axial slices at 224x224) [cite: 34, 39]

## Main Results

[cite_start]Performance comparison on the paired validation set (N = 138)[cite: 103, 105]:

| Fusion Strategy | Accuracy | ROC-AUC | Macro-F1 |
| :--- | :---: | :---: | :---: |
| Feature Concatenation (Baseline) | [cite_start]0.63 [cite: 105] | [cite_start]0.6764 [cite: 105] | [cite_start]0.61 [cite: 105] |
| **Cross-Attention (Our Method)** | [cite_start]**0.7609** [cite: 105] | [cite_start]**0.8341** [cite: 105] | [cite_start]**0.7587** [cite: 105] |

[cite_start]Cross-attention significantly outperformed simple feature concatenation across all metrics[cite: 109, 117].
