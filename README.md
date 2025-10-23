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

**2.5D ViT Model:** A data-efficient 2.5D Vision Transformer (ViT) that encodes 2D axial slices and applies attention across the depth dimension.
**Cross-Attention Fusion:** A lightweight fusion module that allows Aβ and Tau features to "query" each other.
**Interpretability:** The resulting slice-attention maps align with known biological patterns of Aβ (cortical) and Tau (medial-temporal).

## Model Architecture

### 1. Unimodal 2.5D ViT Encoder

* Each PET volume (Aβ or Tau) is treated as a sequence of axial slices along the depth axis.
* A pre-trained 2D ViT backbone encodes each individual 2D slice.
* A multi-head self-attention block aggregates all slice features across the depth dimension to produce a single global representation.

### 2. Multimodal Fusion

During fusion training, the unimodal encoders remain **frozen**. Two fusion strategies were compared:

* **Baseline: Feature-concatenation**
    * The global feature vectors from Aβ and Tau are simply concatenated and passed to a small MLP for classification.
* **Our Method: Cross-attention**
    * Before concatenation, the Aβ and Tau global feature vectors are processed by a cross-attention block, allowing the features to interact and query each other.

## Dataset & Preprocessing

* **Data Source:** ADNI 
* **Data Modalities:** Amyloid-PET (Aβ) and Tau-PET 
* **Labels:** Binary label based on global CDR score (y=1 if CDR>0, else 0) 
* **Data Split:** Strict patient-level 80/20 stratified split 
* **Preprocessing:**
    * **Normalization:** Per-scan min-max normalization to `[0, 1]` 
    * **Resampling:** Using `SimpleITK` with linear interpolation
    * **Target Grid (2.5D-ViT):** `(64, 224, 224)` (i.e., 64 axial slices at 224x224)

## Main Results

Performance comparison on the paired validation set (N = 138):

| Fusion Strategy | Accuracy | ROC-AUC | Macro-F1 |
| :--- | :---: | :---: | :---: |
| Feature Concatenation (Baseline) | 0.63        | 0.6764       | 0.61  |
| **Cross-Attention (Our Method)** | **0.7609**  | **0.8341**   | **0.7587**  |

Cross-attention significantly outperformed simple feature concatenation across all metrics.
