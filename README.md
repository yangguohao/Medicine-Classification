content_en = """# 2026 iFLYTEK AI Developer Challenge - Traditional Chinese Medicine (TCM) Recognition Challenge Solution

[![Competition Link](https://img.shields.io/badge/Competition-iFLYTEK-blue.svg)](https://challenge.xfyun.cn/topic/info?type=tcm-recognition&ch=dwsf259)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the complete solution and source code for the **2026 iFLYTEK AI Developer Challenge - Traditional Chinese Medicine (TCM) Recognition Challenge**.

## 📖 Introduction

### Background
Driven by national policies supporting the inheritance and innovation of Traditional Chinese Medicine (TCM), improving the quality control of herbs is of paramount importance. TCM species are numerous and complex, with many looking visually identical, making manual inspection error-prone and inefficient. This competition aims to leverage Artificial Intelligence (AI) to achieve rapid and precise image recognition of TCM herbs, significantly advancing quality control and modernization in the TCM industry.

### Task
Based on the provided TCM image dataset, design and train an AI model capable of efficiently and accurately identifying and classifying different types of Traditional Chinese Medicine (54-class classification task).

### Dataset
The dataset contains approximately **54** categories of TCM herbs and their corresponding images, divided into training and testing subsets.
Example category label mapping:
- `Honeysuckles` : 0
- `Gardenia` : 1
- `Dandelions` : 32
- `Ginsengs` : 20
- ... total of 54 classes (0 to 53)

### Evaluation Metric
Submissions are evaluated using the **Macro F1-score**.

---

## 🏆 Final Results

* **Leaderboard A Score**: `0.953` (Rank: `2`)
* **Leaderboard B Score**: `0.946` (Rank: `3`)
* **Final Award**: `3rd Prize`

---

## 💡 Solution Architecture

> **Note**: Feel free to adapt and expand this section to match your actual implementation.

### 1. Data Preprocessing & Augmentation
* **Basic Augmentation**: Random Crop, Random Horizontal/Vertical Flips, Color Jittering.
* **Advanced Strategies**: Implemented `MixUp` and `CutMix` to enhance the model's generalization capabilities on fine-grained features.
* **Class Imbalance**: Utilized class-aware resampling (Oversampling) and `Focal Loss` to alleviate data scarcity for certain rare herb species.

### 2. Model Architecture
* **Backbone**: `eva02, convnext02, caformer` three models ensemble, initialized with pre-trained weights.
* **Modification**: Replaced the original fully connected classification layer with a custom linear head adapted for 54 classes, integrated with Dropout to prevent overfitting.

### 3. Training & Optimization
* **Optimizer**: AdamW (Weight Decay = 1e-4)
* **LR Scheduler**: CosineAnnealingWarmRestarts (Cosine annealing with linear warmup)
* **Loss Function**: CrossEntropyLoss with Label Smoothing (smoothing factor = 0.1)
* **Inference Tricks**: Applied Test-Time Augmentation (TTA) to ensure prediction stability; employed a multi-model ensemble strategy to maximize the final Macro F1 score.

Trained Model could be found [here](https://huggingface.co/yangguohao/medicine-classification)
