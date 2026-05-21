# Core Module — Brain Tumor Detection System

The `core` module contains the primary implementation and deep learning architecture for the Brain Tumor Detection System. It includes complete pipelines for brain tumour segmentation, tumour classification, MRI preprocessing, radiomics extraction, Grad-CAM visualization, tumour growth prediction, and survival analysis using advanced deep learning techniques in PyTorch.

This module was designed and developed as the central engine of the project and contains all major AI model implementations, training pipelines, preprocessing workflows, and optimization techniques used throughout the system.

---

# Files

## `brain_tumor_complete.py`

Complete end-to-end implementation of the Brain Tumor Detection System including:

- MRI preprocessing and normalization
- Skull stripping
- BraTS dataset support
- Kaggle MRI classification support
- Deep UNet 3+ architecture
- Tumour segmentation
- Tumour classification
- Radiomics feature extraction
- Grad-CAM explainability
- Synthetic MRI data generation
- Survival prediction support
- Diagnostic report generation
- Visualization utilities

This file contains the full research-oriented implementation with modular components and complete training/inference workflows.

---

## `brain_tumor_fast.py`

Optimized high-performance implementation designed for faster GPU training and inference.

Includes:

- CBAM attention modules
- Fast UNet 3+ architecture
- Mixed Precision Training (AMP)
- OneCycleLR scheduling
- Progressive resizing
- Test-Time Augmentation (TTA)
- Focal Loss and Boundary Loss
- Efficient segmentation pipeline
- Faster classification architecture
- Small tumour detection improvements
- Optimizations for Google Colab and T4 GPUs

This implementation significantly improves training speed and segmentation performance while maintaining high accuracy.

---

# Features

- Deep Learning based Brain Tumor Segmentation
- MRI Tumour Classification
- Multi-modal MRI support:
  - T1
  - T1ce
  - T2
  - FLAIR
- Deep UNet 3+ architecture
- CBAM Attention Mechanisms
- Radiomics Analysis
- Grad-CAM Explainability
- Synthetic Data Generation
- GPU Optimized Training
- Test-Time Augmentation
- Survival Prediction
- Tumour Growth Analysis
- Medical Image Preprocessing
- PyTorch-based modular implementation

---

# Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- SciPy
- Matplotlib
- Nibabel
- Radiomics
- CUDA

---

# Datasets Supported

- BraTS 2021 Dataset
- Kaggle Brain Tumor MRI Dataset

---

# Project Structure

```text
core/
│
├── brain_tumor_complete.py
└── brain_tumor_fast.py
```

---

# Purpose

The `core` module acts as the foundational AI engine of the Brain Tumor Detection System and provides all major machine learning, computer vision, preprocessing, segmentation, and classification functionality used throughout the project.

This module was created for medical imaging research, experimentation, academic learning, and AI-assisted diagnostic system development.

---

# Ownership

Original project architecture, implementation, optimization pipeline, and development created by ANKAN BARIK.

Unauthorized redistribution or claiming ownership of this work without permission is prohibited.
