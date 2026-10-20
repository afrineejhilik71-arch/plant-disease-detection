# Hybrid CNN Architecture for Robust Plant Disease Classification

## Project Overview

This repository contains the code and preliminary experiments for my master's thesis, **“Hybrid CNN Architecture for Robust Multiclass Plant Disease Classification Across Diverse Datasets.”**

The research investigates deep learning methods for plant disease classification, establishes baseline performance using pretrained models, and aims to develop and evaluate a hybrid CNN architecture. A key objective is to investigate model robustness and generalization across different plant disease classes and datasets.

## Research Objectives

The main objectives of this project are to:

- Analyse and prepare plant disease image datasets for experimentation.
- Establish baseline performance using pretrained deep learning models.
- Investigate a hybrid CNN architecture for multiclass plant disease classification.
- Compare models using classification metrics, including macro-F1.
- Investigate generalization across crop classes and diverse datasets.
- Explore explainable AI (XAI) methods to understand model predictions.

## Dataset

The initial baseline experiments use the **Tomato subset of the AgriVision4 plant disease dataset**.

The subset contains five classes:

- Downy Mildew
- Healthy
- Mosaic
- Spot
- White Spot

The dataset preparation workflow includes image inventory, duplicate-image detection using SHA-256 hashes, identification of identical image contents assigned to different class labels, and preparation of training, validation, and test splits.

Images with conflicting labels were provisionally excluded from the initial Tomato experiment. Identical image contents were considered during split preparation to reduce the risk of duplicate-image leakage between subsets.

The dataset images are not included in this repository and must be obtained separately.

## Repository Contents

### `01_agrivision_dataset_analysis.ipynb`

Contains dataset inspection and analysis workflows for AgriVision4, including:

- Image and class-count summaries.
- Duplicate-image detection.
- Identification of potential cross-class label conflicts.
- Crop-level dataset analysis.
- Preparation and inspection of experiment file lists and data splits.

### `02_ResNet50_Baseline.ipynb`

Contains the initial baseline workflow for the Tomato subset, including dataset inspection, duplicate and conflicting-label checks, and experimental data preparation. The repository
