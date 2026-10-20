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

Contains the initial baseline workflow for the Tomato subset, including dataset inspection, duplicate and conflicting-label checks, and experimental data preparation. The repository's original baseline notebook is retained for reference.

### `02_ResNet50_Baseline_Clean.ipynb`

Contains the cleaned Tomato baseline comparison workflow, covering:

- Dataset preparation and verification of the train, validation, and test split.
- Image preprocessing and data loaders.
- Training and fine-tuning of ResNet50, EfficientNet-B0, and DenseNet121.
- Checkpoint selection using validation macro-F1.
- Test-set evaluation and saving of results to a CSV file.

Although the notebook filename refers to ResNet50, it includes experiments for all three baseline architectures.

## Baseline Experiments

Three ImageNet-pretrained models were evaluated on the Tomato subset:

- **ResNet50**
- **EfficientNet-B0**
- **DenseNet121**

Validation macro-F1 was used to select the best checkpoint for each model. The selected checkpoints were then evaluated on the test set.

### Preliminary Results

| Model | Selected Epoch | Validation Accuracy | Validation Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|---:|
| ResNet50 | 4 | 58.82% | 0.3665 | 62.88% | 0.3928 |
| EfficientNet-B0 | 3 | 67.65% | 0.5387 | 71.21% | 0.5468 |
| DenseNet121 | 5 | 67.65% | 0.5737 | 74.24% | 0.6159 |

These results are preliminary and represent a single experimental split. DenseNet121 achieved the highest test accuracy and macro-F1 among these three models in this experiment. However, further experiments are needed to assess result stability, class-wise performance, and generalization before drawing broader conclusions.

## Current Research Status

The initial Tomato baseline experiments have been conducted using ResNet50, EfficientNet-B0, and DenseNet121.

The next stages of the research include:

1. Reviewing recent literature on hybrid CNN and Transformer-based architectures.
2. Improving the consistency and documentation of baseline experiments.
3. Evaluating additional baseline models where appropriate.
4. Investigating performance across more crop classes and datasets.
5. Designing and evaluating the proposed hybrid architecture.
6. Analysing model predictions using explainability techniques and investigating cross-dataset generalization.

The proposed hybrid architecture and broader generalization experiments remain part of the ongoing research.

## Technologies

The experiments use:

- Python
- PyTorch
- torchvision
- pandas
- scikit-learn
- Google Colab

GPU acceleration is used when available.

## Reproducibility and Data Availability

The notebooks document the current experimental workflow. Dataset images and trained model checkpoint files are not included in this repository.

The notebooks use Google Colab and Google Drive paths to access data and save experiment outputs. To reproduce the experiments, users must configure the required dataset paths, install compatible dependencies, and ensure that the data and pretrained weights are accessible.

Dataset access and use should comply with the applicable licence and terms.

## Research Purpose

This repository supports the development and evaluation of a hybrid CNN-based approach for robust multiclass plant disease classification. The long-term objective is to investigate whether the proposed approach can improve classification performance and generalization across diverse plant disease datasets.
