# Hybrid CNN Architecture for Robust Plant Disease Classification

## Project Overview

This repository contains the code and preliminary experiments for my master's thesis, **“Hybrid CNN Architecture for Robust Multiclass Plant Disease Classification Across Diverse Datasets.”**

The research aims to investigate deep learning methods for plant disease classification, establish baseline performance using pretrained models, and develop and evaluate a hybrid CNN architecture. A key objective is to investigate model robustness and generalization across different plant disease classes and datasets.

## Research Objectives

The main objectives of this project are to:

- Analyse and prepare plant disease image datasets for experimentation.
- Establish baseline performance using pretrained deep learning models.
- Investigate a hybrid CNN architecture for multiclass plant disease classification.
- Compare models using classification metrics, including macro-F1.
- Investigate generalization across crop classes and diverse datasets.
- Explore explainable AI methods to understand model predictions.

## Dataset

The initial baseline experiments use the **Tomato subset of the AgriVision4 plant disease dataset**.

The subset contains five classes:

- Downy Mildew
- Healthy
- Mosaic
- Spot
- White Spot

The dataset preparation workflow includes image inventory, duplicate-image detection using SHA-256 hashes, identification of identical image contents assigned to different class labels, and preparation of train, validation, and test splits.

Images with conflicting labels were provisionally excluded from the initial Tomato experiment. Identical image contents were considered when preparing the data split to reduce the risk of duplicate-image leakage between subsets.

The dataset files are not included in this repository. The experiments require access to the dataset separately.

## Repository Contents

### `01_agrivision_dataset_analysis.ipynb`

Contains dataset inspection and analysis workflows for AgriVision4, including:

- Image and class-count summaries.
- Duplicate-image detection.
- Identification of potential cross-class label conflicts.
- Crop-level dataset analysis.
- Preparation and inspection of experiment file lists and train, validation, and test splits.

### `02_ResNet50_Baseline.ipynb`

Contains the initial Tomato baseline workflow, including:

- Tomato dataset inspection and preparation.
- Duplicate and conflicting-label checks.
- Creation and verification of the experimental data split.
- Image preprocessing and data loaders.
- Training and fine-tuning of ResNet50, EfficientNet-B0, and DenseNet121.
- Validation-based checkpoint selection.
- Test-set evaluation and recording of results in a CSV file.

Although the filename refers to ResNet50, the notebook also contains the EfficientNet-B0 and DenseNet121 experiments.

## Baseline Experiments

Three pretrained models were evaluated on the Tomato subset:

- **ResNet50**
- **EfficientNet-B0**
- **DenseNet121**

The models use ImageNet-pretrained weights and the same intended train, validation, and test split. Validation macro-F1 was used to select the best checkpoint for each model.

### Preliminary Results

| Model | Selected Epoch | Validation Accuracy | Validation Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|---:|
| ResNet50 | 4 | 57.35% | 0.358 | 64.39% | 0.399 |
| EfficientNet-B0 | 5 | 66.91% | 0.550 | 73.48% | 0.573 |
| DenseNet121 | 5 | 59.56% | 0.408 | 71.21% | 0.516 |

These results are preliminary and represent one experimental split. They should not be interpreted as evidence that one architecture is universally superior. Further experiments are needed to assess result stability, class-wise performance, and generalization.

## Current Research Status

The initial Tomato baseline experiments have been conducted for ResNet50, EfficientNet-B0, and DenseNet121.

The next stages of the research include:

1. Reviewing recent literature on hybrid CNN and Transformer-based architectures.
2. Improving the consistency and documentation of baseline experiments.
3. Evaluating additional baseline models where appropriate.
4. Investigating performance across more crop classes and datasets.
5. Designing and evaluating the proposed hybrid architecture.
6. Analysing model predictions using explainability techniques and investigating cross-dataset generalization.

The hybrid architecture and broader generalization experiments remain part of the ongoing research.

## Technologies

The experiments use:

- Python
- PyTorch
- torchvision
- pandas
- scikit-learn
- Google Colab

Training experiments use GPU acceleration when available.

## Reproducibility and Data Availability

The notebooks document the current experimental workflow. The dataset images and trained model checkpoint files are not included in this repository.

The notebooks currently use Google Colab and Google Drive paths for accessing data and saving experiment outputs. To reproduce the experiments, users must configure the dataset paths, install compatible dependencies, and ensure that the required data and pretrained weights are accessible.

Dataset access and use should comply with the dataset's applicable licence and terms.

## Research Purpose

This repository supports the development and evaluation of a hybrid CNN-based approach for robust multiclass plant disease classification. The long-term objective is to investigate whether the proposed approach can improve classification performance and generalization across diverse plant disease datasets.
