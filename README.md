# Hybrid CNN Architecture for Plant Disease Classification

## Project Overview

This repository contains the code and experimental work for my master's thesis, **“Hybrid CNN Architecture for Robust Multiclass Plant Disease Classification Across Diverse Datasets.”**

The research aims to investigate deep learning approaches for plant disease classification, compare pretrained baseline models, and develop a hybrid CNN architecture with a focus on robust classification and generalization across diverse datasets.

## Dataset

The initial experiments use the Tomato subset of the AgriVision4 plant disease dataset.

The Tomato subset contains five classes:

- Downy Mildew
- Healthy
- Mosaic
- Spot
- White Spot

The dataset was checked for duplicate images and conflicting labels before preparing the experimental split. Images with conflicting labels were provisionally excluded from the initial experiment.

## Baseline Experiments

The following pretrained models have been evaluated in the initial Tomato experiment:

- ResNet50
- EfficientNet-B0
- DenseNet121

The models were evaluated using a common train, validation, and test split. Validation macro-F1 was used to select the best checkpoint for each model.

### Preliminary Results

| Model | Test Accuracy | Test Macro-F1 |
|---|---:|---:|
| ResNet50 | 62.88% | 0.393 |
| EfficientNet-B0 | 71.21% | 0.547 |
| DenseNet121 | 74.24% | 0.616 |

These results are preliminary and come from a single experimental split. They should not be interpreted as evidence that one model is universally superior.

## Repository Contents

- `01_agrivision_dataset_analysis.ipynb` — dataset inspection and analysis.
- `02_ResNet50_Baseline.ipynb` — ResNet50 baseline experiment.

Additional model experiments, comparative analysis, and hybrid architecture development are planned as the research progresses.

## Current Status

The initial Tomato baseline experiments have been conducted for ResNet50, EfficientNet-B0, and DenseNet121. The next stages include further literature analysis, improving experimental consistency, evaluating additional baselines, and investigating generalization across a broader range of plant disease classes and datasets.

## Reproducibility and Data

The notebooks are intended to document the experimental workflow. Dataset files and trained model checkpoints are not included in this repository. Access to the dataset should follow its applicable licence and terms of use.

The experiments were developed using Python, PyTorch, torchvision, and Google Colab.

## Research Purpose

This repository supports academic research into robust plant disease classification and the development of a hybrid CNN-based approach for multiclass classification across diverse datasets.is
