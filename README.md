# Hybrid CNN Architecture for Robust Plant Disease Classification

## Project Overview

This repository contains the notebooks and preliminary experiments for my master's thesis, **“Hybrid CNN Architecture for Robust Multiclass Plant Disease Classification Across Diverse Datasets.”**

The project studies pretrained deep-learning models for plant disease image classification, establishes reproducible baselines, and will use the results to inform the design and evaluation of a proposed hybrid CNN. The longer-term research goal is to investigate classification performance and generalization across different crop types and datasets.

## Research Objectives

- Inspect and prepare plant disease image datasets for experiments.
- Establish baseline performance using ImageNet-pretrained models.
- Compare models using accuracy, macro-F1, and complementary evaluation metrics.
- Examine class-level errors and potential overfitting.
- Investigate a hybrid CNN architecture for multiclass plant disease classification.
- Explore explainability methods and generalization across crop types and datasets.

## Dataset and Data Handling

The current comparison notebook uses the AgriVision4 dataset and evaluates three crop subsets:

- Tomato
- Papaya
- Bottle Gourd

The notebook reads a prepared manifest at:

`data/splits/agri_vision_manifest.csv`

It retains records where `keep` is true and checks whether the corresponding image files exist after the dataset archive is extracted. The latest recorded notebook output reported **3,749 retained manifest images and 0 missing files**.

The earlier dataset-analysis workflow included image inventory, SHA-256 duplicate detection, investigation of identical images assigned to different labels, and split preparation. Records associated with conflicting labels were provisionally excluded from the initial Tomato experiment.

**Important data-quality note:** The exclusion policy for all cross-class duplicate/conflicting-label groups must be verified against the manifest before final training and reporting. Do not assume the current manifest has correctly excluded every affected record without checking it. Apply a consistent, documented policy to all relevant experiments.

The dataset images and manifest are not included in this repository. Obtain the dataset separately and configure the Google Drive paths expected by the notebooks. Follow the dataset's licence and terms of use.

## Repository Contents

### `01_agrivision_dataset_analysis.ipynb`

Dataset inspection and preparation, including:

- Image and class-count summaries.
- Duplicate-image analysis.
- Investigation of cross-class label conflicts.
- Crop-level dataset analysis.
- Preparation and inspection of manifest and split files.

### `02_ResNet50_Baseline.ipynb`

The original baseline notebook retained for reference, covering the initial Tomato baseline workflow and related data preparation.

### `02_ResNet50_Baseline_Clean.ipynb`

The cleaned initial Tomato baseline workflow. It includes train/validation/test data preparation, preprocessing, training and fine-tuning of ResNet50, EfficientNet-B0 and DenseNet121, validation macro-F1 checkpoint selection, and test-set evaluation.

### `Copy_of_02_baseline_comparison.ipynb`

The expanded baseline-comparison workflow. It compares eight ImageNet-pretrained architectures across Tomato, Papaya and Bottle Gourd:

- VGG16
- ResNet50
- DenseNet121
- InceptionV3
- Xception
- MobileNetV3Large
- EfficientNetV2B0
- ConvNeXtTiny

The notebook is designed to run in Google Colab with a GPU (a Tesla T4 or better is recommended). It reads the prepared manifest and dataset archive from Google Drive and saves experiment outputs to:

`/content/drive/MyDrive/thesis/experiments/baselines/`

Outputs include:

- `results.csv` — per-run metrics.
- `history/` — training histories.
- `preds/` — test predictions and class probabilities.
- `config.json` — recorded environment and experiment settings.
- `baseline_results.xlsx` — summary tables and run-level results.

The notebook saves each completed run and skips crop/model/fold combinations already present in the results file, allowing interrupted experiments to be resumed.

## Baseline Comparison Protocol

The expanded comparison notebook currently uses the following protocol:

- Image size: 224 × 224 pixels.
- Aspect ratio: preserved by padding images to the target size.
- Initial weights: ImageNet-pretrained weights.
- Random seed: 42.
- Batch size: 32.
- Optimizer: AdamW with weight decay of `1e-4`.
- Training phase 1: freeze the backbone and train the classification head for up to 10 epochs.
- Training phase 2: fine-tune the model for up to 30 epochs with a smaller learning rate.
- Early stopping: based on validation loss, restoring the best weights.
- Class imbalance: balanced class weights computed from the training subset.
- Augmentation: horizontal/vertical flips, small rotations, zoom and contrast changes, applied to training images only.
- Evaluation: accuracy, balanced accuracy, macro precision, macro recall, macro-F1, weighted F1, Cohen's kappa, Matthews correlation coefficient (MCC), parameter count, training time and inference time per image.

For five-fold evaluation, fold *k* is used as the test fold, fold *(k + 1) mod 5* as the validation fold, and the remaining three folds as training data. The current notebook configuration uses `FOLDS = [0]` as a pilot. Change it to `FOLDS = [0, 1, 2, 3, 4]` only after the pilot and data-quality policy have been reviewed.

## Preliminary Results

### Initial Tomato baseline experiment

The earlier cleaned Tomato baseline notebook reported:

| Model | Selected epoch | Validation accuracy | Validation macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|---:|
| ResNet50 | 4 | 58.82% | 0.3665 | 62.88% | 0.3928 |
| EfficientNet-B0 | 3 | 67.65% | 0.5387 | 71.21% | 0.5468 |
| DenseNet121 | 5 | 67.65% | 0.5737 | 74.24% | 0.6159 |

These are preliminary results from a single experimental split.

### Expanded comparison pilot

The latest saved output from `Copy_of_02_baseline_comparison.ipynb` contains one completed fold for each of eight models across the three crop subsets. The best model in each crop by test macro-F1 in this pilot was:

| Crop | Best model in pilot | Test macro-F1 | Test accuracy |
|---|---|---:|---:|
| Tomato | DenseNet121 | 71.32% | 76.35% |
| Papaya | ConvNeXtTiny | 67.19% | 70.03% |
| Bottle Gourd | ConvNeXtTiny | 73.85% | 77.15% |

These pilot results are not directly interchangeable with the earlier Tomato baseline results because the notebooks use different experimental workflows and splits. They should not be treated as final estimates or as evidence that one architecture is universally best. The expanded comparison has **one fold per model and crop**, so five-fold evaluation remains to be completed.

## Current Research Status and Next Steps

1. Verify that the manifest consistently excludes all unresolved conflicting-label records, and document the decision.
2. Review the pilot results, training histories and confusion matrices for obvious issues.
3. Complete the five-fold baseline comparison after the data policy is verified.
4. Compare models using fold-level results and mean/standard deviation, with particular attention to macro-F1 and class-wise errors.
5. Use the baseline findings and literature review to define the proposed hybrid CNN and a fair evaluation protocol.
6. Evaluate the hybrid model against selected baselines.
7. Investigate explainability and cross-dataset generalization as the project develops.

The hybrid CNN has **not yet been evaluated** in the current notebooks. The work described here represents dataset preparation and baseline experimentation.

## Technologies

- Python
- TensorFlow / Keras
- NumPy and pandas
- scikit-learn
- Pillow
- Matplotlib
- Google Colab
- Google Drive

GPU acceleration is used when available.

## Reproducing the Experiments

1. Open the relevant notebook in Google Colab.
2. Obtain the AgriVision4 dataset and place the archive at the path expected by the notebook.
3. Prepare the manifest and fold assignments using the dataset-analysis workflow.
4. Verify the conflicting-label exclusion policy before training.
5. Update the Google Drive paths if your folder structure differs.
6. Run the pilot first and inspect its outputs before starting all five folds.
7. Keep the manifest, split assignments, software versions, configuration and experiment outputs together so that results can be traced and reproduced.

Dataset files, trained weights and generated experiment outputs are not committed to this repository. Reproduction may require downloading ImageNet weights and installing compatible dependencies in the Colab environment.

## Research Purpose

This repository supports the systematic development and evaluation of a hybrid CNN approach for robust multiclass plant disease classification. Its current results are preliminary; conclusions about model performance and generalization should be made only after data-quality checks and the planned evaluation are complete.
