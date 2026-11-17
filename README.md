# Hybrid CNN Architecture for Robust Multiclass Plant Disease Classification Across Diverse Datasets

MSc thesis, Computer Science and Engineering, University of South-Eastern Norway (USN)
Sayada Zakia Afrinee · Supervisors: Vimala Nunavath (main), Ayan Chatterjee (co-supervisor)
**Status (November 2026):** baselines for four crops completed; hybrid CNN designed, selected and compared with single CNNs. Final 5-fold cross-validation is pending.

All experiments run in Google Colab (T4 GPU) with TensorFlow/Keras. Datasets, model weights and large outputs are not stored in this repository.

---

## 1. Overview

The thesis develops a hybrid CNN for multiclass plant disease classification and tests it on several crops and datasets. The work so far has five parts:

1. **Data audit and cleaning** of two public datasets, both of which contained duplicate images and conflicting labels.
2. **Leakage-free evaluation** with a grouped, stratified 5-fold split.
3. **Controlled experiments** on Tomato (one change per run), then checked on all crops.
4. **Baselines for four crops**: Tomato, Papaya and Bottle Gourd (Agri-Vision) and Rice (RiceLeafBD).
5. **Hybrid CNN**: fusion design, systematic search for the best CNN combination, fine-tuning and analysis.

---

## 2. Datasets

| Dataset | Crops (classes) | Images found | Clean images | Source |
|---|---|---:|---:|---|
| Agri-Vision Bangladesh | **Tomato (5):** Downy Mildew, Healthy, Mosaic, Spot, White Spot<br>**Papaya (7):** Bacterial Blight, Carica Insect Hole, Curled Yellow Spot, Healthy, Mosaic Virus, Pathogen Symptoms, Yellow Necrotic Spots Holes<br>**Bottle Gourd (7):** Alternaria Leaf Blight, Anthracnose, Downy Mildew, Early Alternaria Leaf Blight, Fungal Damage Leaf, Healthy, Mosaic Virus | 5,246 | 3,749 | Billah et al., *Data in Brief*, 2026 ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12907883/), [data](https://data.mendeley.com/datasets/8t6k37ztxc/1)) |
| RiceLeafBD | Rice (4): Bacterial Leaf Blight, Brown Spot, Healthy, Tungro | 1,560 | 1,429 | Rimi et al., 2025 ([paper](https://arxiv.org/abs/2501.08912), [data](https://data.mendeley.com/datasets/kx9rx8p2mz/1)) |

Only the original images are used, not the augmented copies provided with the datasets.

**Audit findings**
- **Agri-Vision:** 1,560 exact duplicate files; 275 groups of identical images with different labels; 326 duplicate groups across the authors' Train/Validation/Test folders (leakage). Zucchini is excluded because 8 of its 9 classes keep fewer than 50 usable images.
- **RiceLeafBD:** 44 images with conflicting labels and 87 duplicate copies.

**Cleaning rules** (original files are never modified): remove identical images with different labels → keep one copy of each duplicate → remove near-identical images (same perceptual hash) with different labels.

**Leakage check (Tomato, official split):** test images with an identical copy in training reached 100% accuracy, truly new images 77.7%. Results on the official split are therefore inflated.

| Crop | Classes | Clean images | Train / validation / test (fold 0) |
|---|---:|---:|---|
| Tomato | 5 | 795 | 493 / 154 / 148 |
| Papaya | 7 | 1,463 | 865 / 301 / 297 |
| Bottle Gourd | 7 | 1,491 | 895 / 294 / 302 |
| Rice | 4 | 1,429 | 857 / 287 / 285 |

---

## 3. Method

- **Split:** stratified 5-fold cross-validation, grouped by perceptual hash so that near-identical images never appear in two folds. Round *k*: test = fold *k*, validation = fold *(k+1) mod 5*, training = the remaining three folds. All results below are round 0; folds 1–4 are kept for the final evaluation.
- **Model selection:** settings and models are compared on **validation macro-F1** (primary metric); test results are reported only. A change counts as an improvement only if it helps on most crops.
- **Models:** ImageNet-pretrained CNNs from `keras.applications` (VGG16, ResNet50, DenseNet121, InceptionV3, EfficientNetV2B0, ConvNeXtTiny), each with its own preprocessing; head = global average pooling → dropout 0.3 → softmax.
- **Training:** phase 1 trains the head only (up to 15 epochs, lr 1e-3); phase 2 fine-tunes all layers with AdamW (weight decay 1e-4, gradient clipping 1.0), warm-up to lr 3e-5 and cosine decay, up to 40 epochs, early stopping on validation accuracy (patience 10, best weights restored).
- **Standard setting (Run 05):** 224 × 224 px (padded to a square), training classes balanced by oversampling to at least 300 images, moderate augmentation on training images only (flip, rotation, zoom, shift, contrast, brightness).
- **Metrics:** macro-F1, accuracy, balanced accuracy, macro precision and recall, MCC, per-class F1, confusion matrix, parameters, training time.

---

## 4. Results (fold 0)

### 4.1 Tomato experiments: one change per run (validation macro-F1, mean of 6 CNNs)

| Run | Change | Validation macro-F1 | Decision |
|---|---|---:|---|
| 01b | Baseline (class weights, basic augmentation) | 69.0 | reference |
| 03 | Label smoothing 0.1 | 66.7\* | rejected |
| 04 | Balanced training classes (300 per class) | 70.1 | kept |
| 05 | + moderate augmentation | **71.8** | best on Tomato |
| 06 | + MixUp | 69.5 | rejected |
| 07 | + CutMix | 69.0 | rejected |

\*5 CNNs (ResNet50 was not trained in Run 03).

**Check on other crops:** the Run 05 setting changed validation macro-F1 by +3.1 (Tomato), −0.1 (Papaya), −1.6 (Bottle Gourd) and +0.7 (Rice). The Tomato gain did not carry over, so from then on every change was tested on all crops.

**Leaf cropping and ensemble (Run 08):** cropping images to the leaf changed validation macro-F1 by −1.3 (Tomato), −0.7 (Papaya) and +3.5 (Bottle Gourd); a 3-model ensemble gave no reliable gain. Because the Papaya images were almost unchanged by cropping, this run also showed that single-fold differences of up to about ±3 points can be random.

### 4.2 Baselines for all crops (6 CNNs, same settings, mean over CNNs)

| Crop | Validation macro-F1 | Test macro-F1 | Test accuracy |
|---|---:|---:|---:|
| Tomato | 69.0 | 70.4 | 74.9 |
| Papaya | 66.6 | 66.8 | 69.0 |
| Bottle Gourd | 68.2 | 69.2 | 75.1 |
| Rice | 94.4 | 94.1 | 93.8 |

### 4.3 Hybrid CNN

**Design:** the same image goes into two pretrained CNNs. Each branch's features are projected to 256 values; a learned gate decides, per image, how much to trust each branch; a classifier gives the final answer. During training, each branch also has its own auxiliary classifier, so both branches must learn useful features.

**How the hybrid was chosen**
1. **Fusion design (H1–H3)** with VGG16 + ConvNeXtTiny: simple concatenation (H1), gated fusion (H2), and gated fusion with auxiliary classifiers (H3).
2. **Screening:** frozen features of five CNNs (VGG16, ResNet50, DenseNet121, InceptionV3, ConvNeXtTiny), with a logistic-regression classifier on every pair and every group of three. Best pair: DenseNet121 + ConvNeXtTiny; best triple: VGG16 + DenseNet121 + ConvNeXtTiny.
3. **Fine-tuning** of the best pair and triple with the H3 design. Both gave about the same results (79.2 vs 79.0 average validation macro-F1), but the pair is smaller (36 vs 51 million parameters) and faster, so the pair was selected.

**Validation macro-F1 (%)**

| Crop | VGG16 | DenseNet121 | ConvNeXtTiny | **Hybrid: DenseNet121 + ConvNeXtTiny** | Hybrid: VGG16 + DenseNet121 + ConvNeXtTiny |
|---|---:|---:|---:|---:|---:|
| Tomato | 73.8 | 73.6 | 72.8 | **76.8** | 73.7 |
| Papaya | 74.2 | 67.9 | 67.6 | 72.2 | 72.5 |
| Bottle Gourd | 70.4 | 66.9 | 70.6 | 72.0 | 73.2 |
| Rice | 95.7 | 94.8 | 96.4 | 95.8 | 96.7 |

**Test accuracy (%)**

| Crop | Best single CNN | **Hybrid: DenseNet121 + ConvNeXtTiny** |
|---|---:|---:|
| Tomato | 77.7 (VGG16) | 76.4 |
| Papaya | 73.7 (VGG16) | 72.4 |
| Bottle Gourd | 80.8 (ConvNeXtTiny) | **82.8** |
| Rice | 97.5 (ConvNeXtTiny) | 95.4 |

**Inside the hybrid**
- The fused result is better than each branch on its own (+5 to +9 macro-F1 points on Agri-Vision), so the two CNNs are complementary.
- The gate gives similar weights to all diseases (about 60% ConvNeXtTiny, 40% DenseNet121): the branches do not yet specialise in local versus global patterns.
- The largest gains are on difficult classes, e.g. Bottle Gourd Fungal Damage Leaf (test F1: 50 for the best single CNN, 62 for the hybrid) and Papaya Bacterial Blight (68 vs 72).

**Partial fine-tuning (H5):** fine-tuning only the last stage, or the last two stages, of each CNN reduced the gap between training and validation accuracy, but lowered validation macro-F1 on most crops (e.g. Tomato 76.8 → 70.2 and 66.3). The early layers also need to adapt to leaf images, so full fine-tuning is kept.

### 4.4 Comparison with the published RiceLeafBD result

| | Split | Duplicate check | Best accuracy |
|---|---|---|---:|
| Rimi et al., 2025 (4 models; best EfficientNet-V2) | single random 80/20 | no | 91.5% |
| This work (single CNNs and hybrids) | grouped 5-fold split, fold 0 | yes (131 images removed) | **97.5%** (ConvNeXtTiny); hybrids 95.4–96.5% |

### 4.5 Error analysis (Tomato)

19 of 148 test images (13%) are misclassified by at least 5 of 6 CNNs. Mosaic is the hardest class: 32% of its test images are in this group, mostly predicted as Spot. Visual inspection shows mixed symptoms, very faint symptoms, poor-quality photos and possible leaf-miner damage, so part of the remaining error comes from the data rather than the model.

---

## 5. Key findings

1. **First results on Agri-Vision:** the dataset paper reports no model experiments, and no other published results were found; these appear to be the first reported baselines.
2. **RiceLeafBD:** up to 97.5% test accuracy under a stricter, leakage-free protocol, compared with 91.5% published.
3. **Data quality is the main limit:** the same pipeline reaches 95–97% on rice but 72–83% on Agri-Vision.
4. **Hybrid CNN:** close to the best single CNN on every crop, with the largest gains on difficult classes; differences are within single-fold noise.
5. **Methods that did not help:** label smoothing, MixUp, CutMix, leaf cropping, a 3-model ensemble and freezing early layers.

---

## 6. Next steps

1. Screen CNNs and vision transformers pretrained on larger datasets (e.g. DINOv2, CLIP, ImageNet-22k models) as hybrid branches.
2. Grad-CAM: which leaf regions each model and branch uses.
3. Final 5-fold cross-validation of the selected hybrid against the best single CNNs, for all crops.
4. Validation and generalisation study with PlantVillage and PlantDoc.

---

## 7. Notebooks

| Notebook | Content |
|---|---|
| `03_tomato_run01b_baseline` | Tomato baseline, six CNNs |
| `04_tomato_run03_label_smoothing` | Label smoothing 0.1 |
| `05_tomato_run04_balanced_classes` | Balanced training classes |
| `06_tomato_run05_moderate_augmentation` | Moderate augmentation |
| `07_tomato_run06_mixup` | MixUp |
| `08_tomato_run07_cutmix` | CutMix |
| `09_tomato_error_analysis` | Confusion matrix and hardest images |
| `10_baseline_all_crops` | RiceLeafBD audit; baselines for Rice, Papaya, Bottle Gourd |
| `11_run05_setting_all_crops` | Best Tomato setting on all crops |
| `12_leaf_crop_and_ensemble` | Leaf cropping and 3-model ensemble |
| `13_hybrid_H1_H2_H3` | Hybrid fusion designs |
| `14_hybrid_screening` | Screening of CNN pairs and triples |
| `15_hybrid_best_pair_triple` | Fine-tuning of the best pair and triple |
| `16_hybrid_H5_partial_finetuning` | Partial fine-tuning of the selected hybrid |

Earlier notebooks (`01`, `02`, `Copy_of_02`, `Tomato_run01`, `Tomato_run02`) are exploratory and used other splits or settings, so their results are not directly comparable.

---

## 8. How to run

1. Put `agri_vision.zip` and `riceleafbd.zip` in Google Drive: `MyDrive/thesis/data/raw/`.
2. Open a notebook in Google Colab and select a **T4 GPU** runtime.
3. Choose **Runtime → Run all**. Each notebook unzips, audits, cleans and splits the data itself, then trains and saves results to `MyDrive/thesis/experiments/<crop>/<run>/` (results table, Excel summary, training histories, predictions).
4. After a disconnect, run all cells again: finished models are skipped.

**Software:** Python 3, TensorFlow 2 / Keras 3, scikit-learn, pandas, Pillow, imagehash, matplotlib.
