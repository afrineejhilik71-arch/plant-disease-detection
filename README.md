# Hybrid CNN Architecture for Robust Multiclass Plant Disease Classification Across Diverse Datasets

MSc thesis, Computer Science and Engineering, University of South-Eastern Norway (USN)
Student: Sayada Zakia Afrinee · Supervisors: Vimala Nunavath (main), Ayan Chatterjee (co-supervisor)
**Status (30 October 2026):** baseline phase on Agri-Vision Bangladesh, Tomato. The hybrid model has not been built yet.

---

## Summary of progress

1. **Data audit and cleaning.** The Agri-Vision Bangladesh original images contain many duplicates, label conflicts and leakage between the authors' Train/Validation/Test folders. These were removed using documented rules, and Zucchini was excluded.
2. **Stable training protocol.** Early runs showed under-fitting and unstable fine-tuning. The protocol was corrected using the training curves, not the test scores.
3. **Run 01 (baseline):** Tomato, fold 0, 8 ImageNet-pretrained CNNs, 224 × 224 px.
4. **Run 02:** the same setup with one change, 384 × 384 px input.
5. **Next:** further single-change runs (label smoothing, stronger augmentation, in-domain pre-training), then 5-fold cross-validation for all crops, then the hybrid model.

---

## Data

**Dataset:** Agri-Vision Bangladesh (Billah et al., *Data in Brief*, 2026; DOI [10.1016/j.dib.2026.112528](https://doi.org/10.1016/j.dib.2026.112528); data on [Mendeley](https://data.mendeley.com/datasets/8t6k37ztxc/1)). Only the expert-validated original images are used, not the provided augmented copies. Images are **not** included in this repository.

**Audit findings (original images):**

| Check | Result |
|---|---|
| Images found / reported in the paper | 5,246 / 5,266 |
| Exact-duplicate files | 1,560 |
| Groups of identical images with different disease labels | 275 |
| Groups of identical images in more than one official split (leakage) | 326 |
| Image sizes | 512×384, 384×512, 512×287 (paper states 512×512) |

**Cleaning rules** (the original files are never modified):
1. remove identical images that carry different labels;
2. keep one copy of each duplicate;
3. remove near-identical images (same perceptual hash) with different labels;
4. exclude Zucchini, because 8 of its 9 classes keep fewer than 50 usable images.

| Crop | Classes | Clean images |
|---|---|---|
| Bottle Gourd | 7 | 1,491 |
| Papaya | 7 | 1,463 |
| **Tomato** | **5** | **795** |
| **Total** | **19** | **3,749** |

**Split:** stratified 5-fold cross-validation, grouped by perceptual hash, so near-duplicates never cross folds. In round *k*, fold *k* is the test set, fold *(k+1) mod 5* is the validation set, and the other three folds are training data.
Tomato fold 0: **493 training / 154 validation / 148 test images**.

---

## Repository contents

| Notebook | Framework | Purpose | Status |
|---|---|---|---|
| `Tomato_run01_baseline.ipynb` | TensorFlow / Keras | **Run 01**: self-contained pipeline (audit and cleaning → folds → 8 CNNs → results), Tomato fold 0, 224 px | **Current** |
| `Tomato_run02_size384.ipynb` | TensorFlow / Keras | **Run 02**: identical to Run 01 except 384 px input | **Current** |
| `Copy_of_02_baseline_comparison.ipynb` | TensorFlow / Keras | First 8-model pilot on all three crops (fold 0), with training-curve diagnostics | Superseded (earlier protocol) |
| `01_agrivision_dataset_analysis.ipynb` | PyTorch | Initial dataset exploration and duplicate analysis | Archived (exploratory) |
| `02_ResNet50_Baseline.ipynb`, `02_ResNet50_Baseline_Clean.ipynb` | PyTorch | Initial Tomato baselines (ResNet50, EfficientNet-B0, DenseNet121) on an earlier split | Archived (exploratory) |

Results from the archived and superseded notebooks used other splits or protocols, so they **are not directly comparable** with Runs 01–02.

---

## Current training protocol (Runs 01 and 02)

- ImageNet-pretrained backbone → global average pooling → dropout 0.3 → softmax layer.
- Images are padded to a square (aspect ratio kept), at 224 × 224 px (Run 01) or 384 × 384 px (Run 02).
- **Phase 1:** frozen backbone; the new head is trained for up to 15 epochs (lr 1e-3; early stopping on validation loss, patience 3).
- **Phase 2:** full fine-tuning with AdamW (weight decay 1e-4, gradient clipping 1.0). The learning rate warms up from 0 to 3e-5 over 2 epochs, then follows a cosine decay (max 40 epochs). Early stopping on **validation accuracy** (patience 10), and the best weights are restored.
- Balanced class weights; augmentation on training images only (flips, rotation, zoom, contrast); seed 42; batch size 32.
- Mixed precision, except MobileNetV3Large and Xception, which run in float32 for stability.
- Each model is trained in a separate process, which keeps RAM stable on free Colab.
- Metrics: accuracy, balanced accuracy, **macro-F1 (primary)**, weighted F1, MCC, per-class F1, confusion matrix, parameters, training time.

**Why this protocol:** in an earlier 5-fold Tomato run, 4–7 of 40 runs broke at the start of fine-tuning. The training curves showed two causes: the loss exploded with a learning rate of 1e-4, and early stopping on validation loss stopped while validation accuracy was still improving. The warm-up/cosine schedule and accuracy-based early stopping fixed both problems.

---

## Results: Agri-Vision Tomato, fold 0 (148 test images)

### Run 01 (224 px) and Run 02 (384 px)

| Model | Params (M) | Run 01 accuracy | Run 01 macro-F1 | Run 02 accuracy | Run 02 macro-F1 |
|---|---:|---:|---:|---:|---:|
| VGG16 | 14.7 | **77.70** | **73.42** | **77.70** | **72.55** |
| ResNet50 | 23.6 | 66.89 | 62.80 | 77.03 | 71.73 |
| InceptionV3 | 21.8 | 71.62 | 63.63 | 77.03 | 71.18 |
| DenseNet121 | 7.0 | 72.30 | 68.87 | 77.03 | 70.31 |
| ConvNeXtTiny | 27.8 | 72.97 | 69.63 | 77.03 | 69.27 |
| EfficientNetV2B0 | 5.9 | 71.62 | 69.10 | 68.24 | 64.78 |
| Xception | 20.9 | 67.57 | 61.59 | 71.62 | 60.08 |
| MobileNetV3Large ⚠ | 3.0 | 66.89 | 58.46 | 67.57 | 56.33 |
| **Mean of 8 models** | | **70.95** | **65.94** | **74.16** | **67.03** |

⚠ In Run 01, MobileNetV3Large did not fit its training data (training accuracy 29%), so its Run 01 result is not a valid comparison.

### Mean F1 per disease (average over the 8 models)

| Disease | Run 01 (224 px) | Run 02 (384 px) | Change |
|---|---:|---:|---:|
| Healthy | 85.6 | 88.7 | +3.1 |
| Spot | 71.6 | 78.4 | +6.8 |
| White Spot | 49.9 | 52.5 | +2.6 |
| Mosaic | 55.0 | 54.7 | −0.4 |
| Downy Mildew* | 67.5 | 60.9 | −6.6 |

\*Only about 8 Downy Mildew test images, so one image changes its F1 by about 12 points.

### Findings so far

- **Best model:** VGG16 (77.7% accuracy, 73.4% macro-F1 in Run 01). In Run 02, the top five models are within 3.3 macro-F1 points of each other, so a single fold cannot separate them reliably.
- **Higher resolution (384 px):** mean accuracy +3.2 points (6 of 8 models improved), mean macro-F1 +1.1 points, at 2–3× the training time.
- **Local versus global patterns:** larger images helped spot-type diseases (Spot, White Spot) but **not Mosaic**, a whole-leaf pattern. This supports a hybrid design with a global-context branch.
- **Overfitting:** most models reach 85–95% on training images but 70–77% on test images.
- **Caution:** one photo is worth about 0.7 accuracy points on this test fold. All conclusions will be confirmed with 5-fold cross-validation.

---

## Next steps

| Run | Single change (Tomato fold 0, 224 px, top-5 models) | Target problem |
|---|---|---|
| 03 | Label smoothing (0.1) | Overconfidence, noisy labels |
| 04 | Stronger augmentation (MixUp/CutMix, colour jitter) | Overfitting |
| 05 | In-domain pre-training (PlantVillage → Agri-Vision) | Limited data |
| 06 | Combine the helpful changes; tune the 2 best models with Optuna; ensemble and test-time augmentation | Final gains |

Then:
1. **5-fold cross-validation** of the final setup for Tomato, Papaya and Bottle Gourd (mean ± SD).
2. **Validation study:** the same pipeline trained and tested within **PlantDoc** (Singh et al., 2020; field images, 27 classes) and **PlantVillage** (Hughes & Salathé, 2015; laboratory images, 38 classes), with results compared against published results to verify that the pipeline reproduces them.
3. **Hybrid model:** a local CNN branch and a global branch with learned fusion, designed from the literature review and the error analysis, and evaluated with ablations and XAI (Grad-CAM, Grad-CAM++, saliency maps).
4. **Generalisation:** models trained on Agri-Vision tested on **PlantDoc** and **PlantVillage** as unseen datasets, using matching classes only (e.g. Tomato Healthy and Mosaic). To keep this test fair, the validation study is used only to check the basic pipeline, not to tune the hybrid model.

---

## How to run

1. Put the dataset archive at `MyDrive/thesis/data/raw/agri_vision.zip` in Google Drive.
2. Open `Tomato_run01_baseline.ipynb` (or Run 02) in Google Colab and select a **T4 GPU** runtime.
3. Choose **Runtime → Run all**. The notebook unzips, audits, cleans, splits, trains and saves results to `MyDrive/thesis/experiments/tomato/<RUN_NAME>/` (`results.csv`, Excel summary, figure, training histories, predictions).
4. Finished models are skipped on re-run, so after a disconnect, run all cells again.

Each new experiment changes `RUN_NAME`, so every run keeps its own results folder.

**Software:** Python 3.13, TensorFlow 2.21, Keras 3.13, scikit-learn, pandas, Pillow, imagehash (Google Colab). Datasets, model weights and generated outputs are not committed (see `.gitignore`).
