# Hybrid CNN Architecture for Robust Multiclass Plant Disease Classification Across Diverse Datasets

MSc thesis, Computer Science and Engineering, University of South-Eastern Norway (USN)
Sayada Zakia Afrinee · Supervisors: Vimala Nunavath, Ayan Chatterjee · Status: November 2026

## Datasets
| Dataset | Crops (classes) | Clean images |
|---|---|---:|
| Agri-Vision Bangladesh (Billah et al., 2026) | Tomato (5), Papaya (7), Bottle Gourd (7) | 3,749 |
| RiceLeafBD (Rimi et al., 2025) | Rice (4) | 1,429 |

Both datasets contained duplicate images and images with conflicting labels, which were removed (Agri-Vision: 1,560 duplicate files; RiceLeafBD: 131 images). Zucchini is excluded (too few images per class). Datasets are not included in this repository.

## Method
- Stratified 5-fold split grouped by near-duplicate images (no leakage); results below are fold 0.
- ImageNet-pretrained CNNs fine-tuned in two phases; settings and models are chosen on validation macro-F1, and test results are reported only.
- Hybrid CNN: two pretrained CNNs (DenseNet121 + ConvNeXtTiny) see the same image; their features are combined by a learned gate, with an auxiliary classifier on each branch.

## Results (fold 0, test accuracy)
| Crop | Best single CNN | Hybrid (DenseNet121 + ConvNeXtTiny) |
|---|---:|---:|
| Tomato | 77.7% (VGG16) | 76.4% |
| Papaya | 73.7% (VGG16) | 72.4% |
| Bottle Gourd | 80.8% (ConvNeXtTiny) | 82.8% |
| Rice | 97.5% (ConvNeXtTiny) | 95.4% |

## Key findings
- **Agri-Vision:** no published model results were found; these appear to be the first reported baselines.
- **RiceLeafBD:** up to 97.5% test accuracy, compared with 91.5% (EfficientNet-V2) in the published paper, which used a single random split without a duplicate check.
- **Data is the main limit:** the same code reaches 95–97% on rice but 72–83% on Agri-Vision; 13% of Tomato test images are misclassified by almost all models (mostly Mosaic predicted as Spot, often with mixed or faint symptoms).
- **Hybrid:** selected after testing three fusion designs and screening all pairs and triples of five CNNs. It is close to the best single CNN on every crop and gives its largest gains on difficult classes (e.g. Bottle Gourd Fungal Damage Leaf: 50% → 62% F1).
- **Not helpful:** label smoothing, MixUp, CutMix, leaf cropping, a 3-model ensemble, and freezing early layers. Single-fold differences of up to ±3 points can be random; a final 5-fold cross-validation will confirm the comparison.

## Notebooks
| Notebook | Content |
|---|---|
| `03`–`08_tomato_run*` | Tomato experiments, one change per run |
| `09_tomato_error_analysis` | Analysis of the hardest images |
| `10_baseline_all_crops` | RiceLeafBD audit and baselines for all crops |
| `11_run05_setting_all_crops` | Best Tomato setting tested on all crops |
| `12_leaf_crop_and_ensemble` | Leaf cropping and model ensemble |
| `13_hybrid_H1_H2_H3` | Hybrid fusion designs |
| `14_hybrid_screening` | Screening of CNN pairs and triples |
| `15_hybrid_best_pair_triple` | Fine-tuning of the best pair and triple |
| `16_hybrid_H5_partial_finetuning` | Partial fine-tuning of the selected hybrid |

Earlier notebooks (`01`, `02`, `Tomato_run01`, `Tomato_run02`) are exploratory and used other splits or settings.

## How to run
Put `agri_vision.zip` and `riceleafbd.zip` in `MyDrive/thesis/data/raw/`, open a notebook in Google Colab with a T4 GPU, and choose **Runtime → Run all**.
