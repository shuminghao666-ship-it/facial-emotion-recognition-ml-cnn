# Hybrid Machine Learning and CNN Fusion for Facial Emotion Recognition

Repository: `facial-emotion-recognition-ml-cnn`

This repository contains a DS4023 Machine Learning course project for
seven-class facial emotion recognition. The project follows a FER2013-style
setting and classifies grayscale facial images into:

`Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, and `Surprise`.

The main goal is not only to train a CNN, but also to compare it with
traditional machine learning baselines and test whether ML-based feature
analysis, probability fusion, confusion-aware correction, and stacking can
improve or explain CNN predictions.

## Project Pipeline

The project is organized as a four-step pipeline.

| Step | Script | Purpose |
|---|---|---|
| Step 00 | `00_dataset_analysis.py` | Check dataset structure, class distribution, corrupted images, and class imbalance. |
| Step 01 | `01_traditional_ml_baselines.py` | Compare raw pixels and HOG features using traditional ML classifiers. |
| Step 02 | `02_cnn_baseline.py` | Train the CNN baseline and compare CNN with CLAHE vs. without CLAHE. |
| Step 03 | `03_ml_enhanced_cnn.py` | Use CNN features, PCA, HOG-CNN fusion, probability fusion, confusion-aware correction, and stacking to improve and analyze CNN predictions. |

## Dataset

The dataset is included in this repository under the `data/` folder.

Original dataset download link: `TODO: add original FER2013-style dataset link`

Dataset summary:

- Training images: `28,384`
- Test images: `7,503`
- Total images: `35,887`
- Image type: low-resolution grayscale facial emotion images
- Classes: `Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`

### Dataset Folder Structure

```text
data/
├── train/
│   ├── Angry/
│   ├── Disgust/
│   ├── Fear/
│   ├── Happy/
│   ├── Neutral/
│   ├── Sad/
│   └── Surprise/
└── test/
    ├── Angry/
    ├── Disgust/
    ├── Fear/
    ├── Happy/
    ├── Neutral/
    ├── Sad/
    └── Surprise/
```

### Class Distribution

| Class | Train | Test | Total | Percentage |
|---|---:|---:|---:|---:|
| Angry | 3,995 | 958 | 4,953 | 13.80% |
| Disgust | 111 | 436 | 547 | 1.52% |
| Fear | 4,097 | 1,024 | 5,121 | 14.27% |
| Happy | 7,215 | 1,774 | 8,989 | 25.05% |
| Neutral | 4,965 | 1,233 | 6,198 | 17.27% |
| Sad | 4,830 | 1,247 | 6,077 | 16.93% |
| Surprise | 3,171 | 831 | 4,002 | 11.15% |

The dataset is highly imbalanced. `Happy` is the largest class and `Disgust`
is the smallest class. The imbalance ratio is `16.43:1`. Macro F1 is emphasized
because accuracy can be dominated by majority classes, while Macro F1 gives
equal weight to each emotion class.

## Recommended Project Structure

The current repository uses this structure:

```text
facial-emotion-recognition-ml-cnn/
├── data/
│   ├── train/
│   └── test/
├── 00_dataset_analysis.py
├── 01_traditional_ml_baselines.py
├── 02_cnn_baseline.py
├── 03_ml_enhanced_cnn.py
├── project_config.py
├── emotion_common.py
├── models/
├── results/
│   ├── dataset_analysis/
│   ├── step1_result/
│   ├── step2_result/
│   └── step3_result/
├── hybrid_ml_cnn_fer_report.pdf
├── hybrid_ml_cnn_fer_report.tex
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main packages:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `opencv-python`
- `scikit-learn`
- `scikit-image`
- `torch`
- `torchvision`
- `tqdm`
- `Pillow`
- `joblib`

## Running Instructions

Run all scripts from the repository root.

```bash
cd facial-emotion-recognition-ml-cnn
```

### Step 00: Dataset Analysis

```bash
python 00_dataset_analysis.py
```

Main outputs:

```text
results/dataset_analysis/
```

This step counts train/test images, calculates class percentages, checks
unreadable images, and generates class distribution figures.

### Step 01: Traditional ML Baselines

```bash
python 01_traditional_ml_baselines.py
```

Models compared:

- Raw pixels + SGD Linear SVM
- HOG + Logistic Regression
- HOG + KNN
- HOG + SGD Linear SVM

Main outputs:

```text
results/step1_result/
```

### Step 02: CNN Baseline and CLAHE Comparison

```bash
python 02_cnn_baseline.py
```

Experiments:

- CNN with CLAHE
- CNN without CLAHE

Main outputs:

```text
models/
results/step2_result/
```

### Step 03: ML-Enhanced CNN Analysis

```bash
python 03_ml_enhanced_cnn.py
```

Methods evaluated:

- CNN end-to-end baseline
- CNN deep features + traditional ML classifiers
- PCA-reduced CNN features
- HOG-CNN feature fusion
- CNN probability + ML probability fusion
- Step 01-guided confusion-aware correction
- Stacking over CNN and ML probability outputs
- Grad-CAM examples for qualitative analysis

Main outputs:

```text
results/step3_result/
```

## Main Results

### Step 01: Traditional ML Baselines

| Method | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| HOG + KNN | 0.4743 | 0.4296 | 0.4583 |
| HOG + Logistic Regression | 0.4100 | 0.3545 | 0.3941 |
| HOG + SGD Linear SVM | 0.3816 | 0.3447 | 0.3728 |
| Raw Pixel + SGD Linear SVM | 0.2724 | 0.2295 | 0.2627 |

HOG features perform much better than raw pixels. The strongest traditional ML
baseline is `HOG + KNN`.

### Step 02: CNN Baseline

| Model | Validation Macro F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|---|---:|---:|---:|---:|
| CNN with CLAHE | 0.5786 | 0.6186 | 0.5655 | 0.6102 |
| CNN without CLAHE | 0.5818 | 0.6288 | 0.5893 | 0.6231 |

The CNN without CLAHE is selected as the Step 02 baseline because it achieves
the higher validation Macro F1 and better test Macro F1.

### Step 03: ML-Enhanced CNN

| Method | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| CNN End-to-End | 0.6288 | 0.5893 | 0.6231 |
| CNN Feature KNN + CNN probability fusion | 0.6388 | 0.5960 | 0.6330 |
| HOG-CNN KNN + CNN probability fusion | 0.6435 | 0.5992 | 0.6357 |
| Specialist ML Correction | 0.6263 | 0.5865 | 0.6210 |
| Stacking + Balanced Logistic Regression | 0.6388 | 0.6078 | 0.6397 |

Best method:

`Stacked CNN-ML Probabilities + Balanced Logistic Regression`

It improves Macro F1 from `0.5893` for the standalone CNN to `0.6078`, an
absolute gain of `0.0185`.

## Error Analysis Summary

The main confusion patterns are:

- `Angry` vs. `Disgust`: both are negative expressions and can share similar
  brow or nose-region patterns.
- `Fear` vs. `Surprise`: both can include widened eyes and open mouths.
- `Sad` vs. `Neutral`: both may involve subtle or weak facial motion.
- `Fear`, `Sad`, and `Neutral`: low-resolution images make these negative or
  subtle expressions harder to separate.

The confusion-aware correction method is interpretable, but it does not exceed
the best stacking model. Its coverage is limited because it only activates on
selected uncertain cases. Some confused groups overlap, and `Disgust` has very
few training samples. Stacking is more robust because it combines probability
outputs from multiple models globally.

## Report

The final course report is included in the repository:

```text
hybrid_ml_cnn_fer_report.pdf
hybrid_ml_cnn_fer_report.tex
```

The report contains the full experiment design, tables, figures, normalized
confusion matrices, Grad-CAM examples, and discussion.

## Notes About Large Files

The dataset is included under `data/`.

Model checkpoint files such as `.pth`, `.pt`, and `.ckpt` are ignored by
`.gitignore` because they can be large and can be regenerated by running Step
02 and Step 03. Large serialized models and feature arrays such as `.pkl` and
`.npy` are also ignored. Compressed archives and video files are ignored as
well.

If a GitHub upload exceeds size limits, keep the source scripts, report,
selected result CSV/TXT files, and key figures, while excluding model
checkpoints.

## Main Contribution

The project demonstrates that traditional machine learning methods are useful
not only as baselines, but also as complementary components for CNN-based FER.
HOG features help evaluate handcrafted visual structure, ML classifiers provide
alternative decision boundaries, probability fusion combines CNN and ML
decisions, and stacking gives the best balanced performance.

## Future Work

Possible future improvements include:

- Testing on larger in-the-wild FER datasets.
- Improving minority-class recognition, especially `Disgust`.
- Exploring better calibration for probability fusion and stacking.
- Tuning confusion-aware correction thresholds and specialist models.
- Comparing with stronger CNN backbones while keeping the ML-enhancement
  analysis.
