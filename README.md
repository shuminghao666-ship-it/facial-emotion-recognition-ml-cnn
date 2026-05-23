# Emotion Recognition Project

GitHub repository:

```text
https://github.com/shuminghao666-ship-it/facial-emotion-recognition-ml-cnn
```

This project implements a facial emotion recognition pipeline for a DS4023
Machine Learning course project. It uses a seven-class FER2013-style dataset
and compares traditional machine learning baselines, a CNN baseline, and
ML-enhanced CNN fusion methods.

The target emotion classes are:

```text
Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
```

The project is organized into four ordered steps:

1. **Dataset analysis** (`code/00_dataset_analysis.py`) - check dataset
   structure, class distribution, corrupted images, and class imbalance.
2. **Traditional ML baselines** (`code/01_traditional_ml_baselines.py`) -
   compare raw pixels and HOG features with Logistic Regression, KNN, and
   SGD Linear SVM.
3. **CNN baseline and CLAHE comparison** (`code/02_cnn_baseline.py`) - train
   a CNN and compare CNN with CLAHE vs. CNN without CLAHE.
4. **ML-enhanced CNN analysis** (`code/03_ml_enhanced_cnn.py`) - use CNN
   features, PCA, HOG-CNN fusion, probability fusion, confusion-aware
   correction, and stacking to improve and analyze CNN predictions.

## Dataset

The dataset is included directly in this repository under:

```text
data/
```

Original Kaggle dataset source:

```text
https://www.kaggle.com/datasets/jayeshrohansingh/emotion-detection-dataset/data
```

This Kaggle dataset is an emotion detection image dataset organized by train
and test splits. Images are grouped into seven facial emotion classes. In this
project, the images are treated as low-resolution grayscale facial expression
samples for FER2013-style emotion classification.

Dataset summary:

- Training images: `28,384`
- Test images: `7,503`
- Total images: `35,887`
- Imbalance ratio: `16.43:1`

Class distribution:

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
is the smallest class. Because of this imbalance, Macro F1 is emphasized in
addition to accuracy.

## Repository Structure

```text
facial-emotion-recognition-ml-cnn/
├── data/
│   ├── train/
│   └── test/
├── code/
│   ├── 00_dataset_analysis.py
│   ├── 01_traditional_ml_baselines.py
│   ├── 02_cnn_baseline.py
│   ├── 03_ml_enhanced_cnn.py
│   ├── emotion_common.py
│   └── project_config.py
├── results/
│   ├── dataset_analysis/
│   ├── step1_result/
│   ├── step2_result/
│   └── step3_result/
├── requirements.txt
├── .gitignore
└── README.md
```

The report PDF, LaTeX source, course guideline PDF, model checkpoints, and
large regenerated model artifacts are not kept in this GitHub version.

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

Run commands from the repository root.

```bash
cd facial-emotion-recognition-ml-cnn
```

### 1. Run dataset analysis

```bash
python code/00_dataset_analysis.py
```

Main output:

```text
results/dataset_analysis/
```

### 2. Run traditional ML baselines

```bash
python code/01_traditional_ml_baselines.py
```

Models compared:

- Raw pixels + SGD Linear SVM
- HOG + Logistic Regression
- HOG + KNN
- HOG + SGD Linear SVM

Main output:

```text
results/step1_result/
```

### 3. Run CNN baseline

```bash
python code/02_cnn_baseline.py
```

Experiments:

- CNN with CLAHE
- CNN without CLAHE

Main output:

```text
results/step2_result/
```

### 4. Run ML-enhanced CNN analysis

```bash
python code/03_ml_enhanced_cnn.py
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

Main output:

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

HOG features perform much better than raw pixels. The strongest traditional
ML baseline is `HOG + KNN`.

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

```text
Stacked CNN-ML Probabilities + Balanced Logistic Regression
```

It improves Macro F1 from `0.5893` for the standalone CNN to `0.6078`, an
absolute gain of `0.0185`.

## Error Analysis Summary

The main confusion patterns are:

- `Angry` vs. `Disgust`: both are negative expressions and can share similar
  brow or nose-region patterns.
- `Fear` vs. `Surprise`: both can include widened eyes and open mouths.
- `Sad` vs. `Neutral`: both may involve subtle facial changes.
- `Fear`, `Sad`, and `Neutral`: low-resolution images make these negative or
  subtle expressions harder to separate.

The confusion-aware correction method is interpretable, but it does not exceed
the best stacking model. Stacking is more robust because it combines
probability outputs from multiple models globally.

## Main Contribution

This project shows that traditional machine learning methods are useful not
only as baselines, but also as complementary components for CNN-based facial
emotion recognition. HOG features help evaluate handcrafted visual structure,
ML classifiers provide alternative decision boundaries, probability fusion
combines CNN and ML decisions, and stacking gives the best balanced
performance.

## Future Work

Possible future improvements include:

- Improving minority-class recognition, especially `Disgust`.
- Exploring better calibration for probability fusion and stacking.
- Tuning confusion-aware correction thresholds and specialist models.
- Comparing with stronger CNN backbones while keeping the ML-enhancement
  analysis.
