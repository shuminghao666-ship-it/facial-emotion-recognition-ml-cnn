# ============================================================
# Pipeline Step 03:
# ML-Enhanced CNN Analysis for Facial Emotion Recognition
#
# Purpose:
# 1. Load the best CNN model trained in Step 02.
# 2. Extract CNN deep features from train / validation / test sets
# 3. Train traditional ML classifiers on CNN representations.
# 4. Compare PCA, HOG-CNN fusion, probability fusion, and stacking.
# 5. Apply Step 01-guided confusion-aware ML correction.
# 6. Generate Grad-CAM visualizations for correct and incorrect predictions.
# ============================================================

import json
import random
import time
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from torch.utils.data import DataLoader, Dataset

from emotion_common import (
    EmotionCNN,
    apply_clahe,
    get_device,
)

from project_config import (
    CLASSES,
    CLASS_FOLDER_NAMES,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    MODEL_DIR,
    RESULT_DIR,
    TEST_DIR,
    TRAIN_DIR,
)


# ============================================================
# 0. Global Configuration
# ============================================================

RANDOM_STATE = 42

# Must match Step 02.
VALID_SIZE = 0.15

# CNN feature extraction batch size
BATCH_SIZE_FEATURE_EXTRACTION = 256

# Traditional ML Settings
USE_CLASS_WEIGHT_FOR_ML = False
CLASS_WEIGHT = "balanced" if USE_CLASS_WEIGHT_FOR_ML else None

# HOG Settings: keep consistent with Step 01.
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

# Whether to run feature fusion experiment
RUN_FEATURE_FUSION = True

# Whether to run PCA feature-space visualization
RUN_PCA_VISUALIZATION = True
PCA_MAX_POINTS = 4000
RUN_PCA_ML_CLASSIFICATION = True
PCA_CLASSIFICATION_COMPONENTS = 64
RUN_CNN_ML_PROBABILITY_FUSION = True
RUN_STACKING_META_CLASSIFIER = True
RUN_CONFUSION_AWARE_CORRECTION = True

# Step 1 showed that HOG + KNN was the strongest traditional ML baseline.
# These groups target the most visible CNN confusion patterns.
CONFUSED_CLASS_GROUPS = {
    "Angry_Disgust": ["Angry", "Disgust"],
    "Fear_Sad_Neutral": ["Fear", "Sad", "Neutral"],
    "Fear_Surprise": ["Fear", "Surprise"],
    "Sad_Neutral": ["Sad", "Neutral"],
}

CORRECTION_CONFIDENCE_THRESHOLDS = [
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

CORRECTION_MARGIN_THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
]

CORRECTION_FUSION_ALPHAS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]

# Grad-CAM Sample Settings
GRADCAM_CORRECT_PER_CLASS = 1
GRADCAM_WRONG_PER_CLASS = 1


# ============================================================
# 1. Reproducibility and output folders
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

device = get_device()
print("Using device:", device)

OBJECTIVE3_DIR = RESULT_DIR / "step3_result"
OBJECTIVE3_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIR = OBJECTIVE3_DIR / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_DIR = OBJECTIVE3_DIR / "saved_ml_models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = OBJECTIVE3_DIR / "classification_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FIGURE_DIR = OBJECTIVE3_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

METRIC_DIR = OBJECTIVE3_DIR / "metrics"
METRIC_DIR.mkdir(parents=True, exist_ok=True)

CORRECTION_DIR = OBJECTIVE3_DIR / "confusion_correction"
CORRECTION_DIR.mkdir(parents=True, exist_ok=True)

GRADCAM_DIR = OBJECTIVE3_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

GRADCAM_PANEL_DIR = GRADCAM_DIR / "panels"
GRADCAM_PANEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Robust Image Reading
# ============================================================

def read_gray_image_unicode(image_path):
    """
    Read grayscale images from non-ASCII file paths.

    cv2.imread may fail for some non-ASCII paths.
    np.fromfile + cv2.imdecode is used instead.
    """
    try:
        image_path = str(image_path)
        image_data = np.fromfile(image_path, dtype=np.uint8)

        if image_data.size == 0:
            return None

        img = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
        return img

    except Exception as e:
        print(f"Failed to read image: {image_path}")
        print(f"Reason: {e}")
        return None


# ============================================================
# 3. Utility: class folder finding and image path collection
# ============================================================

def find_class_dir(split_dir, class_name):
    """
    Match actual folder names through CLASS_FOLDER_NAMES.
    """
    for folder_name in CLASS_FOLDER_NAMES[class_name]:
        class_dir = split_dir / folder_name
        if class_dir.exists():
            return class_dir

    return split_dir / class_name


def collect_image_paths(split_dir):
    """
    Collect all image paths and numerical labels.
    """
    image_paths = []
    labels = []

    print(f"\nCollecting images from: {split_dir}")

    for label, class_name in enumerate(CLASSES):
        class_dir = find_class_dir(split_dir, class_name)

        if not class_dir.exists():
            print("Folder not found:", class_dir)
            continue

        files = sorted(
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"{class_name}: {len(files)} files")

        for file_path in files:
            image_paths.append(str(file_path))
            labels.append(label)

    return np.array(image_paths), np.array(labels)


# ============================================================
# 4. Dataset for CNN inference / deep feature extraction
# ============================================================

class EmotionFeatureDataset(Dataset):
    """
    Dataset used for:
    - CNN end-to-end evaluation
    - CNN deep feature extraction

    No augmentation is used in Step 03.
    """
    def __init__(self, image_paths, labels, use_clahe):
        self.image_paths = image_paths
        self.labels = labels
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = int(self.labels[index])

        # ----------------------------------------------------
        # Robust image reading
        # ----------------------------------------------------
        img = read_gray_image_unicode(img_path)

        if img is None:
            print(f"Failed to read image, using blank fallback: {img_path}")
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        if self.use_clahe:
            img = apply_clahe(img)

        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        img_tensor = torch.tensor(img, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor, img_path


def create_feature_dataloaders(
    train_paths,
    val_paths,
    test_paths,
    train_labels,
    val_labels,
    test_labels,
    use_clahe,
):
    train_dataset = EmotionFeatureDataset(
        train_paths,
        train_labels,
        use_clahe=use_clahe,
    )

    val_dataset = EmotionFeatureDataset(
        val_paths,
        val_labels,
        use_clahe=use_clahe,
    )

    test_dataset = EmotionFeatureDataset(
        test_paths,
        test_labels,
        use_clahe=use_clahe,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_FEATURE_EXTRACTION,
        shuffle=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_FEATURE_EXTRACTION,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE_FEATURE_EXTRACTION,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 5. Load the Best CNN Model from Step 02
# ============================================================

def load_best_cnn_model():
    """
    Load the Step 02 selected best CNN:
        models / best_emotion_cnn_pytorch.pth
    """
    best_model_path = MODEL_DIR / "best_emotion_cnn_pytorch.pth"

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"\nCannot find best CNN model:\n{best_model_path}\n"
            "Please run Step 02 successfully first."
        )

    checkpoint = torch.load(best_model_path, map_location=device)

    model = EmotionCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    use_clahe = checkpoint.get("use_clahe", True)
    experiment_name = checkpoint.get("experiment_name", "unknown")
    best_val_acc = checkpoint.get("best_val_acc", None)

    print("\n" + "=" * 70)
    print("Loaded Best CNN Model from Step 02")
    print("=" * 70)
    print("Path:", best_model_path)
    print("Selected Experiment:", experiment_name)
    print("Use CLAHE:", use_clahe)
    print("Best Validation Accuracy:", best_val_acc)

    return model, checkpoint, use_clahe


# ============================================================
# 6. Metrics / report / confusion matrix
# ============================================================

def compute_metrics(y_true, y_pred, model_name, feature_type, dataset_name):
    accuracy = accuracy_score(y_true, y_pred)

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "Feature_Type": feature_type,
        "Model": model_name,
        "Dataset": dataset_name,
        "Accuracy": accuracy,
        "Macro_Precision": macro_precision,
        "Macro_Recall": macro_recall,
        "Macro_F1": macro_f1,
        "Weighted_Precision": weighted_precision,
        "Weighted_Recall": weighted_recall,
        "Weighted_F1": weighted_f1,
    }

    print("\n" + "=" * 70)
    print(f"{feature_type} | {model_name} | {dataset_name}")
    print("=" * 70)
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Macro Precision   : {macro_precision:.4f}")
    print(f"Macro Recall      : {macro_recall:.4f}")
    print(f"Macro F1-score    : {macro_f1:.4f}")
    print(f"Weighted F1-score : {weighted_f1:.4f}")

    return metrics


def save_classification_report(
    y_true,
    y_pred,
    model_name,
    feature_type,
    dataset_name,
):
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )

    print("\nClassification Report:")
    print(report)

    save_path = (
        REPORT_DIR
        / f"classification_report_{feature_type}_{model_name}_{dataset_name}.txt"
    )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)


def save_confusion_matrix_plot(
    y_true,
    y_pred,
    model_name,
    feature_type,
    dataset_name,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(CLASSES)),
    )

    fig, ax = plt.subplots(figsize=(9, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASSES,
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=False,
    )

    plt.title(f"{feature_type} - {model_name} - {dataset_name}")
    plt.tight_layout()

    save_path = (
        FIGURE_DIR
        / f"confusion_matrix_{feature_type}_{model_name}_{dataset_name}.png"
    )

    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# 7. CNN baseline evaluation
# ============================================================

def evaluate_cnn_model(model, dataloader, dataset_name):
    """
    Evaluate the end-to-end CNN on validation or test set.
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.inference_mode():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.concatenate(all_probs, axis=0)

    metrics = compute_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        model_name="End_to_End_CNN",
        feature_type="CNN_EndToEnd",
        dataset_name=dataset_name,
    )

    save_classification_report(
        y_true=all_labels,
        y_pred=all_preds,
        model_name="End_to_End_CNN",
        feature_type="CNN_EndToEnd",
        dataset_name=dataset_name,
    )

    save_confusion_matrix_plot(
        y_true=all_labels,
        y_pred=all_preds,
        model_name="End_to_End_CNN",
        feature_type="CNN_EndToEnd",
        dataset_name=dataset_name,
    )

    return metrics, all_labels, all_preds, all_probs


def collect_cnn_outputs(model, dataloader, split_name):
    """
    Collect CNN labels, predictions, and softmax probabilities without
    writing reports. Used by the correction models as ML input features.
    """
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    print(f"\nCollecting CNN softmax outputs for {split_name}...")

    with torch.inference_mode():
        for images, labels, _ in dataloader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.concatenate(all_probs, axis=0)

    print(f"{split_name} CNN output shape:", all_probs.shape)

    return all_labels, all_preds, all_probs


# ============================================================
# 8. Deep feature extraction
# ============================================================

def find_last_linear_layer(model):
    """
    Find the final nn.Linear layer.
    The input to this layer is used as the deep feature representation.
    """
    linear_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))

    if len(linear_layers) == 0:
        raise RuntimeError("No nn.Linear layer found in the CNN model.")

    last_name, last_layer = linear_layers[-1]

    print("\nLast Linear Layer Found:")
    print("Layer name:", last_name)
    print("Layer module:", last_layer)

    return last_name, last_layer


def extract_deep_features(model, dataloader, split_name):
    """
    Extract deep features through a forward hook on the final linear layer.
    """
    model.eval()

    _, last_linear_layer = find_last_linear_layer(model)

    captured_features = {}

    def forward_hook(module, inputs, outputs):
        captured_features["features"] = inputs[0].detach()

    hook_handle = last_linear_layer.register_forward_hook(forward_hook)

    all_features = []
    all_labels = []
    all_paths = []

    print(f"\nExtracting CNN deep features for {split_name}...")

    with torch.inference_mode():
        for images, labels, paths in dataloader:
            images = images.to(device)

            _ = model(images)

            batch_features = captured_features["features"]
            batch_features = batch_features.cpu().numpy()

            all_features.append(batch_features)
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    hook_handle.remove()

    X_features = np.concatenate(all_features, axis=0)
    y_labels = np.array(all_labels)

    print(f"{split_name} deep feature shape:", X_features.shape)

    np.save(
        FEATURE_DIR / f"{split_name.lower()}_cnn_deep_features.npy",
        X_features,
    )

    np.save(
        FEATURE_DIR / f"{split_name.lower()}_labels.npy",
        y_labels,
    )

    pd.DataFrame({
        "image_path": all_paths,
        "label": y_labels,
    }).to_csv(
        FEATURE_DIR / f"{split_name.lower()}_paths_and_labels.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return X_features, y_labels, np.array(all_paths)


# ============================================================
# 9. HOG feature extraction for fusion experiment
# ============================================================

def extract_hog_feature_from_path(image_path, use_clahe):
    """
    Extract HOG features from one image.
    Uses robust image reading for non-ASCII file paths.
    """
    img = read_gray_image_unicode(image_path)

    if img is None:
        print(f"Failed to read image for HOG, using blank fallback: {image_path}")
        img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    if use_clahe:
        img = apply_clahe(img)

    img = img.astype(np.float32) / 255.0

    hog_feature = hog(
        img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        feature_vector=True,
    )

    return hog_feature.astype(np.float32)


def extract_hog_features(image_paths, split_name, use_clahe):
    """
    Extract HOG features for all samples in one split.
    """
    print(f"\nExtracting HOG features for {split_name}...")

    all_hog_features = []

    total = len(image_paths)

    for i, image_path in enumerate(image_paths):
        hog_feature = extract_hog_feature_from_path(
            image_path=image_path,
            use_clahe=use_clahe,
        )

        all_hog_features.append(hog_feature)

        if (i + 1) % 2000 == 0 or (i + 1) == total:
            print(f"{split_name}: {i + 1}/{total} images processed")

    X_hog = np.array(all_hog_features, dtype=np.float32)

    print(f"{split_name} HOG feature shape:", X_hog.shape)

    np.save(
        FEATURE_DIR / f"{split_name.lower()}_hog_features_for_fusion.npy",
        X_hog,
    )

    return X_hog


# ============================================================
# 10. Traditional ML on extracted features
# ============================================================

def build_ml_models():
    models = {
        "Logistic_Regression": LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE,
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1,
        ),

        "SGD_Linear_SVM": SGDClassifier(
            loss="hinge",
            alpha=0.001,
            class_weight=CLASS_WEIGHT,
            max_iter=1000,
            tol=1e-3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    return models


def train_and_evaluate_ml_models(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_type,
):
    """
    Standardize features, train LR / KNN / SGD Linear SVM,
    evaluate on validation and test sets.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    scaler_path = MODEL_SAVE_DIR / f"scaler_{feature_type}.pkl"
    joblib.dump(scaler, scaler_path)

    models = build_ml_models()
    results = []

    for model_name, model in models.items():
        print("\n" + "=" * 70)
        print(f"Training {feature_type} + {model_name}")
        print("=" * 70)

        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time

        model_path = MODEL_SAVE_DIR / f"{feature_type}_{model_name}.pkl"
        joblib.dump(model, model_path)

        print(f"Training completed in {training_time:.2f} seconds")
        print("Model saved to:", model_path)

        # Validation
        val_pred = model.predict(X_val_scaled)

        val_metrics = compute_metrics(
            y_true=y_val,
            y_pred=val_pred,
            model_name=model_name,
            feature_type=feature_type,
            dataset_name="Validation",
        )

        val_metrics["Training_Time_Seconds"] = training_time
        results.append(val_metrics)

        save_classification_report(
            y_true=y_val,
            y_pred=val_pred,
            model_name=model_name,
            feature_type=feature_type,
            dataset_name="Validation",
        )

        save_confusion_matrix_plot(
            y_true=y_val,
            y_pred=val_pred,
            model_name=model_name,
            feature_type=feature_type,
            dataset_name="Validation",
        )

        # Test
        test_pred = model.predict(X_test_scaled)

        test_metrics = compute_metrics(
            y_true=y_test,
            y_pred=test_pred,
            model_name=model_name,
            feature_type=feature_type,
            dataset_name="Test",
        )

        test_metrics["Training_Time_Seconds"] = training_time
        results.append(test_metrics)

        save_classification_report(
            y_true=y_test,
            y_pred=test_pred,
            model_name=model_name,
            feature_type=feature_type,
            dataset_name="Test",
        )

        save_confusion_matrix_plot(
            y_true=y_test,
            y_pred=test_pred,
            model_name=model_name,
            feature_type=feature_type,
            dataset_name="Test",
        )

    return results


def build_pca_features(X_train, X_val, X_test, feature_name):
    n_components = min(
        PCA_CLASSIFICATION_COMPONENTS,
        X_train.shape[1],
        X_train.shape[0] - 1,
    )

    print("\n" + "=" * 70)
    print(f"Building PCA-reduced features for {feature_name}")
    print(f"PCA components: {n_components}")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled).astype(np.float32)
    X_val_pca = pca.transform(X_val_scaled).astype(np.float32)
    X_test_pca = pca.transform(X_test_scaled).astype(np.float32)

    explained_variance = float(np.sum(pca.explained_variance_ratio_))

    print(f"Explained variance ratio: {explained_variance:.4f}")
    print("Train PCA shape:", X_train_pca.shape)
    print("Validation PCA shape:", X_val_pca.shape)
    print("Test PCA shape:", X_test_pca.shape)

    joblib.dump(scaler, MODEL_SAVE_DIR / f"pca_input_scaler_{feature_name}.pkl")
    joblib.dump(pca, MODEL_SAVE_DIR / f"pca_model_{feature_name}.pkl")

    np.save(FEATURE_DIR / f"train_{feature_name}_pca_features.npy", X_train_pca)
    np.save(FEATURE_DIR / f"validation_{feature_name}_pca_features.npy", X_val_pca)
    np.save(FEATURE_DIR / f"test_{feature_name}_pca_features.npy", X_test_pca)

    pd.DataFrame(
        {
            "Feature_Name": [feature_name],
            "PCA_Components": [n_components],
            "Explained_Variance_Ratio": [explained_variance],
        }
    ).to_csv(
        FEATURE_DIR / f"pca_summary_{feature_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return X_train_pca, X_val_pca, X_test_pca, explained_variance


def align_predict_proba(model, X, num_classes):
    raw_probs = model.predict_proba(X)
    aligned_probs = np.zeros((len(X), num_classes), dtype=np.float64)

    for source_index, class_index in enumerate(model.classes_):
        aligned_probs[:, int(class_index)] = raw_probs[:, source_index]

    return aligned_probs


def build_probability_ml_models():
    return {
        "Logistic_Regression": LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1,
        ),
    }


def train_and_evaluate_cnn_ml_probability_fusion(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    cnn_val_probs,
    cnn_test_probs,
    feature_type,
):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    scaler_path = MODEL_SAVE_DIR / f"fusion_probability_scaler_{feature_type}.pkl"
    joblib.dump(scaler, scaler_path)

    results = []
    alpha_grid = np.linspace(0.0, 1.0, 11)

    for model_name, model in build_probability_ml_models().items():
        print("\n" + "=" * 70)
        print(f"Training probability fusion model: {feature_type} + {model_name}")
        print("=" * 70)

        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time

        model_path = MODEL_SAVE_DIR / f"Probability_Fusion_{feature_type}_{model_name}.pkl"
        joblib.dump(model, model_path)

        ml_val_probs = align_predict_proba(model, X_val_scaled, len(CLASSES))
        ml_test_probs = align_predict_proba(model, X_test_scaled, len(CLASSES))

        best_alpha = None
        best_val_f1 = -1.0
        best_val_pred = None

        tuning_rows = []
        for alpha in alpha_grid:
            fused_val_probs = alpha * cnn_val_probs + (1.0 - alpha) * ml_val_probs
            val_pred = np.argmax(fused_val_probs, axis=1)
            val_macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
            val_accuracy = accuracy_score(y_val, val_pred)

            tuning_rows.append(
                {
                    "Feature_Type": feature_type,
                    "Model": model_name,
                    "CNN_Probability_Weight": alpha,
                    "Validation_Accuracy": val_accuracy,
                    "Validation_Macro_F1": val_macro_f1,
                }
            )

            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                best_alpha = alpha
                best_val_pred = val_pred

        pd.DataFrame(tuning_rows).to_csv(
            METRIC_DIR / f"probability_fusion_tuning_{feature_type}_{model_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        fused_test_probs = best_alpha * cnn_test_probs + (1.0 - best_alpha) * ml_test_probs
        test_pred = np.argmax(fused_test_probs, axis=1)

        fusion_model_name = f"CNN_ML_Prob_Fusion_{model_name}"

        val_metrics = compute_metrics(
            y_true=y_val,
            y_pred=best_val_pred,
            model_name=fusion_model_name,
            feature_type=feature_type,
            dataset_name="Validation",
        )
        val_metrics["Training_Time_Seconds"] = training_time
        val_metrics["CNN_Probability_Weight"] = best_alpha
        results.append(val_metrics)

        save_classification_report(
            y_true=y_val,
            y_pred=best_val_pred,
            model_name=fusion_model_name,
            feature_type=feature_type,
            dataset_name="Validation",
        )
        save_confusion_matrix_plot(
            y_true=y_val,
            y_pred=best_val_pred,
            model_name=fusion_model_name,
            feature_type=feature_type,
            dataset_name="Validation",
        )

        test_metrics = compute_metrics(
            y_true=y_test,
            y_pred=test_pred,
            model_name=fusion_model_name,
            feature_type=feature_type,
            dataset_name="Test",
        )
        test_metrics["Training_Time_Seconds"] = training_time
        test_metrics["CNN_Probability_Weight"] = best_alpha
        results.append(test_metrics)

        save_classification_report(
            y_true=y_test,
            y_pred=test_pred,
            model_name=fusion_model_name,
            feature_type=feature_type,
            dataset_name="Test",
        )
        save_confusion_matrix_plot(
            y_true=y_test,
            y_pred=test_pred,
            model_name=fusion_model_name,
            feature_type=feature_type,
            dataset_name="Test",
        )

        print("Probability fusion completed.")
        print(f"Best CNN probability weight: {best_alpha:.2f}")
        print(f"Validation Macro F1: {best_val_f1:.4f}")
        print(f"Training time: {training_time:.2f} seconds")

    return results


def train_probability_model(model_name, X_train, y_train, X_val, X_test):
    if model_name == "Logistic_Regression":
        model = LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE,
        )
    elif model_name == "KNN":
        model = KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported probability model: {model_name}")

    model.fit(X_train, y_train)
    val_probs = align_predict_proba(model, X_val, len(CLASSES))
    test_probs = align_predict_proba(model, X_test, len(CLASSES))
    return model, val_probs, test_probs


def train_and_evaluate_stacking_meta_classifier(
    y_train,
    y_val,
    y_test,
    cnn_val_probs,
    cnn_test_probs,
    feature_blocks,
):
    print("\n" + "=" * 70)
    print("Training stacking meta-classifier")
    print("=" * 70)

    val_probability_blocks = [cnn_val_probs]
    test_probability_blocks = [cnn_test_probs]
    stack_sources = ["CNN_Softmax"]

    for feature_name, feature_data in feature_blocks.items():
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(feature_data["X_train"]).astype(np.float32)
        X_val_scaled = scaler.transform(feature_data["X_val"]).astype(np.float32)
        X_test_scaled = scaler.transform(feature_data["X_test"]).astype(np.float32)

        joblib.dump(
            scaler,
            MODEL_SAVE_DIR / f"stacking_input_scaler_{feature_name}.pkl",
        )

        for model_name in ["Logistic_Regression", "KNN"]:
            start_time = time.time()
            model, val_probs, test_probs = train_probability_model(
                model_name=model_name,
                X_train=X_train_scaled,
                y_train=y_train,
                X_val=X_val_scaled,
                X_test=X_test_scaled,
            )
            training_time = time.time() - start_time

            joblib.dump(
                model,
                MODEL_SAVE_DIR / f"stacking_base_{feature_name}_{model_name}.pkl",
            )

            val_probability_blocks.append(val_probs)
            test_probability_blocks.append(test_probs)
            stack_sources.append(f"{feature_name}_{model_name}")

            print(
                f"Stacking base model trained: {feature_name} + {model_name} "
                f"({training_time:.2f}s)"
            )

    X_meta_val = np.hstack(val_probability_blocks)
    X_meta_test = np.hstack(test_probability_blocks)

    meta_models = {
        "Stacking_Logistic_Regression": LogisticRegression(
            C=1.0,
            max_iter=5000,
            solver="lbfgs",
            class_weight=None,
            random_state=RANDOM_STATE,
        ),
        "Stacking_Balanced_Logistic_Regression": LogisticRegression(
            C=1.0,
            max_iter=5000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    stack_source_df = pd.DataFrame(
        {
            "Source_Index": list(range(len(stack_sources))),
            "Source": stack_sources,
        }
    )
    stack_source_df.to_csv(
        METRIC_DIR / "stacking_probability_sources.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for meta_name, meta_model in meta_models.items():
        start_time = time.time()
        meta_model.fit(X_meta_val, y_val)
        training_time = time.time() - start_time

        joblib.dump(meta_model, MODEL_SAVE_DIR / f"{meta_name}.pkl")

        val_pred = meta_model.predict(X_meta_val)
        test_pred = meta_model.predict(X_meta_test)

        val_metrics = compute_metrics(
            y_true=y_val,
            y_pred=val_pred,
            model_name=meta_name,
            feature_type="Stacked_CNN_ML_Probabilities",
            dataset_name="Validation",
        )
        val_metrics["Training_Time_Seconds"] = training_time
        val_metrics["Stacking_Sources"] = " + ".join(stack_sources)
        results.append(val_metrics)

        save_classification_report(
            y_true=y_val,
            y_pred=val_pred,
            model_name=meta_name,
            feature_type="Stacked_CNN_ML_Probabilities",
            dataset_name="Validation",
        )
        save_confusion_matrix_plot(
            y_true=y_val,
            y_pred=val_pred,
            model_name=meta_name,
            feature_type="Stacked_CNN_ML_Probabilities",
            dataset_name="Validation",
        )

        test_metrics = compute_metrics(
            y_true=y_test,
            y_pred=test_pred,
            model_name=meta_name,
            feature_type="Stacked_CNN_ML_Probabilities",
            dataset_name="Test",
        )
        test_metrics["Training_Time_Seconds"] = training_time
        test_metrics["Stacking_Sources"] = " + ".join(stack_sources)
        results.append(test_metrics)

        save_classification_report(
            y_true=y_test,
            y_pred=test_pred,
            model_name=meta_name,
            feature_type="Stacked_CNN_ML_Probabilities",
            dataset_name="Test",
        )
        save_confusion_matrix_plot(
            y_true=y_test,
            y_pred=test_pred,
            model_name=meta_name,
            feature_type="Stacked_CNN_ML_Probabilities",
            dataset_name="Test",
        )

        print(f"Stacking meta-classifier completed: {meta_name}")
        print(f"Meta feature dimension: {X_meta_val.shape[1]}")
        print(f"Training time: {training_time:.2f} seconds")

    return results


# ============================================================
# 11. Step1-guided confusion-aware ML correction
# ============================================================

def build_correction_features(X_deep, X_hog, cnn_probs):
    """
    Build ML correction features from deep features, HOG features, and
    CNN uncertainty signals.
    """
    sorted_probs = np.sort(cnn_probs, axis=1)
    confidence = sorted_probs[:, -1]
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    entropy = -np.sum(cnn_probs * np.log(cnn_probs + 1e-12), axis=1)

    uncertainty_features = np.column_stack([
        confidence,
        margin,
        entropy,
    ]).astype(np.float32)

    return np.hstack([
        X_deep.astype(np.float32),
        X_hog.astype(np.float32),
        cnn_probs.astype(np.float32),
        uncertainty_features,
    ]).astype(np.float32)


def build_correction_candidate_models():
    """
    Candidate specialist models. KNN is included because Step 01 found
    HOG + distance-weighted KNN to be the best traditional ML baseline.
    """
    return {
        "Balanced_Logistic_Regression": LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "KNN_3_Distance": KNeighborsClassifier(
            n_neighbors=3,
            weights="distance",
            n_jobs=-1,
        ),
        "KNN_5_Distance": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1,
        ),
    }


def train_confusion_specialists(X_train, y_train, X_val, y_val):
    specialists = {}
    specialist_rows = []

    for group_name, class_names in CONFUSED_CLASS_GROUPS.items():
        group_indices = [CLASSES.index(class_name) for class_name in class_names]
        train_mask = np.isin(y_train, group_indices)
        val_mask = np.isin(y_val, group_indices)

        if len(np.unique(y_train[train_mask])) < 2 or val_mask.sum() == 0:
            print(f"Skipping correction group with insufficient data: {group_name}")
            continue

        best_model = None
        best_model_name = None
        best_val_f1 = -1.0
        best_val_acc = -1.0

        for model_name, model in build_correction_candidate_models().items():
            print("\n" + "=" * 70)
            print(f"Training correction specialist: {group_name} + {model_name}")
            print("=" * 70)

            start_time = time.time()
            model.fit(X_train[train_mask], y_train[train_mask])
            training_time = time.time() - start_time

            val_pred = model.predict(X_val[val_mask])
            val_f1 = f1_score(
                y_val[val_mask],
                val_pred,
                average="macro",
                zero_division=0,
            )
            val_acc = accuracy_score(y_val[val_mask], val_pred)

            specialist_rows.append(
                {
                    "Group": group_name,
                    "Classes": " / ".join(class_names),
                    "Model": model_name,
                    "Validation_Accuracy_Within_Group": val_acc,
                    "Validation_Macro_F1_Within_Group": val_f1,
                    "Training_Time_Seconds": training_time,
                    "Train_Samples": int(train_mask.sum()),
                    "Validation_Samples": int(val_mask.sum()),
                }
            )

            print(f"Validation Accuracy within group: {val_acc:.4f}")
            print(f"Validation Macro F1 within group: {val_f1:.4f}")
            print(f"Training time: {training_time:.2f} seconds")

            if val_f1 > best_val_f1:
                best_model = model
                best_model_name = model_name
                best_val_f1 = val_f1
                best_val_acc = val_acc

        specialists[group_name] = {
            "classes": class_names,
            "class_indices": group_indices,
            "model": best_model,
            "model_name": best_model_name,
            "validation_macro_f1": best_val_f1,
            "validation_accuracy": best_val_acc,
        }

        joblib.dump(
            best_model,
            MODEL_SAVE_DIR / f"confusion_specialist_{group_name}_{best_model_name}.pkl",
        )

        print("\nSelected correction specialist:")
        print(f"Group: {group_name}")
        print(f"Model: {best_model_name}")
        print(f"Validation Macro F1 within group: {best_val_f1:.4f}")

    specialist_df = pd.DataFrame(specialist_rows)
    specialist_df.to_csv(
        CORRECTION_DIR / "specialist_model_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return specialists, specialist_df


def correction_candidate_mask(cnn_probs, group_indices, confidence_threshold, margin_threshold):
    top2 = np.argsort(cnn_probs, axis=1)[:, -2:]
    top1 = top2[:, 1]
    top2_class = top2[:, 0]

    confidence = cnn_probs[np.arange(len(cnn_probs)), top1]
    margin = confidence - cnn_probs[np.arange(len(cnn_probs)), top2_class]

    group_candidate = (
        np.isin(top1, group_indices)
        & np.isin(top2_class, group_indices)
    )
    uncertain = (
        (confidence <= confidence_threshold)
        | (margin <= margin_threshold)
    )

    return group_candidate & uncertain


def align_group_predict_proba(model, X, group_indices):
    raw_probs = model.predict_proba(X)
    aligned_probs = np.zeros((len(X), len(group_indices)), dtype=np.float64)
    group_position = {class_index: i for i, class_index in enumerate(group_indices)}

    for source_index, class_index in enumerate(model.classes_):
        class_index = int(class_index)
        if class_index in group_position:
            aligned_probs[:, group_position[class_index]] = raw_probs[:, source_index]

    row_sums = aligned_probs.sum(axis=1, keepdims=True)
    return np.divide(
        aligned_probs,
        row_sums,
        out=np.zeros_like(aligned_probs),
        where=row_sums != 0,
    )


def apply_confusion_correction(
    X_features,
    cnn_probs,
    specialists,
    confidence_threshold,
    margin_threshold,
    cnn_fusion_alpha,
):
    final_pred = np.argmax(cnn_probs, axis=1).astype(np.int64)
    corrected_group = np.array(["CNN"] * len(final_pred), dtype=object)
    already_corrected = np.zeros(len(final_pred), dtype=bool)

    correction_counts = {}

    for group_name, specialist in specialists.items():
        candidate_mask = correction_candidate_mask(
            cnn_probs=cnn_probs,
            group_indices=specialist["class_indices"],
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
        )
        candidate_mask = candidate_mask & ~already_corrected

        correction_counts[group_name] = int(candidate_mask.sum())

        if candidate_mask.sum() == 0:
            continue

        group_indices = specialist["class_indices"]
        specialist_probs = align_group_predict_proba(
            model=specialist["model"],
            X=X_features[candidate_mask],
            group_indices=group_indices,
        )
        cnn_group_probs = cnn_probs[candidate_mask][:, group_indices]
        cnn_group_sums = cnn_group_probs.sum(axis=1, keepdims=True)
        cnn_group_probs = np.divide(
            cnn_group_probs,
            cnn_group_sums,
            out=np.zeros_like(cnn_group_probs),
            where=cnn_group_sums != 0,
        )

        fused_group_probs = (
            cnn_fusion_alpha * cnn_group_probs
            + (1.0 - cnn_fusion_alpha) * specialist_probs
        )
        specialist_pred = np.array(group_indices)[np.argmax(fused_group_probs, axis=1)]

        final_pred[candidate_mask] = specialist_pred
        corrected_group[candidate_mask] = group_name
        already_corrected[candidate_mask] = True

    return final_pred, corrected_group, correction_counts


def tune_confusion_correction_thresholds(
    X_val,
    y_val,
    cnn_val_probs,
    specialists,
):
    tuning_rows = []
    best_row = None
    best_pred = None
    best_groups = None

    for confidence_threshold in CORRECTION_CONFIDENCE_THRESHOLDS:
        for margin_threshold in CORRECTION_MARGIN_THRESHOLDS:
            for cnn_fusion_alpha in CORRECTION_FUSION_ALPHAS:
                val_pred, corrected_group, correction_counts = apply_confusion_correction(
                    X_features=X_val,
                    cnn_probs=cnn_val_probs,
                    specialists=specialists,
                    confidence_threshold=confidence_threshold,
                    margin_threshold=margin_threshold,
                    cnn_fusion_alpha=cnn_fusion_alpha,
                )

                metric_values = {
                    "Accuracy": accuracy_score(y_val, val_pred),
                    "Macro_Precision": precision_score(
                        y_val,
                        val_pred,
                        average="macro",
                        zero_division=0,
                    ),
                    "Macro_Recall": recall_score(
                        y_val,
                        val_pred,
                        average="macro",
                        zero_division=0,
                    ),
                    "Macro_F1": f1_score(
                        y_val,
                        val_pred,
                        average="macro",
                        zero_division=0,
                    ),
                    "Weighted_Precision": precision_score(
                        y_val,
                        val_pred,
                        average="weighted",
                        zero_division=0,
                    ),
                    "Weighted_Recall": recall_score(
                        y_val,
                        val_pred,
                        average="weighted",
                        zero_division=0,
                    ),
                    "Weighted_F1": f1_score(
                        y_val,
                        val_pred,
                        average="weighted",
                        zero_division=0,
                    ),
                }

                row = {
                    "Confidence_Threshold": confidence_threshold,
                    "Margin_Threshold": margin_threshold,
                    "CNN_Fusion_Alpha": cnn_fusion_alpha,
                    "Total_Corrected": int(np.sum(corrected_group != "CNN")),
                    **correction_counts,
                    "Feature_Type": "Step1_Guided_Confusion_Correction",
                    "Model": "Specialist_ML_Probability_Correction",
                    "Dataset": "Validation_Tuning",
                    **metric_values,
                }
                tuning_rows.append(row)

                if best_row is None or metric_values["Macro_F1"] > best_row["Macro_F1"]:
                    best_row = row
                    best_pred = val_pred
                    best_groups = corrected_group

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(
        CORRECTION_DIR / "correction_threshold_tuning.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSelected confusion-correction thresholds:")
    print(f"Confidence threshold: {best_row['Confidence_Threshold']}")
    print(f"Margin threshold: {best_row['Margin_Threshold']}")
    print(f"CNN fusion alpha: {best_row['CNN_Fusion_Alpha']}")
    print(f"Validation Macro F1: {best_row['Macro_F1']:.4f}")
    print(f"Corrected validation samples: {best_row['Total_Corrected']}")

    return best_row, best_pred, best_groups, tuning_df


def save_correction_activation_summary(split_name, true_labels, pred_labels, corrected_group):
    rows = []

    for group_name in ["CNN", *CONFUSED_CLASS_GROUPS.keys()]:
        mask = corrected_group == group_name
        if mask.sum() == 0:
            accuracy = 0.0
            macro_f1 = 0.0
        else:
            accuracy = accuracy_score(true_labels[mask], pred_labels[mask])
            macro_f1 = f1_score(
                true_labels[mask],
                pred_labels[mask],
                average="macro",
                zero_division=0,
            )

        rows.append(
            {
                "Split": split_name,
                "Decision_Source": group_name,
                "Samples": int(mask.sum()),
                "Accuracy": accuracy,
                "Macro_F1": macro_f1,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(
        CORRECTION_DIR / f"correction_activation_summary_{split_name.lower()}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return df


def train_and_evaluate_confusion_aware_correction(
    X_train_deep,
    X_train_hog,
    y_train,
    cnn_train_probs,
    X_val_deep,
    X_val_hog,
    y_val,
    cnn_val_probs,
    X_test_deep,
    X_test_hog,
    y_test,
    cnn_test_probs,
):
    """
    Step1-guided correction:
    - HOG is included because Step 01 found it useful.
    - KNN is included because HOG + KNN was the strongest ML baseline.
    - Specialists are activated only for confused groups and uncertain CNN cases.
    """
    print("\n" + "=" * 70)
    print("Step1-Guided Confusion-Aware ML Correction")
    print("=" * 70)

    X_train_correction = build_correction_features(
        X_deep=X_train_deep,
        X_hog=X_train_hog,
        cnn_probs=cnn_train_probs,
    )
    X_val_correction = build_correction_features(
        X_deep=X_val_deep,
        X_hog=X_val_hog,
        cnn_probs=cnn_val_probs,
    )
    X_test_correction = build_correction_features(
        X_deep=X_test_deep,
        X_hog=X_test_hog,
        cnn_probs=cnn_test_probs,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_correction).astype(np.float32)
    X_val_scaled = scaler.transform(X_val_correction).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_correction).astype(np.float32)

    joblib.dump(
        scaler,
        MODEL_SAVE_DIR / "confusion_correction_feature_scaler.pkl",
    )

    specialists, specialist_df = train_confusion_specialists(
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_val_scaled,
        y_val=y_val,
    )

    if not specialists:
        print("No confusion specialists were trained. Skipping correction.")
        return []

    best_thresholds, val_pred, val_groups, tuning_df = (
        tune_confusion_correction_thresholds(
            X_val=X_val_scaled,
            y_val=y_val,
            cnn_val_probs=cnn_val_probs,
            specialists=specialists,
        )
    )

    test_pred, test_groups, test_correction_counts = apply_confusion_correction(
        X_features=X_test_scaled,
        cnn_probs=cnn_test_probs,
        specialists=specialists,
        confidence_threshold=best_thresholds["Confidence_Threshold"],
        margin_threshold=best_thresholds["Margin_Threshold"],
        cnn_fusion_alpha=best_thresholds["CNN_Fusion_Alpha"],
    )

    save_correction_activation_summary(
        split_name="Validation",
        true_labels=y_val,
        pred_labels=val_pred,
        corrected_group=val_groups,
    )
    save_correction_activation_summary(
        split_name="Test",
        true_labels=y_test,
        pred_labels=test_pred,
        corrected_group=test_groups,
    )

    results = []

    for dataset_name, true_labels, pred_labels in [
        ("Validation", y_val, val_pred),
        ("Test", y_test, test_pred),
    ]:
        metrics = compute_metrics(
            y_true=true_labels,
            y_pred=pred_labels,
            model_name="Specialist_ML_Probability_Correction",
            feature_type="Step1_Guided_Confusion_Correction",
            dataset_name=dataset_name,
        )
        metrics["Confidence_Threshold"] = best_thresholds["Confidence_Threshold"]
        metrics["Margin_Threshold"] = best_thresholds["Margin_Threshold"]
        metrics["CNN_Fusion_Alpha"] = best_thresholds["CNN_Fusion_Alpha"]
        metrics["Correction_Features"] = (
            "CNN deep features + HOG features + CNN probabilities + "
            "confidence/margin/entropy"
        )
        results.append(metrics)

        save_classification_report(
            y_true=true_labels,
            y_pred=pred_labels,
            model_name="Specialist_ML_Probability_Correction",
            feature_type="Step1_Guided_Confusion_Correction",
            dataset_name=dataset_name,
        )
        save_confusion_matrix_plot(
            y_true=true_labels,
            y_pred=pred_labels,
            model_name="Specialist_ML_Probability_Correction",
            feature_type="Step1_Guided_Confusion_Correction",
            dataset_name=dataset_name,
        )

    pd.DataFrame(
        {
            "Test_Group": list(test_correction_counts.keys()),
            "Corrected_Samples": list(test_correction_counts.values()),
        }
    ).to_csv(
        CORRECTION_DIR / "test_correction_counts_by_group.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return results


# ============================================================
# 12. PCA feature-space visualization
# ============================================================

def plot_pca_feature_space(X, y, title, save_name):
    """
    PCA 2D visualization of feature separability.
    """
    if len(X) > PCA_MAX_POINTS:
        rng = np.random.default_rng(RANDOM_STATE)
        selected_indices = rng.choice(
            len(X),
            size=PCA_MAX_POINTS,
            replace=False,
        )

        X_plot = X[selected_indices]
        y_plot = y[selected_indices]
    else:
        X_plot = X
        y_plot = y

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_plot)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)

    plt.figure(figsize=(9, 7))

    for label_idx, class_name in enumerate(CLASSES):
        mask = y_plot == label_idx

        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=12,
            alpha=0.65,
            label=class_name,
        )

    explained_var = pca.explained_variance_ratio_.sum() * 100

    plt.title(f"{title}\nPCA Explained Variance: {explained_var:.2f}%")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / save_name, dpi=300)
    plt.close()


# ============================================================
# 12. Optional Step 01 Result Loading
# ============================================================

def try_load_objective1_results():
    """
    Optional: automatically copy Step 01 summary if available.
    """
    possible_paths = [
        RESULT_DIR
        / "step1_result"
        / "objective1_traditional_ml_metrics_summary.csv",

        RESULT_DIR
        / "step1_result"
        / "test_metrics_summary.csv",

        Path("results")
        / "step1_result"
        / "objective1_traditional_ml_metrics_summary.csv",
    ]

    for path in possible_paths:
        if path.exists():
            print("\nStep 01 result file found:")
            print(path)

            df = pd.read_csv(path)
            return df, path

    print("\nStep 01 summary CSV not found. Skipping automatic merge.")
    return None, None


# ============================================================
# 13. Grad-CAM implementation
# ============================================================

def find_last_conv_layer(model):
    conv_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    if len(conv_layers) == 0:
        raise RuntimeError("No nn.Conv2d layer found in the CNN model.")

    last_name, last_layer = conv_layers[-1]

    print("\nLast Conv Layer Found for Grad-CAM:")
    print("Layer name:", last_name)
    print("Layer module:", last_layer)

    return last_name, last_layer


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()

        _, self.target_layer = find_last_conv_layer(model)

        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self.forward_hook
        )

        self.backward_handle = self.target_layer.register_full_backward_hook(
            self.backward_hook
        )

    def forward_hook(self, module, inputs, outputs):
        self.activations = outputs

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        input_tensor = input_tensor.to(device)

        self.model.zero_grad()

        outputs = self.model(input_tensor)

        predicted_class = int(torch.argmax(outputs, dim=1).item())

        if target_class is None:
            target_class = predicted_class

        score = outputs[:, target_class]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(
            gradients,
            dim=(2, 3),
            keepdim=True,
        )

        cam = torch.sum(weights * activations, dim=1)
        cam = torch.relu(cam)

        cam = cam.squeeze(0)
        cam = cam.detach().cpu().numpy()

        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, predicted_class

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# 14. Grad-CAM visualization helpers
# ============================================================

def prepare_single_image_for_cnn(image_path, use_clahe):
    """
    Prepare one image exactly as CNN input.
    Uses robust image reading for non-ASCII file paths.
    """
    img = read_gray_image_unicode(image_path)

    if img is None:
        print(f"Failed to read image for Grad-CAM, using blank fallback: {image_path}")
        img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    display_img = img.copy()
    processed_img = img.copy()

    if use_clahe:
        processed_img = apply_clahe(processed_img)

    normalized_img = processed_img.astype(np.float32) / 255.0
    normalized_img = np.expand_dims(normalized_img, axis=0)
    normalized_img = np.expand_dims(normalized_img, axis=0)

    input_tensor = torch.tensor(
        normalized_img,
        dtype=torch.float32,
    )

    return display_img, processed_img, input_tensor


def save_gradcam_panel(
    image_path,
    true_label,
    predicted_label,
    gradcam,
    use_clahe,
    save_name,
):
    display_img, processed_img, input_tensor = prepare_single_image_for_cnn(
        image_path=image_path,
        use_clahe=use_clahe,
    )

    cam, _ = gradcam.generate(input_tensor)

    heatmap_uint8 = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    base_bgr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(
        base_bgr,
        0.6,
        heatmap_color,
        0.4,
        0,
    )

    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(display_img, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap_rgb)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    title = (
        f"True: {CLASSES[true_label]} | "
        f"Predicted: {CLASSES[predicted_label]}"
    )

    fig.suptitle(title)
    plt.tight_layout()

    save_path = GRADCAM_PANEL_DIR / save_name
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def generate_gradcam_examples(
    model,
    test_paths,
    true_labels,
    pred_labels,
    use_clahe,
):
    """
    Select correct and incorrect examples for each class,
    then save Grad-CAM panels.
    """
    gradcam = GradCAM(model)

    records = []

    print("\nGenerating Grad-CAM examples...")

    for class_idx, class_name in enumerate(CLASSES):
        correct_indices = np.where(
            (true_labels == class_idx) &
            (pred_labels == class_idx)
        )[0]

        wrong_indices = np.where(
            (true_labels == class_idx) &
            (pred_labels != class_idx)
        )[0]

        selected_correct = correct_indices[:GRADCAM_CORRECT_PER_CLASS]
        selected_wrong = wrong_indices[:GRADCAM_WRONG_PER_CLASS]

        for rank, sample_idx in enumerate(selected_correct, start=1):
            image_path = test_paths[sample_idx]
            predicted_label = int(pred_labels[sample_idx])

            save_name = (
                f"{class_name}_correct_{rank}_"
                f"pred_{CLASSES[predicted_label]}.png"
            )

            save_path = save_gradcam_panel(
                image_path=image_path,
                true_label=class_idx,
                predicted_label=predicted_label,
                gradcam=gradcam,
                use_clahe=use_clahe,
                save_name=save_name,
            )

            records.append({
                "class": class_name,
                "case_type": "correct",
                "image_path": image_path,
                "true_label": class_name,
                "predicted_label": CLASSES[predicted_label],
                "saved_panel": str(save_path),
            })

        for rank, sample_idx in enumerate(selected_wrong, start=1):
            image_path = test_paths[sample_idx]
            predicted_label = int(pred_labels[sample_idx])

            save_name = (
                f"{class_name}_wrong_{rank}_"
                f"pred_{CLASSES[predicted_label]}.png"
            )

            save_path = save_gradcam_panel(
                image_path=image_path,
                true_label=class_idx,
                predicted_label=predicted_label,
                gradcam=gradcam,
                use_clahe=use_clahe,
                save_name=save_name,
            )

            records.append({
                "class": class_name,
                "case_type": "wrong",
                "image_path": image_path,
                "true_label": class_name,
                "predicted_label": CLASSES[predicted_label],
                "saved_panel": str(save_path),
            })

    gradcam.remove_hooks()

    gradcam_df = pd.DataFrame(records)

    gradcam_df.to_csv(
        GRADCAM_DIR / "gradcam_sample_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Grad-CAM panels saved to:")
    print(GRADCAM_PANEL_DIR)

    return gradcam_df


# ============================================================
# 15. Comparison charts
# ============================================================

def plot_metric_comparison(results_df, dataset_name, metric_name):
    subset = results_df[results_df["Dataset"] == dataset_name].copy()

    subset["Method"] = (
        subset["Feature_Type"].astype(str)
        + " + "
        + subset["Model"].astype(str)
    )

    subset = subset.sort_values(metric_name, ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(subset["Method"], subset[metric_name])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Comparison on {dataset_name}")
    plt.ylim(0, 1)

    for i, value in enumerate(subset[metric_name]):
        plt.text(i, value + 0.01, f"{value:.4f}", ha="center", fontsize=8)

    plt.tight_layout()

    save_path = FIGURE_DIR / f"comparison_{dataset_name}_{metric_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# 16. Write Step 03 Summary
# ============================================================

def write_objective3_summary(results_df, objective1_df=None):
    summary_path = OBJECTIVE3_DIR / "objective3_summary.txt"

    validation_df = results_df[results_df["Dataset"] == "Validation"].copy()
    test_df = results_df[results_df["Dataset"] == "Test"].copy()

    best_validation = validation_df.sort_values(
        "Macro_F1",
        ascending=False,
    ).iloc[0]

    best_test = test_df.sort_values(
        "Macro_F1",
        ascending=False,
    ).iloc[0]

    lines = []
    lines.append("Pipeline Step 03 Summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Tasks completed:")
    lines.append("1. Loaded the best CNN model from Step 02.")
    lines.append("2. Extracted CNN deep features from the penultimate representation layer.")
    lines.append("3. Trained Logistic Regression, KNN, and SGD Linear SVM on CNN deep features.")
    lines.append("4. Applied PCA to CNN deep features and trained ML classifiers on the reduced feature space.")
    lines.append("5. Extracted HOG features and fused them with CNN deep features.")
    lines.append("6. Trained Logistic Regression, KNN, and SGD Linear SVM on fused features.")
    lines.append("7. Built CNN + ML probability fusion models and selected fusion weights on the validation set.")
    lines.append("8. Added Step1-guided confusion-aware ML correction for uncertain CNN predictions.")
    lines.append("9. Trained stacking meta-classifiers using CNN and ML probability outputs.")
    lines.append("10. Compared end-to-end CNN, CNN-feature ML, PCA-feature ML, HOG-CNN fusion, probability-fusion, confusion correction, and stacking models.")
    lines.append("11. Generated Grad-CAM visual explanations.")
    lines.append("")
    lines.append("Best Validation Macro-F1 Model:")
    lines.append(
        f"{best_validation['Feature_Type']} + {best_validation['Model']}"
    )
    lines.append(f"Validation Accuracy: {best_validation['Accuracy']:.4f}")
    lines.append(f"Validation Macro F1: {best_validation['Macro_F1']:.4f}")
    lines.append("")
    lines.append("Best Test Macro-F1 Model:")
    lines.append(
        f"{best_test['Feature_Type']} + {best_test['Model']}"
    )
    lines.append(f"Test Accuracy: {best_test['Accuracy']:.4f}")
    lines.append(f"Test Macro F1: {best_test['Macro_F1']:.4f}")
    lines.append(f"Test Weighted F1: {best_test['Weighted_F1']:.4f}")
    lines.append("")

    if objective1_df is not None:
        lines.append("Step 01 result file was found.")
        lines.append(
            "Step 01 and Step 03 use different validation split ratios, "
            "so validation metrics should not be directly compared."
        )
        lines.append(
            "However, test-set metrics can still be used as broad baseline references."
        )
        lines.append(
            "Step 03 uses Step 01 insight directly: HOG and distance-weighted "
            "KNN are included in the specialist correction stage because they were the "
            "strongest traditional ML baseline signals."
        )
        lines.append("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nStep 03 summary saved to:")
    print(summary_path)


# ============================================================
# 17. Main pipeline
# ============================================================

def main():
    # --------------------------------------------------------
    # Step 1. Collect images
    # --------------------------------------------------------
    train_all_paths, train_all_labels = collect_image_paths(TRAIN_DIR)
    test_paths, test_labels = collect_image_paths(TEST_DIR)

    print("\nTotal train/validation images:", len(train_all_paths))
    print("Total test images:", len(test_paths))

    if len(train_all_paths) == 0 or len(test_paths) == 0:
        print("No images found. Stop.")
        return

    # --------------------------------------------------------
    # Step 2. Recreate the same split logic as Step 02
    # --------------------------------------------------------
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_all_paths,
        train_all_labels,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_all_labels,
    )

    print("\nDataset split for Step 03:")
    print("Train:", len(train_paths))
    print("Validation:", len(val_paths))
    print("Test:", len(test_paths))

    # --------------------------------------------------------
    # Step 3. Load Step 02 best CNN
    # --------------------------------------------------------
    model, checkpoint, use_clahe = load_best_cnn_model()

    # --------------------------------------------------------
    # Step 4. Create dataloaders
    # --------------------------------------------------------
    train_loader, val_loader, test_loader = create_feature_dataloaders(
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        use_clahe=use_clahe,
    )

    all_results = []

    # --------------------------------------------------------
    # Step 5. Evaluate end-to-end CNN baseline
    # --------------------------------------------------------
    cnn_val_metrics, cnn_val_true, cnn_val_pred, cnn_val_probs = evaluate_cnn_model(
        model=model,
        dataloader=val_loader,
        dataset_name="Validation",
    )

    cnn_test_metrics, cnn_test_true, cnn_test_pred, cnn_test_probs = evaluate_cnn_model(
        model=model,
        dataloader=test_loader,
        dataset_name="Test",
    )

    cnn_train_true, cnn_train_pred, cnn_train_probs = collect_cnn_outputs(
        model=model,
        dataloader=train_loader,
        split_name="Train",
    )

    all_results.append(cnn_val_metrics)
    all_results.append(cnn_test_metrics)

    # --------------------------------------------------------
    # Step 6. Extract CNN deep features
    # --------------------------------------------------------
    X_train_deep, y_train_deep, train_paths_deep = extract_deep_features(
        model=model,
        dataloader=train_loader,
        split_name="Train",
    )

    X_val_deep, y_val_deep, val_paths_deep = extract_deep_features(
        model=model,
        dataloader=val_loader,
        split_name="Validation",
    )

    X_test_deep, y_test_deep, test_paths_deep = extract_deep_features(
        model=model,
        dataloader=test_loader,
        split_name="Test",
    )

    # --------------------------------------------------------
    # Step 7. CNN deep features + traditional ML
    # --------------------------------------------------------
    deep_feature_results = train_and_evaluate_ml_models(
        X_train=X_train_deep,
        y_train=y_train_deep,
        X_val=X_val_deep,
        y_val=y_val_deep,
        X_test=X_test_deep,
        y_test=y_test_deep,
        feature_type="CNN_Deep_Feature",
    )

    all_results.extend(deep_feature_results)

    if RUN_CNN_ML_PROBABILITY_FUSION:
        deep_feature_fusion_results = train_and_evaluate_cnn_ml_probability_fusion(
            X_train=X_train_deep,
            y_train=y_train_deep,
            X_val=X_val_deep,
            y_val=y_val_deep,
            X_test=X_test_deep,
            y_test=y_test_deep,
            cnn_val_probs=cnn_val_probs,
            cnn_test_probs=cnn_test_probs,
            feature_type="CNN_Deep_Feature_Probability_Fusion",
        )
        all_results.extend(deep_feature_fusion_results)

    # --------------------------------------------------------
    # Step 8. PCA-reduced CNN features + traditional ML
    # --------------------------------------------------------
    if RUN_PCA_ML_CLASSIFICATION:
        X_train_deep_pca, X_val_deep_pca, X_test_deep_pca, pca_explained_var = (
            build_pca_features(
                X_train=X_train_deep,
                X_val=X_val_deep,
                X_test=X_test_deep,
                feature_name="cnn_deep_feature",
            )
        )

        pca_deep_feature_results = train_and_evaluate_ml_models(
            X_train=X_train_deep_pca,
            y_train=y_train_deep,
            X_val=X_val_deep_pca,
            y_val=y_val_deep,
            X_test=X_test_deep_pca,
            y_test=y_test_deep,
            feature_type="PCA_CNN_Deep_Feature",
        )

        for result in pca_deep_feature_results:
            result["PCA_Explained_Variance"] = pca_explained_var

        all_results.extend(pca_deep_feature_results)

        if RUN_CNN_ML_PROBABILITY_FUSION:
            pca_deep_feature_fusion_results = (
                train_and_evaluate_cnn_ml_probability_fusion(
                    X_train=X_train_deep_pca,
                    y_train=y_train_deep,
                    X_val=X_val_deep_pca,
                    y_val=y_val_deep,
                    X_test=X_test_deep_pca,
                    y_test=y_test_deep,
                    cnn_val_probs=cnn_val_probs,
                    cnn_test_probs=cnn_test_probs,
                    feature_type="PCA_CNN_Deep_Feature_Probability_Fusion",
                )
            )
            for result in pca_deep_feature_fusion_results:
                result["PCA_Explained_Variance"] = pca_explained_var
            all_results.extend(pca_deep_feature_fusion_results)

    # --------------------------------------------------------
    # Step 9. HOG + CNN deep feature fusion
    # --------------------------------------------------------
    X_train_hog = None
    X_val_hog = None
    X_test_hog = None
    X_train_fusion = None
    X_val_fusion = None
    X_test_fusion = None

    if RUN_FEATURE_FUSION:
        X_train_hog = extract_hog_features(
            image_paths=train_paths_deep,
            split_name="Train",
            use_clahe=use_clahe,
        )

        X_val_hog = extract_hog_features(
            image_paths=val_paths_deep,
            split_name="Validation",
            use_clahe=use_clahe,
        )

        X_test_hog = extract_hog_features(
            image_paths=test_paths_deep,
            split_name="Test",
            use_clahe=use_clahe,
        )

        # Standardize each feature block separately before concatenation
        deep_scaler_for_fusion = StandardScaler()
        hog_scaler_for_fusion = StandardScaler()

        X_train_deep_scaled_for_fusion = deep_scaler_for_fusion.fit_transform(
            X_train_deep
        ).astype(np.float32)

        X_val_deep_scaled_for_fusion = deep_scaler_for_fusion.transform(
            X_val_deep
        ).astype(np.float32)

        X_test_deep_scaled_for_fusion = deep_scaler_for_fusion.transform(
            X_test_deep
        ).astype(np.float32)

        X_train_hog_scaled_for_fusion = hog_scaler_for_fusion.fit_transform(
            X_train_hog
        ).astype(np.float32)

        X_val_hog_scaled_for_fusion = hog_scaler_for_fusion.transform(
            X_val_hog
        ).astype(np.float32)

        X_test_hog_scaled_for_fusion = hog_scaler_for_fusion.transform(
            X_test_hog
        ).astype(np.float32)

        joblib.dump(
            deep_scaler_for_fusion,
            MODEL_SAVE_DIR / "fusion_deep_feature_scaler.pkl",
        )

        joblib.dump(
            hog_scaler_for_fusion,
            MODEL_SAVE_DIR / "fusion_hog_feature_scaler.pkl",
        )

        X_train_fusion = np.hstack([
            X_train_deep_scaled_for_fusion,
            X_train_hog_scaled_for_fusion,
        ])

        X_val_fusion = np.hstack([
            X_val_deep_scaled_for_fusion,
            X_val_hog_scaled_for_fusion,
        ])

        X_test_fusion = np.hstack([
            X_test_deep_scaled_for_fusion,
            X_test_hog_scaled_for_fusion,
        ])

        print("\nFusion feature shape:")
        print("Train:", X_train_fusion.shape)
        print("Validation:", X_val_fusion.shape)
        print("Test:", X_test_fusion.shape)

        fusion_results = train_and_evaluate_ml_models(
            X_train=X_train_fusion,
            y_train=y_train_deep,
            X_val=X_val_fusion,
            y_val=y_val_deep,
            X_test=X_test_fusion,
            y_test=y_test_deep,
            feature_type="HOG_CNN_Fusion",
        )

        all_results.extend(fusion_results)

        if RUN_CNN_ML_PROBABILITY_FUSION:
            hog_cnn_probability_fusion_results = (
                train_and_evaluate_cnn_ml_probability_fusion(
                    X_train=X_train_fusion,
                    y_train=y_train_deep,
                    X_val=X_val_fusion,
                    y_val=y_val_deep,
                    X_test=X_test_fusion,
                    y_test=y_test_deep,
                    cnn_val_probs=cnn_val_probs,
                    cnn_test_probs=cnn_test_probs,
                    feature_type="HOG_CNN_Fusion_Probability_Fusion",
                )
            )
            all_results.extend(hog_cnn_probability_fusion_results)

    # --------------------------------------------------------
    # Step 10. Step1-guided confusion-aware ML correction
    # --------------------------------------------------------
    if RUN_CONFUSION_AWARE_CORRECTION:
        if X_train_hog is None or X_val_hog is None or X_test_hog is None:
            X_train_hog = extract_hog_features(
                image_paths=train_paths_deep,
                split_name="Train",
                use_clahe=use_clahe,
            )

            X_val_hog = extract_hog_features(
                image_paths=val_paths_deep,
                split_name="Validation",
                use_clahe=use_clahe,
            )

            X_test_hog = extract_hog_features(
                image_paths=test_paths_deep,
                split_name="Test",
                use_clahe=use_clahe,
            )

        correction_results = train_and_evaluate_confusion_aware_correction(
            X_train_deep=X_train_deep,
            X_train_hog=X_train_hog,
            y_train=y_train_deep,
            cnn_train_probs=cnn_train_probs,
            X_val_deep=X_val_deep,
            X_val_hog=X_val_hog,
            y_val=y_val_deep,
            cnn_val_probs=cnn_val_probs,
            X_test_deep=X_test_deep,
            X_test_hog=X_test_hog,
            y_test=y_test_deep,
            cnn_test_probs=cnn_test_probs,
        )
        all_results.extend(correction_results)

    # --------------------------------------------------------
    # Step 11. Stacking meta-classifier over CNN and ML probabilities
    # --------------------------------------------------------
    if RUN_STACKING_META_CLASSIFIER:
        stacking_feature_blocks = {
            "CNN_Deep_Feature": {
                "X_train": X_train_deep,
                "X_val": X_val_deep,
                "X_test": X_test_deep,
            }
        }

        if RUN_PCA_ML_CLASSIFICATION:
            stacking_feature_blocks["PCA_CNN_Deep_Feature"] = {
                "X_train": X_train_deep_pca,
                "X_val": X_val_deep_pca,
                "X_test": X_test_deep_pca,
            }

        if RUN_FEATURE_FUSION and X_train_fusion is not None:
            stacking_feature_blocks["HOG_CNN_Fusion"] = {
                "X_train": X_train_fusion,
                "X_val": X_val_fusion,
                "X_test": X_test_fusion,
            }

        stacking_results = train_and_evaluate_stacking_meta_classifier(
            y_train=y_train_deep,
            y_val=y_val_deep,
            y_test=y_test_deep,
            cnn_val_probs=cnn_val_probs,
            cnn_test_probs=cnn_test_probs,
            feature_blocks=stacking_feature_blocks,
        )
        all_results.extend(stacking_results)

    # --------------------------------------------------------
    # Step 11. PCA feature visualization
    # --------------------------------------------------------
    if RUN_PCA_VISUALIZATION:
        plot_pca_feature_space(
            X=X_test_deep,
            y=y_test_deep,
            title="CNN Deep Feature Space on Test Set",
            save_name="pca_cnn_deep_feature_test.png",
        )

        if RUN_FEATURE_FUSION:
            plot_pca_feature_space(
                X=X_test_fusion,
                y=y_test_deep,
                title="HOG + CNN Fusion Feature Space on Test Set",
                save_name="pca_hog_cnn_fusion_feature_test.png",
            )

    # --------------------------------------------------------
    # Step 12. Grad-CAM explainability
    # --------------------------------------------------------
    generate_gradcam_examples(
        model=model,
        test_paths=test_paths_deep,
        true_labels=cnn_test_true,
        pred_labels=cnn_test_pred,
        use_clahe=use_clahe,
    )

    # --------------------------------------------------------
    # Step 13. Save result summary
    # --------------------------------------------------------
    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        OBJECTIVE3_DIR / "objective3_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 70)
    print("Step 03 Metrics Summary")
    print("=" * 70)
    print(results_df.round(4))

    results_df[results_df["Dataset"] == "Validation"].to_csv(
        OBJECTIVE3_DIR / "objective3_validation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    results_df[results_df["Dataset"] == "Test"].to_csv(
        OBJECTIVE3_DIR / "objective3_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Step 14. Metric comparison plots
    # --------------------------------------------------------
    plot_metric_comparison(
        results_df=results_df,
        dataset_name="Validation",
        metric_name="Accuracy",
    )

    plot_metric_comparison(
        results_df=results_df,
        dataset_name="Validation",
        metric_name="Macro_F1",
    )

    plot_metric_comparison(
        results_df=results_df,
        dataset_name="Test",
        metric_name="Accuracy",
    )

    plot_metric_comparison(
        results_df=results_df,
        dataset_name="Test",
        metric_name="Macro_F1",
    )

    # --------------------------------------------------------
    # Step 15. Optional Step 01 Result Reference
    # --------------------------------------------------------
    objective1_df, objective1_path = try_load_objective1_results()

    if objective1_df is not None:
        objective1_df.to_csv(
            OBJECTIVE3_DIR / "copied_objective1_metrics_for_reference.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # Step 16. Text summary
    # --------------------------------------------------------
    write_objective3_summary(
        results_df=results_df,
        objective1_df=objective1_df,
    )

    # --------------------------------------------------------
    # Step 17. Save Configuration
    # --------------------------------------------------------
    config = {
        "random_state": RANDOM_STATE,
        "validation_size": VALID_SIZE,
        "cnn_feature_source": "input to last Linear layer",
        "use_clahe_from_best_checkpoint": use_clahe,
        "run_feature_fusion": RUN_FEATURE_FUSION,
        "run_pca_visualization": RUN_PCA_VISUALIZATION,
        "run_pca_ml_classification": RUN_PCA_ML_CLASSIFICATION,
        "pca_classification_components": PCA_CLASSIFICATION_COMPONENTS,
        "run_cnn_ml_probability_fusion": RUN_CNN_ML_PROBABILITY_FUSION,
        "run_confusion_aware_correction": RUN_CONFUSION_AWARE_CORRECTION,
        "confused_class_groups": CONFUSED_CLASS_GROUPS,
        "correction_confidence_thresholds": CORRECTION_CONFIDENCE_THRESHOLDS,
        "correction_margin_thresholds": CORRECTION_MARGIN_THRESHOLDS,
        "correction_fusion_alphas": CORRECTION_FUSION_ALPHAS,
        "run_stacking_meta_classifier": RUN_STACKING_META_CLASSIFIER,
        "gradcam_correct_per_class": GRADCAM_CORRECT_PER_CLASS,
        "gradcam_wrong_per_class": GRADCAM_WRONG_PER_CLASS,
        "classes": list(CLASSES),
    }

    with open(
        OBJECTIVE3_DIR / "objective3_configuration.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 70)
    print("Step 03 Finished")
    print("=" * 70)
    print("All results saved to:")
    print(OBJECTIVE3_DIR)


if __name__ == "__main__":
    main()
