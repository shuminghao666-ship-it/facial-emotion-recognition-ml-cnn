# ============================================================
# Pipeline Step 01:
# Traditional Machine Learning Baselines for Facial Emotion Recognition
#
# Purpose:
# 1. Lightweight raw pixel baseline with SGD linear SVM
# 2. HOG feature baseline with Logistic Regression, KNN, and Linear SVM
# 3. Validation-based hyperparameter selection
# 4. Class imbalance comparison through class_weight options
# ============================================================

import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

from project_config import (
    CLASSES,
    CLASS_FOLDER_NAMES,
    IMAGE_EXTENSIONS,
    RESULT_DIR,
    TEST_DIR,
    TRAIN_DIR,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


warnings.filterwarnings("ignore")


# ============================================================
# 1. Path Configuration
# ============================================================

OUTPUT_DIR = RESULT_DIR / "step1_result"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = OUTPUT_DIR / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

REPORT_DIR = OUTPUT_DIR / "classification_reports"
REPORT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Experiment Configuration
# ============================================================

RANDOM_STATE = 42
VALID_SIZE = 0.20

IMAGE_SIZE = (48, 48)
SUPPORTED_EXTENSIONS = set(IMAGE_EXTENSIONS) | {".tif", ".tiff"}

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

SELECTION_METRIC = "Macro_F1"


# ============================================================
# 3. Dataset Structure
# ============================================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Training directory does not exist:\n{TRAIN_DIR}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test directory does not exist:\n{TEST_DIR}")

CLASS_NAMES = list(CLASSES)

print("=" * 70)
print("Detected classes:")
for i, cls in enumerate(CLASS_NAMES):
    print(f"{i}: {cls}")
print("=" * 70)


# ============================================================
# 4. Feature extraction
# ============================================================

def read_grayscale_resized(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32) / 255.0


def find_class_dir(split_dir, class_name):
    for folder_name in CLASS_FOLDER_NAMES[class_name]:
        class_dir = split_dir / folder_name
        if class_dir.exists():
            return class_dir
    return split_dir / class_name


def extract_hog_feature(image_array):
    return hog(
        image_array,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        feature_vector=True,
    ).astype(np.float32)


def extract_raw_pixel_feature(image_array):
    return image_array.reshape(-1).astype(np.float32)


def load_dataset_features(data_dir, class_names):
    hog_features = []
    raw_features = []
    labels = []
    file_paths = []
    skipped_files = []

    print(f"\nLoading images and extracting raw pixel + HOG features: {data_dir}")

    for label_id, class_name in enumerate(class_names):
        class_dir = find_class_dir(data_dir, class_name)

        if not class_dir.exists():
            print(f"Warning: class folder does not exist, skipped: {class_dir}")
            continue

        image_files = sorted(
            file
            for file in class_dir.iterdir()
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        print(f"{class_name:<12} | Image count: {len(image_files)}")

        for image_path in tqdm(image_files, desc=f"Processing {class_name}", leave=False):
            try:
                image_array = read_grayscale_resized(image_path)
                raw_features.append(extract_raw_pixel_feature(image_array))
                hog_features.append(extract_hog_feature(image_array))
                labels.append(label_id)
                file_paths.append(str(image_path))
            except Exception as exc:
                skipped_files.append((str(image_path), str(exc)))

    X_hog = np.asarray(hog_features, dtype=np.float32)
    X_raw = np.asarray(raw_features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    print("\nFeature extraction completed.")
    print(f"Total samples: {len(y)}")
    print(f"Raw pixel feature dimension: {X_raw.shape[1]}")
    print(f"HOG feature dimension: {X_hog.shape[1]}")
    print(f"Skipped corrupted or unreadable files: {len(skipped_files)}")

    if skipped_files:
        skipped_df = pd.DataFrame(skipped_files, columns=["file_path", "error"])
        skipped_df.to_csv(
            OUTPUT_DIR / f"skipped_files_{data_dir.name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return X_hog, X_raw, y, file_paths


def show_class_distribution(y, class_names, dataset_name):
    counts = pd.Series(y).value_counts().sort_index()
    total = int(counts.sum())
    distribution_df = pd.DataFrame(
        {
            "Class_ID": list(range(len(class_names))),
            "Class_Name": class_names,
            "Count": [counts.get(i, 0) for i in range(len(class_names))],
        }
    )
    distribution_df["Percentage"] = (
        distribution_df["Count"] / max(1, total) * 100
    ).round(2)

    print(f"\n{dataset_name} class distribution:")
    print(distribution_df)

    distribution_df.to_csv(
        OUTPUT_DIR / f"{dataset_name.lower().replace(' ', '_')}_class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(8, 5))
    plt.bar(distribution_df["Class_Name"], distribution_df["Count"])
    plt.title(f"{dataset_name} Class Distribution")
    plt.xlabel("Emotion Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / f"{dataset_name.lower().replace(' ', '_')}_class_distribution.png",
        dpi=300,
    )
    plt.close()

    return distribution_df


# ============================================================
# 5. Model utilities
# ============================================================

def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro_Recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted_Precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Weighted_Recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Weighted_F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def evaluate_model(model_name, feature_type, model, X_data, y_true, class_names, dataset_tag):
    print("\n" + "=" * 70)
    print(f"Evaluating {feature_type} + {model_name} on {dataset_tag}")
    print("=" * 70)

    y_pred = model.predict(X_data)
    metric_values = compute_metrics(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print(f"Accuracy           : {metric_values['Accuracy']:.4f}")
    print(f"Macro Precision    : {metric_values['Macro_Precision']:.4f}")
    print(f"Macro Recall       : {metric_values['Macro_Recall']:.4f}")
    print(f"Macro F1-score     : {metric_values['Macro_F1']:.4f}")
    print(f"Weighted F1-score  : {metric_values['Weighted_F1']:.4f}")
    print("\nClassification Report:")
    print(report)

    safe_name = f"{feature_type}_{model_name}_{dataset_tag}"

    report_path = REPORT_DIR / f"{safe_name}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    disp.plot(
        cmap="Blues",
        values_format="d",
        xticks_rotation=45,
        ax=ax,
        colorbar=False,
    )
    plt.title(f"{feature_type} + {model_name} - {dataset_tag} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{safe_name}_confusion_matrix.png", dpi=300)
    plt.close()

    metrics = {
        "Feature_Type": feature_type,
        "Model": model_name,
        "Dataset": dataset_tag,
        **metric_values,
    }

    return metrics, y_pred


def make_model(model_name, params):
    if model_name == "Logistic_Regression":
        return LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )

    if model_name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            n_jobs=-1,
        )

    if model_name == "SGD_Linear_SVM":
        return SGDClassifier(
            loss="hinge",
            alpha=params["alpha"],
            class_weight=params["class_weight"],
            max_iter=1000,
            tol=1e-3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def get_search_space(feature_type):
    if feature_type == "Raw_Pixel":
        return {
            "SGD_Linear_SVM": [
                {"alpha": 0.0001, "class_weight": None},
            ],
        }

    return {
        "Logistic_Regression": [
            {"C": 0.1, "class_weight": None},
            {"C": 1.0, "class_weight": None},
            {"C": 0.1, "class_weight": "balanced"},
            {"C": 1.0, "class_weight": "balanced"},
        ],
        "SGD_Linear_SVM": [
            {"alpha": 0.001, "class_weight": None},
            {"alpha": 0.0001, "class_weight": None},
            {"alpha": 0.001, "class_weight": "balanced"},
            {"alpha": 0.0001, "class_weight": "balanced"},
        ],
        "KNN": [
            {"n_neighbors": 3, "weights": "uniform"},
            {"n_neighbors": 5, "weights": "uniform"},
            {"n_neighbors": 7, "weights": "uniform"},
            {"n_neighbors": 3, "weights": "distance"},
            {"n_neighbors": 5, "weights": "distance"},
            {"n_neighbors": 7, "weights": "distance"},
        ],
    }


def tune_and_select_models(feature_type, X_train, y_train, X_val, y_val):
    tuning_records = []
    selected_models = {}
    selected_params = {}
    selected_train_times = {}

    search_space = get_search_space(feature_type)

    for model_name, param_grid in search_space.items():
        best_record = None
        best_model = None

        for params in param_grid:
            print("\n" + "=" * 70)
            print(f"Tuning {feature_type} + {model_name}")
            print(f"Parameters: {params}")
            print("=" * 70)

            model = make_model(model_name, params)
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            y_val_pred = model.predict(X_val)
            metrics = compute_metrics(y_val, y_val_pred)

            record = {
                "Feature_Type": feature_type,
                "Model": model_name,
                "Parameters": str(params),
                "Training_Time_Seconds": training_time,
                **metrics,
            }
            tuning_records.append(record)
            pd.DataFrame(tuning_records).to_csv(
                OUTPUT_DIR / "validation_tuning_results_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

            print(f"Validation Accuracy: {metrics['Accuracy']:.4f}")
            print(f"Validation Macro F1: {metrics['Macro_F1']:.4f}")
            print(f"Training time: {training_time:.2f} seconds")

            if best_record is None or metrics[SELECTION_METRIC] > best_record[SELECTION_METRIC]:
                best_record = record
                best_model = model

        selected_models[model_name] = best_model
        selected_params[model_name] = best_record["Parameters"]
        selected_train_times[model_name] = best_record["Training_Time_Seconds"]

        print("\nSelected best validation setting:")
        print(f"{feature_type} + {model_name}")
        print(f"Parameters: {best_record['Parameters']}")
        print(f"Validation Macro F1: {best_record['Macro_F1']:.4f}")

    return selected_models, selected_params, selected_train_times, tuning_records


def plot_model_comparison(results_df, dataset_tag, metric_name="Accuracy"):
    subset = results_df[results_df["Dataset"] == dataset_tag].copy()
    subset["Label"] = subset["Feature_Type"] + "\n" + subset["Model"]

    plt.figure(figsize=(11, 5))
    plt.bar(subset["Label"], subset[metric_name])
    plt.title(f"{metric_name} Comparison on {dataset_tag}")
    plt.xlabel("Feature + Model")
    plt.ylabel(metric_name)
    plt.ylim(0, 1)
    plt.xticks(rotation=35, ha="right")

    for idx, value in enumerate(subset[metric_name]):
        plt.text(idx, value + 0.01, f"{value:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{dataset_tag}_{metric_name}_comparison.png", dpi=300)
    plt.close()


# ============================================================
# 6. Load data and create train / validation split
# ============================================================

X_train_hog_full, X_train_raw_full, y_train_full, train_file_paths = load_dataset_features(
    TRAIN_DIR,
    CLASS_NAMES,
)

X_test_hog, X_test_raw, y_test, test_file_paths = load_dataset_features(
    TEST_DIR,
    CLASS_NAMES,
)

show_class_distribution(y_train_full, CLASS_NAMES, "Original Train Set")
show_class_distribution(y_test, CLASS_NAMES, "Original Test Set")

indices = np.arange(len(y_train_full))
train_indices, val_indices = train_test_split(
    indices,
    test_size=VALID_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train_full,
)

feature_sets = {
    "HOG": {
        "train_full": X_train_hog_full,
        "test": X_test_hog,
    },
    "Raw_Pixel": {
        "train_full": X_train_raw_full,
        "test": X_test_raw,
    },
}

for feature_data in feature_sets.values():
    feature_data["train"] = feature_data["train_full"][train_indices]
    feature_data["val"] = feature_data["train_full"][val_indices]

y_train = y_train_full[train_indices]
y_val = y_train_full[val_indices]

print("\n" + "=" * 70)
print("Data split completed")
print("=" * 70)
print(f"Training set size   : {len(y_train)}")
print(f"Validation set size : {len(y_val)}")
print(f"Test set size       : {len(y_test)}")
print(f"HOG feature dim     : {feature_sets['HOG']['train'].shape[1]}")
print(f"Raw pixel dim       : {feature_sets['Raw_Pixel']['train'].shape[1]}")

show_class_distribution(y_train, CLASS_NAMES, "Training Split")
show_class_distribution(y_val, CLASS_NAMES, "Validation Split")


# ============================================================
# 7. Standardize features
# ============================================================

for feature_type, feature_data in feature_sets.items():
    print(f"\nStandardizing {feature_type} features...")
    scaler = StandardScaler()
    feature_data["train_scaled"] = scaler.fit_transform(feature_data["train"]).astype(
        np.float32
    )
    feature_data["val_scaled"] = scaler.transform(feature_data["val"]).astype(np.float32)
    feature_data["test_scaled"] = scaler.transform(feature_data["test"]).astype(np.float32)
    joblib.dump(scaler, MODEL_DIR / f"scaler_{feature_type}.pkl")
    print(f"{feature_type} feature standardization completed.")


# ============================================================
# 8. Tune, select, and save models
# ============================================================

all_tuning_records = []
selected_model_records = []

for feature_type, feature_data in feature_sets.items():
    selected_models, selected_params, selected_train_times, tuning_records = (
        tune_and_select_models(
            feature_type=feature_type,
            X_train=feature_data["train_scaled"],
            y_train=y_train,
            X_val=feature_data["val_scaled"],
            y_val=y_val,
        )
    )

    all_tuning_records.extend(tuning_records)
    feature_data["selected_models"] = selected_models
    feature_data["selected_params"] = selected_params

    for model_name, model in selected_models.items():
        joblib.dump(model, MODEL_DIR / f"{feature_type}_{model_name}.pkl")
        selected_model_records.append(
            {
                "Feature_Type": feature_type,
                "Model": model_name,
                "Best_Parameters": selected_params[model_name],
                "Training_Time_Seconds": selected_train_times[model_name],
            }
        )

tuning_df = pd.DataFrame(all_tuning_records)
tuning_df.to_csv(
    OUTPUT_DIR / "validation_tuning_results.csv",
    index=False,
    encoding="utf-8-sig",
)

selected_models_df = pd.DataFrame(selected_model_records)
selected_models_df.to_csv(
    OUTPUT_DIR / "selected_model_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 9. Evaluate selected models on validation and test sets
# ============================================================

all_metrics = []
validation_predictions = {}
test_predictions = {}

for feature_type, feature_data in feature_sets.items():
    for model_name, model in feature_data["selected_models"].items():
        val_metrics, y_pred_val = evaluate_model(
            model_name=model_name,
            feature_type=feature_type,
            model=model,
            X_data=feature_data["val_scaled"],
            y_true=y_val,
            class_names=CLASS_NAMES,
            dataset_tag="Validation",
        )
        all_metrics.append(val_metrics)
        validation_predictions[f"{feature_type}_{model_name}"] = y_pred_val

        test_metrics, y_pred_test = evaluate_model(
            model_name=model_name,
            feature_type=feature_type,
            model=model,
            X_data=feature_data["test_scaled"],
            y_true=y_test,
            class_names=CLASS_NAMES,
            dataset_tag="Test",
        )
        all_metrics.append(test_metrics)
        test_predictions[f"{feature_type}_{model_name}"] = y_pred_test


# ============================================================
# 10. Save result tables and figures
# ============================================================

results_df = pd.DataFrame(all_metrics)
results_df = results_df.merge(
    selected_models_df[["Feature_Type", "Model", "Best_Parameters"]],
    on=["Feature_Type", "Model"],
    how="left",
)

results_df.to_csv(
    OUTPUT_DIR / "objective1_traditional_ml_metrics_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

validation_results_df = results_df[results_df["Dataset"] == "Validation"].copy()
test_results_df = results_df[results_df["Dataset"] == "Test"].copy()

validation_results_df.to_csv(
    OUTPUT_DIR / "validation_metrics_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

test_results_df.to_csv(
    OUTPUT_DIR / "test_metrics_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 70)
print("Validation metrics summary")
print("=" * 70)
print(validation_results_df.round(4))

print("\n" + "=" * 70)
print("Test metrics summary")
print("=" * 70)
print(test_results_df.round(4))

plot_model_comparison(results_df, "Validation", "Accuracy")
plot_model_comparison(results_df, "Validation", "Macro_F1")
plot_model_comparison(results_df, "Test", "Accuracy")
plot_model_comparison(results_df, "Test", "Macro_F1")


# ============================================================
# 11. Save Experiment Configuration
# ============================================================

config_text = f"""
Pipeline Step 01: Traditional Machine Learning Baselines for FER

Train Directory:
{TRAIN_DIR}

Test Directory:
{TEST_DIR}

Image Size:
{IMAGE_SIZE}

Train / Validation Split:
{int((1 - VALID_SIZE) * 100)}% / {int(VALID_SIZE * 100)}%

Random State:
{RANDOM_STATE}

Feature Types:
1. Raw pixel feature: flattened 48x48 grayscale pixels, used as a lightweight baseline only
2. HOG feature: hand-crafted gradient orientation descriptor, used as the main traditional ML feature

HOG Parameters:
- orientations = {HOG_ORIENTATIONS}
- pixels_per_cell = {HOG_PIXELS_PER_CELL}
- cells_per_block = {HOG_CELLS_PER_BLOCK}
- block_norm = {HOG_BLOCK_NORM}

Models:
1. Raw Pixel + SGD Linear SVM baseline
2. HOG + Logistic Regression
3. HOG + KNN
4. HOG + SGD Linear SVM

Model Selection:
Validation-based hyperparameter tuning using {SELECTION_METRIC}.
Raw Pixel is kept as a lightweight reference baseline, while HOG models receive the main tuning effort.
The independent test set is used only for final evaluation.

Output Folder:
{OUTPUT_DIR.resolve()}
"""

with open(OUTPUT_DIR / "experiment_configuration.txt", "w", encoding="utf-8") as file:
    file.write(config_text)

print("\n" + "=" * 70)
print("Pipeline Step 01 completed.")
print("=" * 70)
print(f"All results have been saved to:\n{OUTPUT_DIR.resolve()}")
