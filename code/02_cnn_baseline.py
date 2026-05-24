# ============================================================
# Pipeline Step 02:
# CNN Baseline and CLAHE Comparison for Facial Emotion Recognition
#
# Purpose:
# 1. Train CNN models with and without CLAHE preprocessing.
# 2. Use class-weighted loss for class imbalance.
# 3. Select the best checkpoint by validation Macro F1.
# 4. Save held-out evaluation metrics, class-wise reports, confusion matrices, and curves.
#
# Efficiency Notes:
# 1. Images are loaded and resized into RAM once.
# 2. CLAHE is applied once per experiment instead of every epoch.
# 3. CUDA AMP and pin_memory are enabled when CUDA is available.
# 4. Plot windows are disabled; figures are saved directly.
# ============================================================

import random
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
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
# 0. Hyperparameters and Basic Settings
# ============================================================

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
RANDOM_STATE = 42

MIN_CLASS_WEIGHT = 0.5
MAX_CLASS_WEIGHT = 3.0

# Compare CLAHE and non-CLAHE settings.
RUN_CLAHE_COMPARISON = True

# Save figures without opening plot windows.
SHOW_PLOTS = False

# num_workers=0 is stable for this in-memory dataset.
NUM_WORKERS = 0

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OBJECTIVE2_DIR = RESULT_DIR / "step2_result"
FIGURE_DIR = OBJECTIVE2_DIR / "figures"
REPORT_DIR = OBJECTIVE2_DIR / "classification_reports"
METRIC_DIR = OBJECTIVE2_DIR / "metrics"

for output_dir in [OBJECTIVE2_DIR, FIGURE_DIR, REPORT_DIR, METRIC_DIR]:
    output_dir.mkdir(parents=True, exist_ok=True)

device = get_device()
print("Using device:", device)

USE_CUDA = device.type == "cuda"
USE_AMP = USE_CUDA
PIN_MEMORY = USE_CUDA

if USE_CUDA:
    torch.backends.cudnn.benchmark = True

print("Use CUDA:", USE_CUDA)
print("Use AMP mixed precision:", USE_AMP)
print("Pin memory:", PIN_MEMORY)
print("Batch size:", BATCH_SIZE)


# ============================================================
# 1. Basic path checking
# ============================================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"TRAIN_DIR does not exist:\n{TRAIN_DIR}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Held-out evaluation directory does not exist:\n{TEST_DIR}")

print("\nTRAIN_DIR:", TRAIN_DIR)
print("EVAL_DIR :", TEST_DIR)
print("MODEL_DIR:", MODEL_DIR)
print("RESULT_DIR:", RESULT_DIR)
print("OBJECTIVE2_DIR:", OBJECTIVE2_DIR)


# ============================================================
# 2. Robust Image Reading
# ============================================================

def read_gray_image_unicode(image_path):
    """
    Read grayscale images from non-ASCII file paths.

    cv2.imread may fail for some non-ASCII paths, so this function uses:
        np.fromfile + cv2.imdecode
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
# 3. Dataset path collection
# ============================================================

def find_class_dir(split_dir, class_name):
    for folder_name in CLASS_FOLDER_NAMES[class_name]:
        class_dir = split_dir / folder_name
        if class_dir.exists():
            return class_dir

    return split_dir / class_name


def collect_image_paths(split_dir):
    image_paths = []
    labels = []

    print(f"\nCollecting image paths from: {split_dir}")

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
# 4. Preload all images into memory
# ============================================================

def preload_images_to_memory(image_paths, split_name):
    """
    Read and resize every image only once, then store in RAM.

    Output shape:
        (N, IMG_SIZE, IMG_SIZE)
    dtype:
        uint8
    """
    print("\n" + "=" * 70)
    print(f"Preloading {split_name} images into RAM")
    print("=" * 70)

    total = len(image_paths)
    images = np.empty((total, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    failed_paths = []

    start_time = time.time()

    for i, image_path in enumerate(image_paths):
        img = read_gray_image_unicode(image_path)

        if img is None:
            failed_paths.append(image_path)
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        else:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        images[i] = img

        if (i + 1) % 2000 == 0 or (i + 1) == total:
            print(f"{split_name}: {i + 1}/{total} images loaded")

    elapsed = time.time() - start_time

    print(f"{split_name} preload completed.")
    print(f"Shape: {images.shape}")
    print(f"Time : {elapsed:.2f} seconds")
    print(f"Failed images: {len(failed_paths)}")

    if failed_paths:
        failed_file = OBJECTIVE2_DIR / f"failed_images_{split_name.lower()}.txt"
        with open(failed_file, "w", encoding="utf-8") as f:
            for path in failed_paths:
                f.write(path + "\n")
        print("Failed image paths saved to:", failed_file)

    return images


# ============================================================
# 5. Apply CLAHE once per experiment
# ============================================================

def apply_clahe_to_image_array(images, split_name):
    """
    Apply CLAHE to all images only once before training.
    """
    print("\n" + "=" * 70)
    print(f"Applying CLAHE to {split_name} images")
    print("=" * 70)

    processed = np.empty_like(images)

    total = len(images)
    start_time = time.time()

    for i in range(total):
        processed[i] = apply_clahe(images[i])

        if (i + 1) % 2000 == 0 or (i + 1) == total:
            print(f"{split_name}: {i + 1}/{total} images processed")

    elapsed = time.time() - start_time

    print(f"CLAHE preprocessing finished for {split_name}.")
    print(f"Time: {elapsed:.2f} seconds")

    return processed


# ============================================================
# 6. In-memory dataset
# ============================================================

class InMemoryEmotionDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def random_augment(self, img):
        """
        Moderate data augmentation for FER images.
        """
        if random.random() < 0.5:
            img = cv2.flip(img, 1)

        if random.random() < 0.4:
            angle = random.uniform(-10, 10)
            h, w = img.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(
                img,
                matrix,
                (w, h),
                borderMode=cv2.BORDER_REFLECT,
            )

        if random.random() < 0.3:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-5, 5)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        if random.random() < 0.2:
            h, w = img.shape
            tx = random.randint(-2, 2)
            ty = random.randint(-2, 2)
            matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(
                img,
                matrix,
                (w, h),
                borderMode=cv2.BORDER_REFLECT,
            )

        if random.random() < 0.1:
            noise = np.random.normal(0, 3, img.shape).astype(np.float32)
            img = img.astype(np.float32) + noise
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def __getitem__(self, index):
        img = self.images[index]
        label = int(self.labels[index])

        if self.augment:
            img = img.copy()
            img = self.random_augment(img)

        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        img_tensor = torch.from_numpy(img).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor


# ============================================================
# 7. Class weight calculation
# ============================================================

def calculate_class_weights(labels):
    class_counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float32)
    total = len(labels)

    safe_class_counts = np.maximum(class_counts, 1.0)

    raw_weights = total / (len(CLASSES) * safe_class_counts)
    softened_weights = np.sqrt(raw_weights)
    clipped_weights = np.clip(
        softened_weights,
        MIN_CLASS_WEIGHT,
        MAX_CLASS_WEIGHT,
    )

    weights = torch.tensor(clipped_weights, dtype=torch.float32)

    print("\nClass counts:")
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name}: {int(class_counts[i])}")

    print("\nRaw class weights:")
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name}: {raw_weights[i]:.4f}")

    print("\nSoftened and clipped class weights:")
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name}: {weights[i].item():.4f}")

    return weights


# ============================================================
# 8. DataLoader creation
# ============================================================

def create_dataloaders(
    train_images,
    val_images,
    test_images,
    train_labels,
    val_labels,
    test_labels,
):
    train_dataset = InMemoryEmotionDataset(
        train_images,
        train_labels,
        augment=True,
    )

    val_dataset = InMemoryEmotionDataset(
        val_images,
        val_labels,
        augment=False,
    )

    test_dataset = InMemoryEmotionDataset(
        test_images,
        test_labels,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 9. Train / Evaluate functions with AMP support
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, scaler):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=PIN_MEMORY)
        labels = labels.to(device, non_blocking=PIN_MEMORY)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=PIN_MEMORY)
            labels = labels.to(device, non_blocking=PIN_MEMORY)

            if USE_AMP:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)


# ============================================================
# 10. Visualization functions
# ============================================================

def plot_training_curves(
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    experiment_name,
):
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="Training Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Training and Validation Accuracy ({experiment_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"accuracy_curve_{experiment_name}.png", dpi=300)

    if SHOW_PLOTS:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss ({experiment_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"loss_curve_{experiment_name}.png", dpi=300)

    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_confusion_matrix(cm, experiment_name, normalize=False):
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        display_cm = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=np.float64),
            where=row_sums != 0,
        )
        value_format = ".2f"
        title_suffix = "Normalized"
        file_suffix = "normalized"
    else:
        display_cm = cm
        value_format = "d"
        title_suffix = "Count"
        file_suffix = "count"

    plt.figure(figsize=(8, 8))
    plt.imshow(display_cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {title_suffix} ({experiment_name})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(np.arange(len(CLASSES)), CLASSES, rotation=45)
    plt.yticks(np.arange(len(CLASSES)), CLASSES)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(
                j,
                i,
                format(display_cm[i, j], value_format),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / f"confusion_matrix_{experiment_name}_{file_suffix}.png",
        dpi=300,
    )

    if SHOW_PLOTS:
        plt.show()
    plt.close()


def save_classification_outputs(experiment_name, true_labels, pred_labels, cm):
    report_text = classification_report(
        true_labels,
        pred_labels,
        target_names=CLASSES,
        zero_division=0,
    )
    report_dict = classification_report(
        true_labels,
        pred_labels,
        target_names=CLASSES,
        zero_division=0,
        output_dict=True,
    )

    with open(
        REPORT_DIR / f"classification_report_{experiment_name}.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report_text)

    classwise_rows = []
    for class_name in CLASSES:
        class_metrics = report_dict[class_name]
        classwise_rows.append(
            {
                "Experiment": experiment_name,
                "Class": class_name,
                "Precision": class_metrics["precision"],
                "Recall": class_metrics["recall"],
                "F1": class_metrics["f1-score"],
                "Support": class_metrics["support"],
            }
        )

    classwise_df = pd.DataFrame(classwise_rows)
    classwise_df.to_csv(
        METRIC_DIR / f"classwise_metrics_{experiment_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    cm_df.to_csv(
        METRIC_DIR / f"confusion_matrix_{experiment_name}_count.csv",
        encoding="utf-8-sig",
    )

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0,
    )
    cm_normalized_df = pd.DataFrame(cm_normalized, index=CLASSES, columns=CLASSES)
    cm_normalized_df.to_csv(
        METRIC_DIR / f"confusion_matrix_{experiment_name}_normalized.csv",
        encoding="utf-8-sig",
    )

    summary_metrics = {
        "Experiment": experiment_name,
        "Accuracy": report_dict["accuracy"],
        "Macro_Precision": report_dict["macro avg"]["precision"],
        "Macro_Recall": report_dict["macro avg"]["recall"],
        "Macro_F1": report_dict["macro avg"]["f1-score"],
        "Weighted_Precision": report_dict["weighted avg"]["precision"],
        "Weighted_Recall": report_dict["weighted avg"]["recall"],
        "Weighted_F1": report_dict["weighted avg"]["f1-score"],
    }

    return report_text, summary_metrics


# ============================================================
# 11. Single experiment execution
# ============================================================

def run_experiment(
    experiment_name,
    use_clahe,
    raw_train_images,
    raw_val_images,
    raw_test_images,
    train_labels,
    val_labels,
    test_labels,
):
    print("\n" + "=" * 70)
    print("Experiment:", experiment_name)
    print("Use CLAHE:", use_clahe)
    print("=" * 70)

    experiment_start_time = time.time()

    # --------------------------------------------------------
    # Apply CLAHE only once per experiment
    # --------------------------------------------------------
    if use_clahe:
        train_images = apply_clahe_to_image_array(raw_train_images, "Train")
        val_images = apply_clahe_to_image_array(raw_val_images, "Validation")
        test_images = apply_clahe_to_image_array(raw_test_images, "Held-out Evaluation")
    else:
        train_images = raw_train_images
        val_images = raw_val_images
        test_images = raw_test_images

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------
    train_loader, val_loader, test_loader = create_dataloaders(
        train_images,
        val_images,
        test_images,
        train_labels,
        val_labels,
        test_labels,
    )

    # --------------------------------------------------------
    # Model / loss / optimizer
    # --------------------------------------------------------
    class_weights = calculate_class_weights(train_labels).to(device)

    model = EmotionCNN(num_classes=len(CLASSES)).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_val_macro_f1 = 0.0

    best_model_path = MODEL_DIR / f"best_emotion_cnn_pytorch_{experiment_name}.pth"
    final_model_path = MODEL_DIR / f"final_emotion_cnn_pytorch_{experiment_name}.pth"

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_macro_f1s = []

    print("\nStart training...")

    training_start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
        )

        val_loss, val_acc, val_true_labels, val_pred_labels = evaluate(
            model,
            val_loader,
            criterion,
        )
        val_macro_f1 = f1_score(
            val_true_labels,
            val_pred_labels,
            average="macro",
            zero_division=0,
        )

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_macro_f1s.append(val_macro_f1)

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f} "
            f"Val Macro F1: {val_macro_f1:.4f} "
            f"LR: {current_lr:.6f} "
            f"Time: {epoch_time:.2f}s"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_val_macro_f1 = val_macro_f1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "img_size": IMG_SIZE,
                    "use_clahe": use_clahe,
                    "experiment_name": experiment_name,
                    "selection_metric": "Validation_Macro_F1",
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "best_val_macro_f1": best_val_macro_f1,
                },
                best_model_path,
            )

            print("Best model saved.")

    total_training_time = time.time() - training_start_time

    history_df = pd.DataFrame(
        {
            "Epoch": np.arange(1, len(train_losses) + 1),
            "Train_Loss": train_losses,
            "Validation_Loss": val_losses,
            "Train_Accuracy": train_accs,
            "Validation_Accuracy": val_accs,
            "Validation_Macro_F1": val_macro_f1s,
        }
    )
    history_df.to_csv(
        METRIC_DIR / f"training_history_{experiment_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "img_size": IMG_SIZE,
            "use_clahe": use_clahe,
            "experiment_name": experiment_name,
            "final_val_acc": val_accs[-1],
            "final_val_loss": val_losses[-1],
            "final_val_macro_f1": val_macro_f1s[-1],
        },
        final_model_path,
    )

    print("\nFinal model saved to:", final_model_path)
    print("Best model saved to:", best_model_path)
    print(f"Total training time for {experiment_name}: {total_training_time:.2f} seconds")

    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        experiment_name,
    )

    # --------------------------------------------------------
    # Final held-out evaluation
    # --------------------------------------------------------
    print("\nRunning held-out evaluation for best model...")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, true_labels, pred_labels = evaluate(
        model,
        test_loader,
        criterion,
    )

    print("\nHeld-out Evaluation Loss:", test_loss)
    print("Held-out Evaluation Accuracy:", test_acc)

    cm = confusion_matrix(
        true_labels,
        pred_labels,
        labels=np.arange(len(CLASSES)),
    )

    report, summary_metrics = save_classification_outputs(
        experiment_name,
        true_labels,
        pred_labels,
        cm,
    )

    print("\nClassification Report:")
    print(report)

    plot_confusion_matrix(cm, experiment_name, normalize=False)
    plot_confusion_matrix(cm, experiment_name, normalize=True)

    total_experiment_time = time.time() - experiment_start_time

    result = {
        "experiment_name": experiment_name,
        "use_clahe": use_clahe,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_val_macro_f1": best_val_macro_f1,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "test_macro_precision": summary_metrics["Macro_Precision"],
        "test_macro_recall": summary_metrics["Macro_Recall"],
        "test_macro_f1": summary_metrics["Macro_F1"],
        "test_weighted_f1": summary_metrics["Weighted_F1"],
        "training_time_seconds": total_training_time,
        "total_experiment_time_seconds": total_experiment_time,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
    }

    return result


# ============================================================
# 12. Experiment summary
# ============================================================

def write_experiment_summary(results):
    summary_path = OBJECTIVE2_DIR / "experiment_comparison_summary.txt"

    metrics_df = pd.DataFrame(
        [
            {
                "Experiment": result["experiment_name"],
                "Use_CLAHE": result["use_clahe"],
                "Selection_Metric": "Validation_Macro_F1",
                "Best_Validation_Accuracy": result["best_val_acc"],
                "Best_Validation_Loss": result["best_val_loss"],
                "Best_Validation_Macro_F1": result["best_val_macro_f1"],
                "Test_Accuracy": result["test_acc"],
                "Test_Loss": result["test_loss"],
                "Test_Macro_Precision": result["test_macro_precision"],
                "Test_Macro_Recall": result["test_macro_recall"],
                "Test_Macro_F1": result["test_macro_f1"],
                "Test_Weighted_F1": result["test_weighted_f1"],
                "Training_Time_Seconds": result["training_time_seconds"],
                "Total_Experiment_Time_Seconds": result[
                    "total_experiment_time_seconds"
                ],
                "Best_Model_Path": result["best_model_path"],
                "Final_Model_Path": result["final_model_path"],
            }
            for result in results
        ]
    )
    metrics_df.to_csv(
        METRIC_DIR / "cnn_experiment_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lines = []
    lines.append("Experiment Comparison Summary")
    lines.append("=" * 60)
    lines.append("")

    for result in results:
        lines.append(f"Experiment: {result['experiment_name']}")
        lines.append(f"Use CLAHE: {result['use_clahe']}")
        lines.append("Selection Metric: Validation Macro F1")
        lines.append(f"Best Validation Accuracy: {result['best_val_acc']:.4f}")
        lines.append(f"Best Validation Loss: {result['best_val_loss']:.4f}")
        lines.append(f"Best Validation Macro F1: {result['best_val_macro_f1']:.4f}")
        lines.append(f"Held-out Evaluation Accuracy: {result['test_acc']:.4f}")
        lines.append(f"Held-out Evaluation Loss: {result['test_loss']:.4f}")
        lines.append(f"Held-out Evaluation Macro Precision: {result['test_macro_precision']:.4f}")
        lines.append(f"Held-out Evaluation Macro Recall: {result['test_macro_recall']:.4f}")
        lines.append(f"Held-out Evaluation Macro F1: {result['test_macro_f1']:.4f}")
        lines.append(f"Held-out Evaluation Weighted F1: {result['test_weighted_f1']:.4f}")
        lines.append(f"Training Time: {result['training_time_seconds']:.2f} seconds")
        lines.append(f"Total Experiment Time: {result['total_experiment_time_seconds']:.2f} seconds")
        lines.append(f"Best Model Path: {result['best_model_path']}")
        lines.append("")

    best_result = max(results, key=lambda x: x["best_val_macro_f1"])

    lines.append("Selected Best Experiment")
    lines.append("-" * 60)
    lines.append(f"Experiment: {best_result['experiment_name']}")
    lines.append(f"Use CLAHE: {best_result['use_clahe']}")
    lines.append("Selection Metric: Validation Macro F1")
    lines.append(f"Best Validation Accuracy: {best_result['best_val_acc']:.4f}")
    lines.append(f"Best Validation Macro F1: {best_result['best_val_macro_f1']:.4f}")
    lines.append(f"Held-out Evaluation Accuracy: {best_result['test_acc']:.4f}")
    lines.append(f"Held-out Evaluation Macro F1: {best_result['test_macro_f1']:.4f}")
    lines.append(f"Held-out Evaluation Weighted F1: {best_result['test_weighted_f1']:.4f}")
    lines.append(f"Best Model Path: {best_result['best_model_path']}")
    lines.append("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nExperiment summary saved to:", summary_path)

    return best_result


def save_default_best_model(best_result):
    selected_checkpoint = torch.load(
        best_result["best_model_path"],
        map_location=device,
    )

    default_best_path = MODEL_DIR / "best_emotion_cnn_pytorch.pth"

    torch.save(
        selected_checkpoint,
        default_best_path,
    )

    print("\nDefault best model updated:")
    print(default_best_path)
    print("Selected experiment:", best_result["experiment_name"])


# ============================================================
# 13. Main function
# ============================================================

def main():
    # --------------------------------------------------------
    # 1. Collect image paths
    # --------------------------------------------------------
    print("\nCollecting training images from:", TRAIN_DIR)
    train_all_paths, train_all_labels = collect_image_paths(TRAIN_DIR)

    print("\nCollecting held-out evaluation images from:", TEST_DIR)
    test_paths, test_labels = collect_image_paths(TEST_DIR)

    print("\nTotal training/validation images:", len(train_all_paths))
    print("Total held-out evaluation images:", len(test_paths))

    if len(train_all_paths) == 0 or len(test_paths) == 0:
        print("No images found.")
        return

    # --------------------------------------------------------
    # 2. Train / validation split
    # --------------------------------------------------------
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_all_paths,
        train_all_labels,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=train_all_labels,
    )

    print("\nDataset split:")
    print("Train:", len(train_paths))
    print("Validation:", len(val_paths))
    print("Held-out Evaluation:", len(test_paths))

    # --------------------------------------------------------
    # 3. Preload resized raw images into RAM only once
    # --------------------------------------------------------
    raw_train_images = preload_images_to_memory(train_paths, "Train")
    raw_val_images = preload_images_to_memory(val_paths, "Validation")
    raw_test_images = preload_images_to_memory(test_paths, "Held-out Evaluation")

    # --------------------------------------------------------
    # 4. Define experiments
    # --------------------------------------------------------
    if RUN_CLAHE_COMPARISON:
        experiments = [
            {
                "experiment_name": "with_clahe",
                "use_clahe": True,
            },
            {
                "experiment_name": "without_clahe",
                "use_clahe": False,
            },
        ]
    else:
        experiments = [
            {
                "experiment_name": "with_clahe",
                "use_clahe": True,
            }
        ]

    # --------------------------------------------------------
    # 5. Run experiments
    # --------------------------------------------------------
    results = []

    for experiment in experiments:
        result = run_experiment(
            experiment_name=experiment["experiment_name"],
            use_clahe=experiment["use_clahe"],
            raw_train_images=raw_train_images,
            raw_val_images=raw_val_images,
            raw_test_images=raw_test_images,
            train_labels=train_labels,
            val_labels=val_labels,
            test_labels=test_labels,
        )

        results.append(result)

    # --------------------------------------------------------
    # 6. Save summary and default best model
    # --------------------------------------------------------
    best_result = write_experiment_summary(results)
    save_default_best_model(best_result)

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
