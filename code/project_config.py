from pathlib import Path

# ============================================================
# 1. Project Root
# ============================================================
# Root directory of this project. The source files live in code/, so the
# repository root is one level above this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# 2. Dataset Paths
# ============================================================
# Dataset directory. It should contain train/ and validation/ subfolders.
# The validation split is used as the held-out evaluation split.
DATASET_DIR = PROJECT_ROOT / "data"

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "validation"

# ============================================================
# 3. Output Paths
# ============================================================
# Model output directory for Step 02 and Step 03.
MODEL_DIR = PROJECT_ROOT / "models"

# Result output directory for all experiment steps.
RESULT_DIR = PROJECT_ROOT / "results"

# ============================================================
# 4. Class Settings
# ============================================================
CLASSES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

# Compatible folder names in the dataset.
CLASS_FOLDER_NAMES = {
    "Angry": ["Angry", "angry"],
    "Disgust": ["Disgust", "disgust", "disgusted", "Disgusted"],
    "Fear": ["Fear", "fear", "fearful", "Fearful"],
    "Happy": ["Happy", "happy"],
    "Neutral": ["Neutral", "neutral"],
    "Sad": ["Sad", "sad"],
    "Surprise": ["Surprise", "Surprised", "surprise", "surprised"],
}

# ============================================================
# 5. Image Settings
# ============================================================
IMG_SIZE = 48

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".pgm",
]
