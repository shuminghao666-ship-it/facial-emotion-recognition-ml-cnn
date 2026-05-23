from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from project_config import CLASSES, IMG_SIZE, MODEL_DIR


def get_device():
    """Return the best available PyTorch device for local training or inference."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def apply_clahe(img):
    """Apply CLAHE contrast enhancement to a grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def preprocess_face(face_img, use_clahe=True):
    """Convert one face image into a normalized CNN input tensor."""
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))

    if use_clahe:
        face_img = apply_clahe(face_img)

    face_img = face_img.astype(np.float32) / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=0)
    return torch.tensor(face_img, dtype=torch.float32)


class EmotionCNN(nn.Module):
    """CNN classifier used by Step 02 and Step 03."""

    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.30),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.50),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def load_emotion_model(model_path=None, device=None):
    """Load a trained emotion CNN checkpoint for inference."""
    if device is None:
        device = get_device()

    if model_path is None:
        model_path = MODEL_DIR / "best_emotion_cnn_pytorch.pth"
    else:
        model_path = Path(model_path)

    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get("classes", CLASSES)
    use_clahe = checkpoint.get("use_clahe", True)

    model = EmotionCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.use_clahe = use_clahe

    return model, classes


def predict_emotion(model, face_img, classes=CLASSES, device=None, use_clahe=None):
    """Predict the emotion label and confidence score for one face image."""
    if device is None:
        device = next(model.parameters()).device

    if use_clahe is None:
        use_clahe = getattr(model, "use_clahe", True)

    face_tensor = preprocess_face(face_img, use_clahe=use_clahe).to(device)

    with torch.no_grad():
        outputs = model(face_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, index = torch.max(probs, dim=0)

    return classes[index.item()], confidence.item()
