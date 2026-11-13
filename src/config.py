"""
Central configuration for the brain tumor project.

Every script imports its settings from here, so paths, class names,
image size, and hyperparameters live in ONE place. If you move the
dataset or change the image size, you only edit this file.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
# PROJECT_ROOT is the folder that contains this src/ directory,
# no matter where the script is started from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"          # raw Kaggle dataset (Training/ + Testing/)
MODELS_DIR = PROJECT_ROOT / "models"     # saved model weights (.pt files)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"   # plots, metrics, heatmaps, summaries

# ----------------------------------------------------------------------
# Dataset / classes
# ----------------------------------------------------------------------
# The 4 classes, in this fixed order. The order matters: the model's
# output index i always corresponds to CLASS_NAMES[i], and the confusion
# matrix rows/columns follow the same order.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# Directory names inside data/raw (Kaggle layout)
TRAIN_DIR = RAW_DATA_DIR / "Training"
TEST_DIR = RAW_DATA_DIR / "Testing"

# Validation split: fraction of the *training* folder held out for
# validation (the Kaggle "Testing" folder is kept as the final test set).
VALIDATION_FRACTION = 0.2

# Preprocessing pipeline used for the model input ("raw" | "clahe" |
# "blur_clahe"). Decided by the Phase 4 ablation (outputs/ablation_table.md):
# raw and blur_clahe tied on val accuracy/macro-F1 (0.9634); raw won the
# tie-break with the lowest val loss (0.1059 vs 0.1174). CLAHE gave no
# measurable benefit with this pretrained backbone.
PIPELINE = "raw"

# ----------------------------------------------------------------------
# Image settings
# ----------------------------------------------------------------------
IMG_SIZE = 224        # ResNet50 expects 224x224 inputs
IMG_CHANNELS = 3      # ResNet50 expects 3 channels

# ImageNet normalization constants (required because we use a backbone
# pretrained on ImageNet). Applied to model inputs, NOT to displayed images.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ----------------------------------------------------------------------
# Training hyperparameters
# ----------------------------------------------------------------------
SEED = 42                  # global random seed for reproducibility

# Stage 1: train only the new head, backbone frozen
HEAD_EPOCHS = 15
HEAD_LR = 1e-3

# Stage 2: unfreeze layer4, fine-tune everything with a lower LR
FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-4

BATCH_SIZE = 32            # 6GB GPU -> keep moderate; reduce if OOM
NUM_WORKERS = 2            # dataloader worker processes
EARLY_STOPPING_PATIENCE = 5   # stop if val loss does not improve for N epochs
WEIGHT_DECAY = 1e-4

# ----------------------------------------------------------------------
# Saved artifacts
# ----------------------------------------------------------------------
BEST_MODEL_PATH = MODELS_DIR / "best_model.pt"
TRAINING_CURVES_PNG = OUTPUTS_DIR / "training_curves.png"
TRAIN_HISTORY_JSON = OUTPUTS_DIR / "training_history.json"
DUPLICATE_REPORT_JSON = OUTPUTS_DIR / "duplicate_files.json"

# ----------------------------------------------------------------------
# Device (CPU vs GPU)
# ----------------------------------------------------------------------
import torch

def get_device() -> torch.device:
    """Return the best available device: CUDA GPU if present, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
