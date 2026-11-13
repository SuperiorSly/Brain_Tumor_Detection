"""
Dataset creation (Phase 3).

Responsibilities:
1. Collect image paths + labels from data/raw/{Training,Testing}/<class>/.
2. EXCLUDE the test images that are pixel-identical to training images
   (found in Phase 1, see outputs/duplicate_files.json). This is the
   duplicate handling promised in Phase 1.
3. Stratified train/validation split from the *Training* folder only.
   The Kaggle "Testing" folder stays untouched as the final held-out
   test set (evaluated only in Phase 5).
4. Light augmentation for the training set ONLY (no aug on val/test).
5. Preprocess images with the pipeline from config (config.PIPELINE)
   and return normalized float tensors for ResNet50.

DATA LEAKAGE NOTE (also in README and evaluation output):
This dataset has no patient IDs, so images from the same patient may be
in both train and test. Our split is therefore NOT patient-level and
reported metrics may be optimistic. This limitation is stated wherever
results are reported.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DUPLICATE_REPORT_JSON,
    NUM_CLASSES,
    NUM_WORKERS,
    PIPELINE,
    TEST_DIR,
    TRAIN_DIR,
    VALIDATION_FRACTION,
)
from src.preprocessing import normalize_for_model, preprocess_for_model

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------------------
# File collection and splitting
# ----------------------------------------------------------------------

def _list_files(class_dir: Path) -> list[Path]:
    """All image files directly inside one class folder (sorted)."""
    if not class_dir.is_dir():
        return []
    return sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_test_exclusions() -> set[str]:
    """
    Load the list of test files that duplicate training images
    (Phase 1 pixel-content check). Returns POSIX-style relative paths
    like 'glioma/Te-gl_0019.jpg' so it works on any OS.
    """
    if not DUPLICATE_REPORT_JSON.exists():
        return set()
    data = json.loads(DUPLICATE_REPORT_JSON.read_text())
    entries = data.get("cross_split_pixel_content", [])
    return {e["test_file"].replace("\\", "/") for e in entries}


def collect_samples(split_dir: Path, is_test: bool = False
                    ) -> list[tuple[Path, int]]:
    """
    Build a list of (file_path, class_index) for one split.

    Args:
        split_dir: Training or Testing folder.
        is_test:   if True, exclude the duplicate files found in Phase 1.

    Returns:
        List of (Path, label) pairs, label = index into CLASS_NAMES.
    """
    excluded = load_test_exclusions() if is_test else set()
    samples = []
    n_excluded = 0
    for label, cls in enumerate(CLASS_NAMES):
        for path in _list_files(split_dir / cls):
            rel = path.relative_to(split_dir).as_posix()
            if is_test and rel in excluded:
                n_excluded += 1
                continue
            samples.append((path, label))
    if is_test and excluded:
        print(f"[dataset] excluded {n_excluded} test images that duplicate "
              f"training content (Phase 1 duplicate check)")
    return samples


def stratified_split(samples: list[tuple[Path, int]],
                     val_fraction: float = VALIDATION_FRACTION,
                     seed: int = 42) -> tuple[list, list]:
    """
    Split samples into train/val keeping the class proportions.

    Stratified means: shuffle the files of EACH CLASS separately and move
    the first val_fraction share of every class into validation. This
    keeps the class balance identical in both subsets.

    Returns:
        (train_samples, val_samples)
    """
    by_class: dict[int, list] = defaultdict(list)
    for item in samples:
        by_class[item[1]].append(item)

    rng = random.Random(seed)   # dedicated RNG -> independent of global state
    train, val = [], []
    for label in sorted(by_class):
        items = sorted(by_class[label])          # deterministic base order
        rng.shuffle(items)                       # seeded shuffle
        n_val = max(1, round(len(items) * val_fraction))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


# ----------------------------------------------------------------------
# Augmentation (training set only!)
# ----------------------------------------------------------------------

def _augment_image(img_rgb: np.ndarray) -> np.ndarray:
    """
    Light augmentation for MRI images: small rotation, random flips,
    slight brightness/contrast jitter. Only used on the TRAINING set.

    Returns the same uint8 image type, values still 0..255.
    """
    h, w = img_rgb.shape[:2]

    # 1. Small rotation (-10..+10 degrees), borders mirrored so no black gaps.
    angle = random.uniform(-10.0, 10.0)
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_rgb = cv2.warpAffine(img_rgb, matrix, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT101)

    # 2. Flips. Horizontal flips are common for brain MRIs; vertical flips
    #    are applied rarely because an upside-down brain is unrealistic.
    if random.random() < 0.5:
        img_rgb = cv2.flip(img_rgb, 1)   # horizontal
    if random.random() < 0.2:
        img_rgb = cv2.flip(img_rgb, 0)   # vertical

    # 3. Slight brightness/contrast: out = alpha * img + beta
    alpha = random.uniform(0.9, 1.1)   # contrast
    beta = random.uniform(-10.0, 10.0)  # brightness
    img_rgb = np.clip(img_rgb.astype(np.float32) * alpha + beta, 0, 255)
    return img_rgb.astype(np.uint8)


# ----------------------------------------------------------------------
# PyTorch Dataset
# ----------------------------------------------------------------------

class MRIDataset(Dataset):
    """
    MRI classification dataset.

    Args:
        samples: list of (file_path, label) pairs.
        pipeline: preprocessing pipeline name (see src/preprocessing.py).
        augment:  apply light augmentation (TRAIN set only).
        name:     human-readable name for logs.
    """

    def __init__(self, samples, pipeline: str = PIPELINE,
                 augment: bool = False, name: str = "dataset"):
        self.samples = samples
        self.pipeline = pipeline
        self.augment = augment
        self.name = name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        bgr = cv2.imread(str(path))          # OpenCV loads BGR
        if bgr is None:
            raise RuntimeError(f"Unreadable image: {path}")

        rgb = preprocess_for_model(bgr, self.pipeline)   # uint8 RGB 224x224x3
        if self.augment:
            rgb = _augment_image(rgb)

        # HWC uint8 -> normalized float32 -> CHW tensor for PyTorch
        norm = normalize_for_model(rgb)
        tensor = torch.from_numpy(norm).permute(2, 0, 1).contiguous()
        return tensor, label


def _worker_init_fn(worker_id: int):
    """Give each DataLoader worker a deterministic numpy/random seed."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


# ----------------------------------------------------------------------
# One-call dataloader factory
# ----------------------------------------------------------------------

def make_dataloaders(smoke_limit: int | None = None,
                     pipeline: str = PIPELINE) -> dict:
    """
    Build train/val/test DataLoaders + metadata in one call.

    Args:
        smoke_limit: if set (e.g. 2), keep only N images PER CLASS in the
                     train and val sets (pipeline sanity check, NOT for
                     real training).
        pipeline:    preprocessing pipeline name.

    Returns:
        dict with keys: train_loader, val_loader, test_loader,
        class_weights (tensor), counts (dict), pipeline.
    """
    train_samples, val_samples = stratified_split(
        collect_samples(TRAIN_DIR, is_test=False))

    test_samples = collect_samples(TEST_DIR, is_test=True)

    if smoke_limit is not None:
        train_samples = _limit_per_class(train_samples, smoke_limit)
        val_samples = _limit_per_class(val_samples, smoke_limit)
        print(f"[dataset] SMOKE MODE: {smoke_limit} images per class -> "
              f"{len(train_samples)} train / {len(val_samples)} val. "
              f"Metrics from this run are NOT meaningful!")

    counts = {
        "train": _count_per_class(train_samples),
        "val": _count_per_class(val_samples),
        "test": _count_per_class(test_samples),
    }

    train_ds = MRIDataset(train_samples, pipeline, augment=True, name="train")
    val_ds = MRIDataset(val_samples, pipeline, augment=False, name="val")
    test_ds = MRIDataset(test_samples, pipeline, augment=False, name="test")

    # Class weights from the TRAIN labels only. Balanced classes -> all 1.0,
    # but the mechanism is here in case the data changes.
    labels = [label for _, label in train_samples]
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights = class_counts.sum() / (NUM_CLASSES * np.maximum(class_counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32)

    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init_fn,
    )

    return {
        "train_loader": DataLoader(
            train_ds, batch_size=min(BATCH_SIZE, max(len(train_ds), 1)),
            shuffle=True, generator=torch.Generator().manual_seed(42),
            **loader_kwargs),
        "val_loader": DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs),
        "test_loader": DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs),
        "class_weights": class_weights,
        "counts": counts,
        "pipeline": pipeline,
    }


def _limit_per_class(samples: list, limit: int) -> list:
    """Keep at most `limit` images per class (for smoke runs)."""
    per_class: dict[int, list] = defaultdict(list)
    for item in samples:
        per_class[item[1]].append(item)
    limited = []
    for label in sorted(per_class):
        limited.extend(per_class[label][:limit])
    return limited


def _count_per_class(samples: list) -> dict[str, int]:
    counts = defaultdict(int)
    for _, label in samples:
        counts[CLASS_NAMES[label]] += 1
    return dict(counts)
