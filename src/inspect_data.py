"""
Phase 1 data inspection.

Answers four questions about the dataset and writes a report to outputs/:
  1. How many images per class, per split? (class balance)
  2. What are the image sizes? (do they vary? do they need resizing?)
  3. What do the images look like? (sample grid -> outputs/data_sample_grid.png)
  4. Are there duplicate images between the Training and Testing folders?
     (data leakage check)

Two duplicate checks are performed:
  a. MD5 of the raw file bytes  -> exact copies (identical files)
  b. MD5 of the decoded pixels, resized to 64x64 -> same image content even
     if the file was re-encoded/saved (e.g., by Kaggle when packaging)

Run from the project root:
    python -m src.inspect_data
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    CLASS_NAMES,
    OUTPUTS_DIR,
    TEST_DIR,
    TRAIN_DIR,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def list_images(class_dir: Path) -> list[Path]:
    """Return all image files directly inside one class folder."""
    if not class_dir.is_dir():
        return []
    return sorted(p for p in class_dir.iterdir()
                  if p.suffix.lower() in IMAGE_EXTS)


def count_per_class(split_dir: Path) -> dict[str, int]:
    """Count images per class in one split (Training or Testing)."""
    return {cls: len(list_images(split_dir / cls)) for cls in CLASS_NAMES}


def image_size_stats(split_dir: Path) -> dict:
    """Collect (width, height) for every image in the split."""
    sizes = Counter()
    areas = []
    n_unreadable = 0
    for cls in CLASS_NAMES:
        for img_path in list_images(split_dir / cls):
            img = cv2.imread(str(img_path))
            if img is None:  # unreadable / corrupted file
                n_unreadable += 1
                continue
            h, w = img.shape[:2]
            sizes[(w, h)] += 1
            areas.append(w * h)
    if not areas:
        return {"n_read": 0}
    return {
        "n_read": len(areas),
        "n_unreadable": n_unreadable,
        "n_unique_sizes": len(sizes),
        "most_common_sizes_WxH": [
            f"{w}x{h} ({n} images)" for (w, h), n in sizes.most_common(5)
        ],
        "min_WxH": f"{sizes and min(w for (w, h) in sizes)}x"
                   f"{sizes and min(h for (w, h) in sizes)}",
        "max_WxH": f"{max(w for (w, h) in sizes)}x{max(h for (w, h) in sizes)}",
    }


def md5_of_file(path: Path) -> str:
    """MD5 hash of the raw file bytes (identical files -> identical hash)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_signature(path: Path, size: int = 64) -> str | None:
    """
    Content signature of an image: decode as grayscale, resize to a fixed
    small size, then hash the raw pixel bytes. Two images with identical
    content (even after re-encoding as JPEG/PNG) get the same signature.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return hashlib.md5(img.tobytes()).hexdigest()


def build_signature_map(paths: list[Path]) -> dict[str, list[str]]:
    """hash -> list of relative paths having that hash."""
    sig_map: dict[str, list[str]] = {}
    for p in paths:
        sig_map.setdefault(pixel_signature(p), []).append(
            str(p.relative_to(p.parents[1])))
    return sig_map


def find_duplicates(train_dir: Path, test_dir: Path) -> dict:
    """
    Find duplicate images:
      - inside the Training folder (train-train)
      - between Training and Testing folders (leakage!)
    Runs BOTH checks: exact byte copies and pixel-content copies.
    """
    train_paths = [p for cls in CLASS_NAMES for p in list_images(train_dir / cls)]
    test_paths = [p for cls in CLASS_NAMES for p in list_images(test_dir / cls)]

    # --- Check A: exact file duplicates (byte hash) ---
    train_file_hashes: dict[str, list[str]] = {}
    for p in train_paths:
        train_file_hashes.setdefault(md5_of_file(p), []).append(
            str(p.relative_to(train_dir)))
    dup_within_train = [paths for paths in train_file_hashes.values()
                        if len(paths) > 1]
    cross_split_exact = []
    for p in test_paths:
        h = md5_of_file(p)
        if h in train_file_hashes:
            cross_split_exact.append({
                "test_file": str(p.relative_to(test_dir)),
                "train_copies": train_file_hashes[h],
            })

    # --- Check B: pixel-content duplicates (robust to re-encoding) ---
    train_sig = build_signature_map(train_paths)
    cross_split_pixel = []
    for p in test_paths:
        sig = pixel_signature(p)
        if sig in train_sig:
            cross_split_pixel.append({
                "test_file": str(p.relative_to(test_dir)),
                "train_copies": train_sig[sig],
            })

    return {
        "n_train_dupe_groups_exact": len(dup_within_train),
        "train_dupe_groups_examples": dup_within_train[:10],
        "n_cross_split_exact": len(cross_split_exact),
        "cross_split_exact_examples": cross_split_exact[:10],
        "n_cross_split_pixel_content": len(cross_split_pixel),
        "cross_split_pixel_examples": cross_split_pixel[:10],
        "cross_split_pixel_content_full": cross_split_pixel,  # complete list
    }


def save_sample_grid(split_dir: Path, out_path: Path, n_per_class: int = 4):
    """Save a grid: rows = classes, columns = sample images."""
    fig, axes = plt.subplots(len(CLASS_NAMES), n_per_class,
                             figsize=(3 * n_per_class, 3 * len(CLASS_NAMES)))
    axes = np.atleast_2d(axes)
    for r, cls in enumerate(CLASS_NAMES):
        paths = list_images(split_dir / cls)[:n_per_class]
        for c in range(n_per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(paths):
                img = cv2.cvtColor(cv2.imread(str(paths[c])), cv2.COLOR_BGR2RGB)
                ax.imshow(img, cmap="gray")
                if c == 0:
                    ax.set_title(cls, fontsize=11, loc="left")
    fig.suptitle(f"Sample images from {split_dir.name} (rows = classes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    report = {}

    # 1. Image counts per class
    train_counts = count_per_class(TRAIN_DIR)
    test_counts = count_per_class(TEST_DIR)
    report["train_counts"] = train_counts
    report["test_counts"] = test_counts

    train_total = sum(train_counts.values())
    test_total = sum(test_counts.values())
    print("=" * 60)
    print("IMAGES PER CLASS")
    print("=" * 60)
    print(f"{'class':<12}{'train':>8}{'test':>8}")
    for cls in CLASS_NAMES:
        print(f"{cls:<12}{train_counts[cls]:>8}{test_counts[cls]:>8}")
    print(f"{'TOTAL':<12}{train_total:>8}{test_total:>8}")

    # 2. Class imbalance
    if train_total:
        shares = {c: train_counts[c] / train_total for c in CLASS_NAMES}
        report["train_class_share"] = {c: round(s, 4) for c, s in shares.items()}
        report["max_min_ratio"] = round(
            max(shares.values()) / min(shares.values()), 3)
        print("\nCLASS IMBALANCE (train share)")
        for cls in CLASS_NAMES:
            print(f"  {cls:<12}{shares[cls] * 100:6.1f}%")
        print(f"  max/min ratio: {report['max_min_ratio']}")

    # 3. Image sizes
    report["train_sizes"] = image_size_stats(TRAIN_DIR)
    report["test_sizes"] = image_size_stats(TEST_DIR)
    print("\nIMAGE SIZES (train)")
    for k, v in report["train_sizes"].items():
        print(f"  {k}: {v}")

    # 4. Duplicates / leakage check
    print("\nCHECKING DUPLICATES (byte hash + pixel content)...")
    report["duplicates"] = find_duplicates(TRAIN_DIR, TEST_DIR)
    d = report["duplicates"]
    print(f"  train-train exact duplicate groups   : {d['n_train_dupe_groups_exact']}")
    print(f"  train<->test exact duplicates        : {d['n_cross_split_exact']}")
    print(f"  train<->test pixel-content duplicates: {d['n_cross_split_pixel_content']}")
    if d["n_cross_split_pixel_content"]:
        print("  (same image content in BOTH splits -> inflated test accuracy risk)")

    # Full list of cross-split duplicates, so dataset.py can EXCLUDE these
    # test images at load time (a near 7% share of the test set).
    full_dupe_path = OUTPUTS_DIR / "duplicate_files.json"
    full_dupe_path.write_text(json.dumps(
        {"cross_split_pixel_content": d["cross_split_pixel_content_full"]}, indent=2))
    print(f"Full duplicate list saved to: {full_dupe_path}")

    # 5. Sample grid
    grid_path = OUTPUTS_DIR / "data_sample_grid.png"
    save_sample_grid(TRAIN_DIR, grid_path)
    print(f"\nSample grid saved to: {grid_path}")

    # Save machine-readable report (a data summary can be shown in the app later)
    out_json = OUTPUTS_DIR / "data_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"Report saved to: {out_json}")


if __name__ == "__main__":
    main()
