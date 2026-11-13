"""
Demo for Phase 2: visualizes every preprocessing step for sample images
and sanity-checks the pipeline functions.

Saves the figure to outputs/preprocessing_demo.png and prints shape /
dtype / value-range checks for each stage.

Run from the project root:
    python -m src.demo_preprocessing
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import CLASS_NAMES, IMG_SIZE, OUTPUTS_DIR, TEST_DIR, TRAIN_DIR
from src.preprocessing import (
    apply_clahe,
    gaussian_blur,
    normalize_for_model,
    preprocess_for_model,
    replicate_to_3_channels,
    resize_image,
    threshold_otsu,
    to_grayscale,
)

SAMPLE_CLASSES = ["glioma", "pituitary"]   # one tumor type + one more


def pick_sample(class_name: str) -> Path:
    """Deterministically pick the first image (alphabetical) of a class."""
    class_dir = TRAIN_DIR / class_name
    return sorted(class_dir.iterdir())[0]


def run_and_collect(image_path: Path) -> list[tuple[str, np.ndarray]]:
    """Apply every step to one image, return (title, display_image) pairs."""
    bgr = cv2.imread(str(image_path))
    assert bgr is not None, f"Could not read {image_path}"

    steps: list[tuple[str, np.ndarray]] = []

    # 1. Original, as OpenCV loaded it (BGR -> shown as RGB)
    steps.append(("original (BGR, shown as RGB)", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    # 2. Resize to the model input size
    resized = resize_image(bgr, IMG_SIZE)
    steps.append((f"resized {IMG_SIZE}x{IMG_SIZE}",
                  cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))

    # 3. Gaussian blur (noise reduction)
    blurred = gaussian_blur(resized)
    steps.append(("gaussian blur k=5", cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)))

    # 4. Grayscale
    gray = to_grayscale(blurred)
    steps.append(("grayscale", gray))

    # 5. CLAHE contrast enhancement
    enhanced = apply_clahe(gray)
    steps.append(("CLAHE", enhanced))

    # 6. Otsu threshold on the enhanced image (demo only!)
    steps.append(("otsu threshold (demo only)", threshold_otsu(enhanced)))

    # 7. Final model input: replicate to 3 channels + ImageNet normalization
    model_input = normalize_for_model(replicate_to_3_channels(enhanced))
    steps.append((f"model input (normalized)\nshape {model_input.shape}", model_input[:, :, 0]))

    return steps


def check_pipelines():
    """Verify preprocess_for_model matches the manual step composition."""
    img = cv2.imread(str(pick_sample(SAMPLE_CLASSES[0])))

    for pipeline in ["raw", "clahe", "blur_clahe"]:
        out = preprocess_for_model(img, pipeline)
        assert out.shape == (IMG_SIZE, IMG_SIZE, 3), out.shape
        assert out.dtype == np.uint8, out.dtype
        print(f"pipeline '{pipeline}': shape={out.shape} dtype={out.dtype} "
              f"range [{out.min()}, {out.max()}]")

    # The blur_clahe pipeline must equal the manual composition above.
    manual = replicate_to_3_channels(
        apply_clahe(to_grayscale(gaussian_blur(resize_image(img)))))
    assert np.array_equal(preprocess_for_model(img, "blur_clahe"), manual)
    print("pipeline 'blur_clahe' == manual composition: OK")

    # Normalization sanity: output is float32 with values outside 0..1.
    norm = normalize_for_model(preprocess_for_model(img, "clahe"))
    assert norm.dtype == np.float32
    print(f"normalized: dtype={norm.dtype} "
          f"range [{norm.min():.2f}, {norm.max():.2f}] "
          f"mean {norm.mean():.3f}")


def main():
    check_pipelines()
    print()

    # Build the demo figure: rows = sample images, cols = steps.
    n_cols = 7
    fig, axes = plt.subplots(len(SAMPLE_CLASSES), n_cols,
                             figsize=(3 * n_cols, 3.2 * len(SAMPLE_CLASSES)))
    axes = np.atleast_2d(axes)

    for r, cls in enumerate(SAMPLE_CLASSES):
        path = pick_sample(cls)
        print(f"{cls}: {path.name}")
        for c, (title, img) in enumerate(run_and_collect(path)):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=9)
        axes[r, 0].set_ylabel(cls, fontsize=11)

    fig.suptitle("Preprocessing steps (Phase 2 demo)")
    fig.tight_layout()
    out_path = OUTPUTS_DIR / "preprocessing_demo.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDemo figure saved to: {out_path}")


if __name__ == "__main__":
    main()
