"""
Image preprocessing functions (Phase 2).

Every function is small, standalone, and documented so each step can be
explained and tested separately. OpenCV loads images as BGR (not RGB!),
so the functions state explicitly which format they expect and return.

Two levels are provided:

1. Individual steps (resize, blur, grayscale, CLAHE, threshold,
   normalization) - useful for the demo figure and for learning.
2. Ready-made pipelines for the model (raw / clahe / blur_clahe) -
   used by the Dataset in training and by the ablation study. The
   ablation (Phase 4) decides which pipeline the final app uses.

Pipeline contract (important for Reproducibility):
    preprocess_for_model() takes a BGR uint8 image as loaded by OpenCV
    and returns an RGB uint8 image of shape (IMG_SIZE, IMG_SIZE, 3),
    still in the 0..255 range so it can also be *displayed*.
    normalize_for_model() then turns that into a float32 array with
    ImageNet mean/std normalization, ready for the network.

Note: ResNet50 expects 3 channels. Grayscale results are therefore
replicated to 3 identical channels before normalization.
"""

import cv2
import numpy as np

from src.config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

# Pipelines compared in the ablation study (Phase 4).
PIPELINES = ["raw", "clahe", "blur_clahe"]


# ----------------------------------------------------------------------
# Individual preprocessing steps
# ----------------------------------------------------------------------

def resize_image(image: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """
    Resize an image to a square (size x size).

    ResNet50 requires a fixed input size of 224x224. INTER_AREA is the
    best interpolation for *downscaling* (our images are mostly larger
    than 224, e.g. 512x512).

    Args:
        image: input image, any number of channels (BGR or grayscale).
        size:  target width and height in pixels.

    Returns:
        Resized image, same channel count and dtype as the input.
    """
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def gaussian_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Reduce noise with a Gaussian blur.

    MRI scans contain sensor noise and JPEG artifacts. A small Gaussian
    kernel smooths them; the kernel size must be odd so the pixel grid
    has a defined center.

    Args:
        image: input image (BGR or grayscale, uint8 or float).
        ksize: square kernel size, must be odd (e.g. 3, 5, 7).

    Returns:
        Blurred image, same shape and dtype as the input.
    """
    if ksize % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got {ksize}")
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=0)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to a single-channel grayscale image.

    MRI scans are inherently grayscale; collapsing the (usually
    identical) color channels makes later steps like CLAHE simpler.

    Args:
        image: BGR image of shape (H, W, 3).

    Returns:
        Grayscale image of shape (H, W), uint8.
    """
    if image.ndim == 2:          # already grayscale
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray,
                clip_limit: float = 2.0,
                tile_size: int = 8) -> np.ndarray:
    """
    Enhance local contrast with CLAHE.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) equalizes
    the histogram in small tiles instead of the whole image, and clips
    the amplification to avoid over-boosting noise. This makes tumor
    boundaries easier to see in low-contrast scans.

    Args:
        gray:      grayscale image (H, W), uint8.
        clip_limit: amplification ceiling; 2.0 is a common default.
        tile_size:  side length of the tiles CLAHE operates on.

    Returns:
        Contrast-enhanced grayscale image, same shape as the input.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def replicate_to_3_channels(gray: np.ndarray) -> np.ndarray:
    """
    Replicate a single-channel image to 3 identical channels.

    ResNet50's input layer expects 3 channels, so a grayscale image is
    stacked into (H, W, 3) with the same values in every channel.

    Args:
        gray: grayscale image of shape (H, W).

    Returns:
        Image of shape (H, W, 3) with identical channels.
    """
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def normalize_imagenet(image_rgb: np.ndarray) -> np.ndarray:
    """
    Normalize a 0..255 image with the ImageNet mean and standard deviation.

    The ResNet50 backbone was pretrained on ImageNet, whose training
    images were normalized with these constants. Using the same values
    keeps the input distribution similar to what the pretrained weights
    learned.

    Args:
        image_rgb: image in RGB channel order (NOT OpenCV's BGR!),
                   shape (H, W, 3), values 0..255.

    Returns:
        float32 array, shape (H, W, 3), roughly in the range -2..+2.
    """
    image_rgb = image_rgb.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    return (image_rgb - mean) / std


def threshold_otsu(gray: np.ndarray) -> np.ndarray:
    """
    Binarize an image with Otsu's automatic threshold.

    DEMO ONLY: Otsu separates "bright" from "dark" pixels, which looks
    like segmentation but is far too naive for tumors (it also catches
    the skull, background, and bright normal tissue). It is included to
    show why classical thresholding is not enough and why we use a
    learned model instead.

    Args:
        gray: grayscale image (H, W), uint8.

    Returns:
        Binary uint8 image (0 = background, 255 = foreground).
    """
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ----------------------------------------------------------------------
# Ready-made pipelines for the model
# ----------------------------------------------------------------------

def preprocess_for_model(image_bgr: np.ndarray,
                         pipeline: str = "raw") -> np.ndarray:
    """
    Run a full preprocessing pipeline and return a displayable image.

    Args:
        image_bgr: image as loaded by OpenCV (BGR, uint8, any size).
        pipeline:  one of PIPELINES:
                   - "raw":        resize only
                   - "clahe":      resize -> grayscale -> CLAHE
                   - "blur_clahe": resize -> blur -> grayscale -> CLAHE

    Returns:
        RGB uint8 image of shape (IMG_SIZE, IMG_SIZE, 3), values 0..255.
        Grayscale pipelines are replicated to 3 identical channels.
    """
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown pipeline '{pipeline}', choose from {PIPELINES}")

    img = resize_image(image_bgr)

    if pipeline == "raw":
        # Keep the original channels, only fix the channel order for
        # normalization later (OpenCV loads BGR, ImageNet expects RGB).
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if pipeline == "clahe":
        gray = to_grayscale(img)
        gray = apply_clahe(gray)
        # GRAY2BGR creates three IDENTICAL channels, so the BGR/RGB
        # channel order is irrelevant here.
        return replicate_to_3_channels(gray)

    # pipeline == "blur_clahe"
    img = gaussian_blur(img)
    gray = to_grayscale(img)
    gray = apply_clahe(gray)
    return replicate_to_3_channels(gray)


def normalize_for_model(image_rgb: np.ndarray) -> np.ndarray:
    """
    Second pipeline stage: normalize the preprocessed RGB image for the
    network. Kept separate from preprocess_for_model() so the app can
    *show* the preprocessed image (0..255) while still feeding the
    normalized version to the model.

    Args:
        image_rgb: RGB uint8 image, shape (H, W, 3).

    Returns:
        Normalized float32 array, shape (H, W, 3).
    """
    return normalize_imagenet(image_rgb)
