"""
Phase 6: Grad-CAM heatmaps, implemented MANUALLY with PyTorch hooks.

Why manual? The pytorch-grad-cam package does the same thing, but the
core algorithm is ~30 lines and writing it ourselves makes every step
explainable (better for learning and for defending the project).

How Grad-CAM works (3 steps, see GradCAM.generate):
 1. A forward hook on model.layer4 (the last convolutional block) saves
    the feature maps A_k (2048 channels, each 7x7 for a 224x224 input).
 2. We pick the class score and compute its gradient w.r.t. those
    feature maps. Large gradient = "this channel mattered for the score".
 3. The per-channel gradients are averaged over the 7x7 grid -> one
    weight alpha_k per channel. The CAM is the ReLU'd weighted sum of
    the channel maps, normalized to 0..1, then upscaled to image size.

IMPORTANT interpretation limits (also in the notes file):
  - Grad-CAM shows which regions INFLUENCED the prediction.
  - It is NOT a tumor segmentation and NOT an exact boundary.
  - It is computed on the PREPROCESSED model input (our pipeline is
    "raw" = resize only, so the overlay sits on the resized original).

Outputs:
  outputs/gradcam_correct.png    - sample of CORRECT test predictions
  outputs/gradcam_incorrect.png  - sample of WRONG test predictions
  outputs/gradcam_notes.md       - honest qualitative assessment

Run from the project root (after training):
    python -m src.gradcam
"""

import json
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
    OUTPUTS_DIR,
    PIPELINE,
)
from src.dataset import make_dataloaders
from src.evaluate import load_checkpoint
from src.model import build_model
from src.preprocessing import preprocess_for_model
from src.train import collect_predictions

CORRECT_PNG = OUTPUTS_DIR / "gradcam_correct.png"
INCORRECT_PNG = OUTPUTS_DIR / "gradcam_incorrect.png"
NOTES_MD = OUTPUTS_DIR / "gradcam_notes.md"

N_CORRECT_PER_CLASS = 2   # correct samples per class in the figure
N_INCORRECT_TOTAL = 8     # wrong predictions to visualize


# ----------------------------------------------------------------------
# The Grad-CAM engine
# ----------------------------------------------------------------------

class GradCAM:
    """
    Manual Grad-CAM for a CNN using a forward hook.

    Usage:
        cam = GradCAM(model)
        heatmap, class_idx = cam.generate(input_tensor)   # heatmap 7x7 in 0..1
        cam.remove()                                      # detach the hook
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        # Default: ResNet50's layer4 = the LAST convolutional block.
        self.target_layer = target_layer if target_layer is not None \
            else model.layer4
        self._activation = None
        # The hook fires on every forward pass and stores the output
        # tensor WITH its computation graph (needed for autograd.grad).
        self._handle = self.target_layer.register_forward_hook(
            self._save_activation)

    def _save_activation(self, module, inputs, output):
        self._activation = output

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        """
        Compute the Grad-CAM for one image.

        Args:
            input_tensor: normalized model input, shape (1, 3, H, W).
                          Call with gradients ENABLED (the default).
            class_idx:    class to explain; None = the predicted class.

        Returns:
            (heatmap, class_idx): heatmap as numpy 7x7, values 0..1.
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)              # hook fills _activation
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]

        # Step 2: d(score)/d(feature maps)
        grads = torch.autograd.grad(score, self._activation)[0]   # (1,C,7,7)

        # Step 3: alpha_k = mean gradient over the spatial grid
        weights = grads.mean(dim=(2, 3))               # (1,C)
        acts = self._activation[0]                     # (C,7,7)
        cam = (weights[0][:, None, None] * acts).sum(dim=0)  # (7,7)
        cam = torch.relu(cam)                          # only positive evidence
        if float(cam.detach().max()) > 0:
            cam = cam / cam.max()                      # normalize to 0..1

        self.model.zero_grad(set_to_none=True)
        return cam.detach().cpu().numpy(), class_idx

    def remove(self):
        """Detach the hook (call when done)."""
        self._handle.remove()


# ----------------------------------------------------------------------
# Heatmap -> display
# ----------------------------------------------------------------------

def overlay_heatmap(rgb_uint8: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """
    Overlay a 0..1 heatmap on an RGB image.

    The heatmap is upscaled to the image size with bilinear interpolation
    and colored with OpenCV's JET colormap, then alpha-blended:
    result = alpha * heatmap + (1 - alpha) * image.
    """
    h, w = rgb_uint8.shape[:2]
    heat = cv2.resize(heatmap.astype(np.float32), (w, h),
                      interpolation=cv2.INTER_LINEAR)
    heat_bgr = cv2.applyColorMap((heat * 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    blended = alpha * heat_rgb + (1 - alpha) * rgb_uint8
    return np.clip(blended, 0, 255).astype(np.uint8)


def focus_region_box(rgb_uint8: np.ndarray, heatmap: np.ndarray,
                     top_fraction: float = 0.15,
                     max_box_fraction: float = 0.60) -> np.ndarray:
    """
    Draw a rectangle around the region with the HIGHEST attention.

    How it works: the heatmap is upscaled to the image size, the top
    `top_fraction` of pixels (by attention value) are thresholded, and a
    box is drawn around the largest contiguous high-attention region.

    Two cases produce NO box (the image is returned unchanged):
      - no contiguous high-attention region was found, or
      - the region is too diffuse: if its bounding box covers more than
        `max_box_fraction` of the image, the attention is spread over
        most of the scan and a box would just point at "the whole brain"
        (this is what caused random-looking squares on some images).

    IMPORTANT: this is a ROUGH visual aid ("where the model looked"),
    NOT an exact tumor boundary. The Grad-CAM resolution (7x7, upsampled)
    and the classification objective do not carry boundary information.
    """
    h, w = rgb_uint8.shape[:2]
    heat = cv2.resize(heatmap.astype(np.float32), (w, h),
                      interpolation=cv2.INTER_LINEAR)

    threshold = float(np.quantile(heat, 1.0 - top_fraction))
    binary = (heat >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = rgb_uint8.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        # Skip diffuse attention: a box covering most of the image
        # carries no localization information.
        if bw * bh > max_box_fraction * h * w:
            return out
        # Teal rectangle (matches the theme accent), RGB order.
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (45, 212, 191), 2)
    return out


# ----------------------------------------------------------------------
# Sample selection and figure building
# ----------------------------------------------------------------------

def pick_samples(y_true: list[int], y_pred: list[int]) -> tuple[dict, list]:
    """
    Choose which test images to visualize.

    Correct: the first N_CORRECT_PER_CLASS per class.
    Incorrect: grouped by mistake type (true -> predicted), taking the
    most common mistake types first, up to 2 each, N_INCORRECT_TOTAL total.
    """
    correct = defaultdict(list)
    incorrect_groups = defaultdict(list)
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            correct[t].append(idx)
        else:
            incorrect_groups[(t, p)].append(idx)

    correct_sel = {cls: correct[cls][:N_CORRECT_PER_CLASS]
                   for cls in range(len(CLASS_NAMES))}

    # Most frequent mistake types first (ties keep insertion order).
    top_groups = sorted(incorrect_groups, key=lambda k: -len(incorrect_groups[k]))
    incorrect_sel = []
    for group in top_groups:
        incorrect_sel.extend(incorrect_groups[group][:2])
        if len(incorrect_sel) >= N_INCORRECT_TOTAL:
            break
    return correct_sel, incorrect_sel


def build_correct_figure(dataset, samples_per_class: dict, cam: GradCAM,
                         pipeline: str, out_path):
    """Rows = classes, per sample: overlay + raw heatmap."""
    n_cols = max(1, 2 * max(len(v) for v in samples_per_class.values()))
    fig, axes = plt.subplots(len(CLASS_NAMES), n_cols,
                             figsize=(2.6 * n_cols, 2.9 * len(CLASS_NAMES)))
    axes = np.atleast_2d(axes)

    for r, cls in enumerate(CLASS_NAMES):
        indices = samples_per_class[r]
        for k in range(n_cols):
            ax = axes[r, k]
            ax.axis("off")
            sample_i, part_i = divmod(k, 2)   # 2 panels per sample
            if sample_i >= len(indices):
                continue
            idx = indices[sample_i]
            path, label = dataset.samples[idx]
            rgb = preprocess_for_model(cv2.imread(str(path)), pipeline)
            tensor = dataset[idx][0].unsqueeze(0).to(DEVICE)
            heatmap, pred = cam.generate(tensor)
            if part_i == 0:
                ax.imshow(overlay_heatmap(rgb, heatmap))
                ax.set_title(f"pred {CLASS_NAMES[pred]}  (correct)",
                             fontsize=8)
            else:
                ax.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
                ax.set_title("heatmap", fontsize=8)
        axes[r, 0].set_ylabel(cls, fontsize=10)

    fig.suptitle("Grad-CAM - CORRECT test predictions "
                 "(left: overlay, right: raw heatmap)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"saved: {out_path}")


def build_incorrect_figure(dataset, indices: list, y_true: list[int],
                           y_pred: list[int], cam: GradCAM,
                           pipeline: str, out_path):
    """Rows = mistake types (true -> predicted), per sample overlay+heatmap."""
    # Group the selected indices by mistake type using the ALREADY known
    # predictions from the test pass (no need to re-run the model).
    groups = defaultdict(list)
    for idx in indices:
        groups[(y_true[idx], y_pred[idx])].append(idx)

    rows = list(groups)
    n_cols = max(1, 2 * max(len(groups[g]) for g in rows))
    fig, axes = plt.subplots(len(rows), n_cols,
                             figsize=(2.6 * n_cols, 2.9 * len(rows)))
    axes = np.atleast_2d(axes)

    for r, (t, p) in enumerate(rows):
        for k in range(n_cols):
            ax = axes[r, k]
            ax.axis("off")
            sample_i, part_i = divmod(k, 2)
            if sample_i >= len(groups[(t, p)]):
                continue
            idx = groups[(t, p)][sample_i]
            path, label = dataset.samples[idx]
            rgb = preprocess_for_model(cv2.imread(str(path)), pipeline)
            tensor = dataset[idx][0].unsqueeze(0).to(DEVICE)
            heatmap, pred = cam.generate(tensor)
            if part_i == 0:
                ax.imshow(overlay_heatmap(rgb, heatmap))
                ax.set_title(f"pred {CLASS_NAMES[pred]}  (WRONG)",
                             fontsize=8, color="red")
            else:
                ax.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
                ax.set_title("heatmap", fontsize=8)
        axes[r, 0].set_ylabel(f"true {CLASS_NAMES[t]}\n-> pred {CLASS_NAMES[p]}",
                              fontsize=9)

    fig.suptitle("Grad-CAM - INCORRECT test predictions "
                 "(left: overlay, right: raw heatmap)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"saved: {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    checkpoint = load_checkpoint(BEST_MODEL_PATH)
    pipeline = checkpoint.get("pipeline", PIPELINE)
    print(f"pipeline (from checkpoint): {pipeline}")

    loaders = make_dataloaders(pipeline=pipeline)
    test_loader = loaders["test_loader"]
    dataset = test_loader.dataset   # order preserved (shuffle=False)

    model = build_model(pretrained=False).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    # One full test pass to know which predictions were correct/wrong.
    y_true, y_pred = collect_predictions(model, test_loader, DEVICE)
    correct_sel, incorrect_sel = pick_samples(y_true, y_pred)
    print(f"selected {sum(len(v) for v in correct_sel.values())} correct "
          f"and {len(incorrect_sel)} incorrect samples")

    cam = GradCAM(model)
    try:
        build_correct_figure(dataset, correct_sel, cam, pipeline, CORRECT_PNG)
        build_incorrect_figure(dataset, incorrect_sel, y_true, y_pred, cam,
                               pipeline, INCORRECT_PNG)
    finally:
        cam.remove()

    summary = {
        "method": "manual Grad-CAM via PyTorch forward hook on model.layer4",
        "n_correct_shown": sum(len(v) for v in correct_sel.values()),
        "n_incorrect_shown": len(incorrect_sel),
        "note": "Grad-CAM highlights regions that INFLUENCED the class "
                "score. It is not a segmentation and not an exact tumor "
                "boundary.",
    }
    (OUTPUTS_DIR / "gradcam_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nfigures + gradcam_summary.json saved to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
