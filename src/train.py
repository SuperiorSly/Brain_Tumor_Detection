"""
Phase 3: two-stage training of ResNet50.

Stage 1 (head training): backbone frozen, only the new 4-class head is
    trained with a higher learning rate.
Stage 2 (fine-tuning): layer4 is unfrozen and the whole trainable part is
    fine-tuned with a lower learning rate.

Both stages use AdamW + cross-entropy (class weights supported),
a ReduceLROnPlateau scheduler and early stopping on the VALIDATION loss.
The best model (lowest val loss across both stages) is saved to
models/best_model.pt, training curves to outputs/.

The TEST set is NOT touched here - it is evaluated only in Phase 5.

Usage (from the project root):
    python -m src.train            # full training
    python -m src.train --smoke    # 2 images per class, pipeline check only
"""

import argparse
import copy
import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    FINETUNE_EPOCHS,
    FINETUNE_LR,
    HEAD_EPOCHS,
    HEAD_LR,
    PIPELINE,
    SEED,
    TRAIN_HISTORY_JSON,
    TRAINING_CURVES_PNG,
    WEIGHT_DECAY,
)
from src.dataset import make_dataloaders
from src.model import build_model, freeze_backbone, trainable_summary, unfreeze_layer4


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    """
    Seed python, numpy and torch (CPU + all GPUs) for reproducibility.

    Honest note: with cudnn.benchmark enabled for speed, results are
    ALMOST reproducible (same seed -> very close metrics) but not
    bit-identical on every machine.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Training utilities
# ----------------------------------------------------------------------

def make_optimizer(model: nn.Module, lr: float) -> torch.optim.AdamW:
    """
    AdamW over the trainable parameters.

    Weight decay is skipped for 1-dimensional parameters (biases and
    BatchNorm scale/shift): decaying them would distort the normalization,
    a standard practice in transfer learning.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.ndim > 1 else no_decay).append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr)


def run_one_epoch(model, loader, criterion, device, optimizer=None):
    """
    One pass over a DataLoader.

    If optimizer is given -> training mode (forward + backward + step).
    Otherwise -> evaluation mode (forward only, no gradients).
    Returns (mean_loss, accuracy).
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, n = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch = labels.size(0)
            total_loss += loss.item() * batch
            correct += (logits.argmax(dim=1) == labels).sum().item()
            n += batch
    return total_loss / n, correct / n


def train_stage(model, train_loader, val_loader, criterion, device, *,
                epochs: int, lr: float, stage_name: str,
                patience: int = EARLY_STOPPING_PATIENCE) -> tuple[dict, list]:
    """
    Train one stage and return (best, history_rows).

    Early stopping: if the validation loss does not improve for
    `patience` epochs, the stage stops. The best model state (lowest val
    loss) is kept in memory (moved to CPU) and returned.
    """
    optimizer = make_optimizer(model, lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    best = {"val_loss": float("inf"), "val_acc": 0.0,
            "epoch": 0, "state": None}
    rows, epochs_since_improvement = [], 0

    print(f"\n--- {stage_name}: {epochs} epochs max, lr={lr}, "
          f"early stopping patience={patience} ---")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_one_epoch(
            model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        improved = val_loss < best["val_loss"] - 1e-6
        if improved:
            best = {
                "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch,
                # keep a CPU copy of the best weights
                "state": {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()},
            }
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        current_lr = optimizer.param_groups[0]["lr"]
        marker = "  <-- best" if improved else ""
        print(f"[{stage_name} {epoch:>2}/{epochs}] "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"lr {current_lr:.2e} | {elapsed:5.1f}s{marker}")

        rows.append({
            "stage": stage_name, "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": current_lr, "time_s": elapsed,
        })

        if epochs_since_improvement >= patience:
            print(f"early stopping: no val-loss improvement for "
                  f"{patience} epochs")
            break

    return best, rows


def collect_predictions(model, loader, device) -> tuple[list, list]:
    """
    Run inference over a DataLoader and return (y_true, y_pred) lists.
    Shared by the ablation (Phase 4) and evaluation (Phase 5).
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred


# ----------------------------------------------------------------------
# Plotting / saving
# ----------------------------------------------------------------------

def plot_curves(rows: list[dict], out_path) -> None:
    """Training/validation loss + accuracy curves for both stages."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    x = list(range(1, len(rows) + 1))

    for ax, key, label in [(ax_loss, "loss", "Loss"),
                           (ax_acc, "acc", "Accuracy")]:
        ax.plot(x, [r[f"train_{key}"] for r in rows], "o-", label=f"train {key}")
        ax.plot(x, [r[f"val_{key}"] for r in rows], "s-", label=f"val {key}")
        ax.set_xlabel("epoch (both stages concatenated)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()

    # Mark where stage 2 (fine-tuning) starts.
    boundary = next((i for i, r in enumerate(rows)
                     if r["stage"] != rows[0]["stage"]), None)
    if boundary:
        for ax in (ax_loss, ax_acc):
            ax.axvline(boundary + 1, color="gray", linestyle="--", alpha=0.7)
            ax.text(boundary + 1.1, ax.get_ylim()[1] * 0.9, "layer4 fine-tune",
                    fontsize=8, color="gray")

    fig.suptitle(f"Training curves (pipeline: {PIPELINE})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="train on only 2 images per class (pipeline "
                             "check, metrics NOT meaningful)")
    args = parser.parse_args()

    set_seed()
    device = DEVICE
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # --- Data -----------------------------------------------------------
    loaders = make_dataloaders(smoke_limit=2 if args.smoke else None)
    counts = loaders["counts"]
    for split in ("train", "val", "test"):
        print(f"{split:>5}: {sum(counts[split].values()):>5} images "
              f"{counts[split]}")

    # The dataset is balanced, so weights are ~1.0; the mechanism stays
    # ready in case class sizes ever change.
    weights = loaders["class_weights"]
    print(f"class weights: {[round(w, 3) for w in weights.tolist()]} "
          f"(~1.0 because classes are balanced)")
    print(f"preprocessing pipeline: {loaders['pipeline']}")

    if args.smoke:
        print("\n!!! SMOKE MODE - metrics are NOT meaningful, "
              "this only checks that the pipeline runs !!!")

    # --- Model + loss ---------------------------------------------------
    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    # --- Stage 1: head only ---------------------------------------------
    freeze_backbone(model)
    print(f"\nStage 1 - head only: {trainable_summary(model)}")
    best_stage1, rows1 = train_stage(
        model, loaders["train_loader"], loaders["val_loader"], criterion,
        device, epochs=HEAD_EPOCHS, lr=HEAD_LR, stage_name="stage1-head")

    # Continue stage 2 from the BEST stage-1 weights.
    model.load_state_dict(best_stage1["state"])

    # --- Stage 2: fine-tune layer4 ---------------------------------------
    unfreeze_layer4(model)
    print(f"\nStage 2 - layer4 fine-tune: {trainable_summary(model)}")
    best_stage2, rows2 = train_stage(
        model, loaders["train_loader"], loaders["val_loader"], criterion,
        device, epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR, stage_name="stage2-layer4")

    # --- Pick the global best (lowest val loss across stages) ------------
    if best_stage2["val_loss"] <= best_stage1["val_loss"]:
        global_best, global_stage = best_stage2, "stage2-layer4"
    else:
        global_best, global_stage = best_stage1, "stage1-head"

    # --- Save model, history, curves --------------------------------------
    BEST_MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save({
        "state_dict": global_best["state"],
        "class_names": CLASS_NAMES,
        "pipeline": loaders["pipeline"],
        "img_size": 224,
        "stage": global_stage,
        "epoch": global_best["epoch"],
        "val_loss": global_best["val_loss"],
        "val_acc": global_best["val_acc"],
        "smoke_run": bool(args.smoke),
    }, BEST_MODEL_PATH)

    all_rows = rows1 + rows2
    TRAIN_HISTORY_JSON.write_text(json.dumps(all_rows, indent=2))
    plot_curves(all_rows, TRAINING_CURVES_PNG)

    print("\n" + "=" * 60)
    print("TRAINING DONE"
          + (" (SMOKE RUN - metrics not meaningful!)" if args.smoke else ""))
    print(f"best stage:    {global_stage} (epoch {global_best['epoch']})")
    print(f"best val loss: {global_best['val_loss']:.4f}")
    print(f"best val acc:  {global_best['val_acc']:.4f}")
    print(f"model saved to: {BEST_MODEL_PATH}")
    print(f"curves saved to: {TRAINING_CURVES_PNG}")
    print("=" * 60)


if __name__ == "__main__":
    main()
