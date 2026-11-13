"""
Phase 5: evaluation on the HELD-OUT TEST SET only.

This script is run ONCE after training, on data the model has never seen
during training or model selection. It reports:

  - overall accuracy and macro-F1
  - per-class precision / recall / F1 (sklearn classification_report)
  - confusion matrix (saved as an image)
  - TUMOR SAFETY metrics: recall for "any tumor" vs "no tumor", because
    the worst error is telling a patient with a tumor that they are fine
  - the most common mistakes (e.g. glioma <-> meningioma)

Everything is saved to outputs/ and as a JSON + markdown summary the
Streamlit app can display.

!!! DATA LEAKAGE LIMITATION (also in README) !!!
This dataset has no patient IDs, so images from the same patient may
appear in both train and test. Reported metrics may be OPTIMISTIC and
must not be interpreted as clinical performance. This note is included
in every saved output on purpose.

Safety: refuses to evaluate a checkpoint produced by `--smoke` training
unless you pass --allow-smoke, so meaningless numbers never leak into
reports.

Run from the project root (AFTER full training has finished):
    python -m src.evaluate
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support)

from src.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    DEVICE,
    OUTPUTS_DIR,
    PIPELINE,
)
from src.dataset import make_dataloaders
from src.train import collect_predictions

REPORT_TXT = OUTPUTS_DIR / "classification_report.txt"
CONFUSION_PNG = OUTPUTS_DIR / "confusion_matrix.png"
SUMMARY_JSON = OUTPUTS_DIR / "evaluation_summary.json"
SUMMARY_MD = OUTPUTS_DIR / "evaluation_summary.md"

# "notumor" is the only class without a tumor; every other class is a tumor.
TUMOR_LABELS = [i for i, name in enumerate(CLASS_NAMES) if name != "notumor"]
NOTUMOR_LABEL = CLASS_NAMES.index("notumor")

LEAKAGE_NOTE = (
    "LIMITATION: the dataset has no patient IDs, so images from the same "
    "patient may occur in both training and test. Metrics may be OPTIMISTIC "
    "(possible data leakage). This is an educational prototype, NOT a "
    "medical device and NOT clinically validated.")


def load_checkpoint(path, allow_smoke: bool = False):
    """Load the saved checkpoint dict (always on CPU first)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("smoke_run") and not allow_smoke:
        raise SystemExit(
            "This checkpoint was produced by `--smoke` training (8 images) "
            "and its metrics are NOT meaningful.\n"
            "Run `python -m src.train` for a real training first, or pass "
            "--allow-smoke to evaluate it anyway.")
    return checkpoint


def compute_tumor_safety(y_true: list[int], y_pred: list[int]) -> dict:
    """
    Tumor-vs-notumor safety metrics.

    The clinically worst error is a FALSE NEGATIVE: an image that DOES
    contain a tumor predicted as 'notumor'. We therefore report the recall
    of the pooled 'any tumor' class and count the missed tumors.
    """
    tumor_true_idx = [i for i, t in enumerate(y_true) if t in TUMOR_LABELS]
    missed = [i for i in tumor_true_idx if y_pred[i] == NOTUMOR_LABEL]

    return {
        "n_tumor_images": len(tumor_true_idx),
        "n_missed_as_notumor": len(missed),
        "missed_tumor_rate": round(len(missed) / len(tumor_true_idx), 4),
        "tumor_recall": round(1 - len(missed) / len(tumor_true_idx), 4),
    }


def top_confusions(cm: np.ndarray, top_n: int = 5) -> list[dict]:
    """
    Most common off-diagonal confusions, sorted by count.
    share_of_true = how many % of all true-X images ended up as Y.
    """
    mistakes = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm[i, j] > 0:
                mistakes.append({
                    "true": CLASS_NAMES[i],
                    "predicted_as": CLASS_NAMES[j],
                    "count": int(cm[i, j]),
                    "share_of_true": round(
                        cm[i, j] / max(cm[i].sum(), 1), 4),
                })
    mistakes.sort(key=lambda m: m["count"], reverse=True)
    return mistakes[:top_n]


def plot_confusion_matrix(cm: np.ndarray, out_path) -> None:
    """Confusion matrix heatmap with counts, saved as PNG."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30,
                  ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("Confusion matrix - TEST set")

    # Write the count in every cell (white text on dark cells).
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(BEST_MODEL_PATH),
                        help="path to the checkpoint to evaluate")
    parser.add_argument("--allow-smoke", action="store_true",
                        help="override the smoke-checkpoint safety guard")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.model, allow_smoke=args.allow_smoke)

    # The checkpoint records which pipeline the model was TRAINED with -
    # evaluating with a different pipeline would be unfair and wrong.
    pipeline = checkpoint.get("pipeline", PIPELINE)
    print(f"evaluating {args.model}")
    print(f"pipeline (from checkpoint): {pipeline}")
    print(f"device: {DEVICE}")

    loaders = make_dataloaders(pipeline=pipeline)
    test_loader = loaders["test_loader"]
    n_test = sum(loaders["counts"]["test"].values())
    print(f"test images: {n_test} (duplicates vs training already excluded)")

    # --- Rebuild the model and load the trained weights -----------------
    from src.model import build_model
    model = build_model(pretrained=False).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    y_true, y_pred = collect_predictions(model, test_loader, DEVICE)

    # --- Metrics ---------------------------------------------------------
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    precision, recall, f1_per_class, support = \
        precision_recall_fscore_support(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    tumor_safety = compute_tumor_safety(y_true, y_pred)
    mistakes = top_confusions(cm)

    report = classification_report(y_true, y_pred,
                                   target_names=CLASS_NAMES, digits=4)

    # --- Console output ---------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(report)
    print(f"accuracy: {accuracy:.4f} | macro-F1: {macro_f1:.4f}")
    print("\nTUMOR SAFETY (worst error = missing a tumor):")
    print(f"  tumor images in test : {tumor_safety['n_tumor_images']}")
    print(f"  predicted as 'notumor': {tumor_safety['n_missed_as_notumor']} "
          f"({tumor_safety['missed_tumor_rate'] * 100:.2f}% missed)")
    print(f"  'any tumor' recall   : {tumor_safety['tumor_recall']:.4f}")
    print("\nMOST COMMON MISTAKES:")
    for m in mistakes:
        print(f"  true {m['true']:<11} -> predicted {m['predicted_as']:<11}"
              f" {m['count']:>4} images ({m['share_of_true'] * 100:.1f}% of true {m['true']})")
    print("\n" + LEAKAGE_NOTE)

    # --- Save everything ---------------------------------------------------
    OUTPUTS_DIR.mkdir(exist_ok=True)
    REPORT_TXT.write_text(report)

    plot_confusion_matrix(cm, CONFUSION_PNG)
    print(f"\nconfusion matrix saved to: {CONFUSION_PNG}")
    print(f"classification report saved to: {REPORT_TXT}")

    per_class = {name: {
        "precision": round(float(precision[k]), 4),
        "recall": round(float(recall[k]), 4),
        "f1": round(float(f1_per_class[k]), 4),
        "support": int(support[k]),
    } for k, name in enumerate(CLASS_NAMES)}

    summary = {
        "model_path": str(args.model),
        "pipeline": pipeline,
        "test_images": n_test,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "class_names": CLASS_NAMES,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "tumor_safety": tumor_safety,
        "top_mistakes": mistakes,
        "leakage_note": LEAKAGE_NOTE,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    # Markdown summary (the app displays this).
    md = [
        "# Test set evaluation",
        "",
        f"- model: `{args.model}` (pipeline: {pipeline})",
        f"- test images: **{n_test}** (114 duplicates vs training excluded)",
        f"- accuracy: **{accuracy:.4f}**",
        f"- macro-F1: **{macro_f1:.4f}**",
        "",
        "| class | precision | recall | F1 | support |",
        "|---|---|---|---|---|",
    ]
    for name in CLASS_NAMES:
        p = per_class[name]
        md.append(f"| {name} | {p['precision']:.4f} | {p['recall']:.4f} | "
                  f"{p['f1']:.4f} | {p['support']} |")
    md += [
        "",
        f"**Tumor safety:** 'any tumor' recall {tumor_safety['tumor_recall']:.4f} "
        f"({tumor_safety['n_missed_as_notumor']} of "
        f"{tumor_safety['n_tumor_images']} tumor images missed as 'notumor').",
        "",
        "**Most common mistakes:**",
    ]
    md += [f"- {m['true']} -> {m['predicted_as']}: {m['count']} images "
           f"({m['share_of_true'] * 100:.1f}% of true {m['true']})"
           for m in mistakes]
    md += ["", f"> {LEAKAGE_NOTE}"]
    SUMMARY_MD.write_text("\n".join(md))
    print(f"summary saved to: {SUMMARY_JSON}")
    print(f"summary saved to: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
