"""
Phase 4: preprocessing ablation study.

Question: which preprocessing pipeline should the final model use?
  - "raw":        resize only
  - "clahe":      resize -> grayscale -> CLAHE
  - "blur_clahe": resize -> blur -> grayscale -> CLAHE

Method: for EVERY pipeline we train the same model with the SAME seed
and the SAME fixed epoch budget (4 epochs head + 3 epochs layer4
fine-tune, no early stopping so all pipelines get identical training
time). We then compare on the VALIDATION set (the test set stays held
out for Phase 5):

  - accuracy
  - macro-F1 (unweighted mean of per-class F1, robust to imbalance)

The winner (highest macro-F1) decides config.PIPELINE for the final
training run and the app.

Caveat (stated honestly): a short fixed budget is a fair comparison,
but the winning pipeline could theoretically behave differently with
the full training schedule. That is a known limitation of ablations.

Run from the project root:
    python -m src.ablation
"""

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from src.config import DEVICE, FINETUNE_LR, HEAD_LR, OUTPUTS_DIR, SEED
from src.dataset import make_dataloaders
from src.model import build_model, freeze_backbone, trainable_summary, unfreeze_layer4
from src.preprocessing import PIPELINES
from src.train import collect_predictions, set_seed, train_stage

# Fixed budget for ALL pipelines (fair comparison, no early stopping).
STAGE1_EPOCHS = 4
STAGE2_EPOCHS = 3
RESULTS_JSON = OUTPUTS_DIR / "ablation_results.json"
TABLE_MD = OUTPUTS_DIR / "ablation_table.md"
CHART_PNG = OUTPUTS_DIR / "ablation_chart.png"


def run_pipeline(pipeline: str) -> dict:
    """Train + evaluate ONE pipeline and return its result row."""
    print("\n" + "=" * 60)
    print(f"PIPELINE: {pipeline}")
    print("=" * 60)

    # Same seed for every pipeline -> identical data order/augmentation.
    set_seed(SEED)
    loaders = make_dataloaders(pipeline=pipeline)
    counts = loaders["counts"]
    print(f"train {sum(counts['train'].values())} | "
          f"val {sum(counts['val'].values())} | test {sum(counts['test'].values())} (untouched)")

    model = build_model(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=loaders["class_weights"].to(DEVICE))
    t_start = time.time()

    # Stage 1: head only (patience=999 -> no early stopping, fixed budget).
    freeze_backbone(model)
    print(f"stage 1 head only: {trainable_summary(model)}")
    best1, rows1 = train_stage(
        model, loaders["train_loader"], loaders["val_loader"], criterion,
        DEVICE, epochs=STAGE1_EPOCHS, lr=HEAD_LR,
        stage_name=f"{pipeline}-s1", patience=999)

    # Stage 2: layer4 fine-tune, continuing from the best stage-1 weights.
    model.load_state_dict(best1["state"])
    unfreeze_layer4(model)
    print(f"stage 2 layer4: {trainable_summary(model)}")
    best2, rows2 = train_stage(
        model, loaders["train_loader"], loaders["val_loader"], criterion,
        DEVICE, epochs=STAGE2_EPOCHS, lr=FINETUNE_LR,
        stage_name=f"{pipeline}-s2", patience=999)

    best, stage = (best2, "stage2") if best2["val_loss"] <= best1["val_loss"] \
        else (best1, "stage1")

    # Final metrics on the validation set with sklearn.
    model.load_state_dict(best["state"])
    y_true, y_pred = collect_predictions(model, loaders["val_loader"], DEVICE)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    elapsed = time.time() - t_start
    print(f"\n>>> {pipeline}: val accuracy {acc:.4f}, macro-F1 {f1:.4f} "
          f"(best {stage}, total {elapsed / 60:.1f} min)")

    return {
        "pipeline": pipeline,
        "val_accuracy": round(acc, 4),
        "val_macro_f1": round(f1, 4),
        "best_val_loss": round(best["val_loss"], 4),
        "best_stage": stage,
        "total_minutes": round(elapsed / 60, 1),
        "history": rows1 + rows2,
    }


def save_table(results: list[dict], winner: str) -> None:
    """Markdown table + console output (one table, as requested)."""
    lines = [
        "# Preprocessing ablation (validation set, fixed budget 4+3 epochs)",
        "",
        "| pipeline | val accuracy | val macro-F1 | best val loss | best stage | time (min) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        star = " **<- winner**" if r["pipeline"] == winner else ""
        lines.append(
            f"| {r['pipeline']}{star} | {r['val_accuracy']:.4f} | "
            f"{r['val_macro_f1']:.4f} | {r['best_val_loss']:.4f} | "
            f"{r['best_stage']} | {r['total_minutes']} |")
    lines += ["",
              "Winner is chosen by macro-F1 (accuracy as tie-break). "
              "The test set was NOT used."]
    TABLE_MD.write_text("\n".join(lines))
    print("\n" + "\n".join(lines[2:]))
    print(f"\nTable saved to: {TABLE_MD}")


def save_chart(results: list[dict]) -> None:
    """Grouped bar chart: accuracy and macro-F1 per pipeline."""
    x = np.arange(len(results))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, [r["val_accuracy"] for r in results],
           width, label="val accuracy")
    ax.bar(x + width / 2, [r["val_macro_f1"] for r in results],
           width, label="val macro-F1")
    ax.set_xticks(x, [r["pipeline"] for r in results])
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Preprocessing ablation (validation set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHART_PNG, dpi=120)
    plt.close(fig)
    print(f"Chart saved to: {CHART_PNG}")


def main() -> None:
    results = [run_pipeline(p) for p in PIPELINES]

    # Winner: highest macro-F1, accuracy as tie-break.
    winner = max(results, key=lambda r: (r["val_macro_f1"], r["val_accuracy"]))
    print("\n" + "=" * 60)
    print(f"WINNER: '{winner['pipeline']}' "
          f"(macro-F1 {winner['val_macro_f1']:.4f})")
    print("Set config.PIPELINE to this value for the final training run.")
    print("=" * 60)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(
        {"results": results, "winner": winner["pipeline"],
         "budget": {"stage1_epochs": STAGE1_EPOCHS,
                    "stage2_epochs": STAGE2_EPOCHS,
                    "note": "fixed budget, no early stopping, same seed"}},
        indent=2))
    print(f"Results saved to: {RESULTS_JSON}")
    save_table(results, winner["pipeline"])
    save_chart(results)


if __name__ == "__main__":
    main()
