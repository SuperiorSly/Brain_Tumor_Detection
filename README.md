# Brain Tumor Detection and Classification Using Image Processing and Deep Learning

An educational project that classifies brain MRI scans into 4 classes —
**glioma, meningioma, notumor, pituitary** — using a fine-tuned ResNet50
(transfer learning), classical image processing (OpenCV), and Grad-CAM
heatmaps that show which image regions influenced each prediction.
Includes a preprocessing ablation study and a Streamlit web app.

> ## ⚠️ DISCLAIMER — Educational prototype. Not for medical diagnosis.
> This project was built for a university course to demonstrate a typical
> ML pipeline. It is **not** a medical device, has **not** been clinically
> validated, and its outputs must **never** be used for real diagnostic
> decisions. See [Limitations](#limitations-honest-assessment) before
> interpreting any number in this README.

## Results (held-out test set)

The final model was trained on 4,480 MRI images (pipeline: `raw`) and
evaluated ONCE on 1,486 held-out test images (the Kaggle `Testing`
folder, minus 114 images that duplicate training content).

**Accuracy 94.8% | Macro-F1 94.6%**

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| glioma | 1.0000 | 0.8177 | 0.8997 | 395 |
| meningioma | 0.8680 | 0.9867 | 0.9236 | 300 |
| notumor | 0.9422 | 1.0000 | 0.9702 | 391 |
| pituitary | 0.9803 | 0.9975 | 0.9888 | 400 |

- **Tumor safety** (the worst error is missing a tumor): "any tumor"
  recall **97.8%** — 24 of 1,095 tumor images were called "notumor".
  All 24 misses are gliomas.
- **Most common mistakes:** glioma→meningioma (44 images, 11.1% of
  gliomas), glioma→notumor (24, 6.1%). Everything else is ≤4 images.
- Full artifacts: `outputs/classification_report.txt`,
  `outputs/confusion_matrix.png`, `outputs/evaluation_summary.json` / `.md`,
  `outputs/training_curves.png`.

### Preprocessing ablation (validation set, fixed 4+3 epoch budget, same seed)

| pipeline | val accuracy | val macro-F1 | best val loss |
|---|---|---|---|
| **raw** (resize only) | **0.9634** | **0.9634** | **0.1059** |
| clahe | 0.9563 | 0.9561 | 0.1364 |
| blur_clahe | 0.9634 | 0.9634 | 0.1174 |

`raw` and `blur_clahe` tied on accuracy/macro-F1; `raw` won the tie-break
with the lowest validation loss, so the final model and app use plain
resize + ImageNet normalization. Honest reading: with a strong
ImageNet-pretrained backbone, CLAHE gave no measurable benefit here
(differences of <1% are within run-to-run noise).

### Grad-CAM findings

`outputs/gradcam_correct.png` and `outputs/gradcam_incorrect.png` show
heatmaps for 8 correct + 8 incorrect test predictions; the honest
assessment is in `outputs/gradcam_notes.md`. Headlines:

- For meningioma and large glioma masses the model genuinely looks at
  the lesion (including for most *wrong* predictions: right region,
  wrong class label).
- **One confirmed shortcut:** for one `notumor` image the strongest heat
  sits on a *text watermark banner* at the top of the image — the model
  uses the banner as "no tumor" evidence.
- Some heat falls on borders/skull/ear in a few samples.
- Grad-CAM shows *influence*, not tumor boundaries (7×7 cells upscaled).

## Project structure

```
├── README.md
├── requirements.txt
├── app.py                  # Streamlit app (Phase 7)
├── data/                   # dataset (not committed; see .gitignore)
│   └── raw/{Training,Testing}/<class>/*.jpg
├── models/                 # saved weights (best_model.pt)
├── outputs/                # plots, metrics, heatmaps, reports
├── notebooks/              # optional exploration
└── src/
    ├── config.py           # paths, classes, image size, hyperparameters
    ├── dataset.py          # Dataset/DataLoader, split, augmentation
    ├── preprocessing.py    # OpenCV steps + model pipelines
    ├── model.py            # ResNet50 setup, freeze/unfreeze
    ├── train.py            # 2-stage training loop, early stopping
    ├── evaluate.py         # test-set metrics, confusion matrix, summary
    ├── gradcam.py          # manual Grad-CAM (PyTorch hooks)
    ├── ablation.py         # preprocessing comparison experiment
    ├── inspect_data.py     # data inspection + duplicate check
    └── demo_preprocessing.py  # step-by-step preprocessing figure
```

## Setup

Requires Python 3.10+ (tested with 3.14) and optionally an NVIDIA GPU.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows (bash: source .venv/bin/activate)

# 2. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` includes `--extra-index-url
https://download.pytorch.org/whl/cu126` because plain `pip install torch`
on Windows gives a **CPU-only** wheel. Tested with an NVIDIA GTX 1660;
if you have a different GPU/driver, pick the right CUDA version at
https://pytorch.org/get-started/locally/. CPU-only training also works,
just slower.

## How to get the data

**Option A — automatic (recommended):** `kagglehub` downloads public
datasets without credentials. The first run of any training script will
NOT do this automatically; run the snippet below once:

```python
import kagglehub
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
# then copy the Training/ and Testing/ folders into data/raw/
```

**Option B — manual:** download from
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
and extract so that the layout is:

```
data/raw/
├── Training/    # 5600 images (1400 per class)
│   ├── glioma/  ├── meningioma/  ├── notumor/  └── pituitary/
└── Testing/     # 1600 images (400 per class)
    ├── glioma/  ├── meningioma/  ├── notumor/  └── pituitary/
```

Dataset facts (measured by `src/inspect_data.py`): perfectly balanced
classes (v2 of the dataset), 381 unique image sizes (dominant 512×512,
range 150×168–1375×1446), 138 duplicate groups *inside* Training, and
114 test images with pixel-identical content to training images
(different files, same content) — the loader excludes those 114.

## How to run

All scripts run from the project root:

```bash
python -m src.inspect_data        # counts, sizes, sample grid, duplicates
python -m src.demo_preprocessing  # preprocessing steps figure
python -m src.train --smoke       # 8-image pipeline check (NOT real training)
python -m src.ablation            # raw vs clahe vs blur_clahe comparison
python -m src.train               # full two-stage training (~20-30 min on GPU)
python -m src.evaluate            # test-set evaluation (requires trained model)
python -m src.gradcam             # Grad-CAM figures for correct/wrong predictions
streamlit run app.py              # the web app
```

Training details: ResNet50 (ImageNet weights) → new 4-class head.
Stage 1 trains only the head (lr 1e-3); Stage 2 unfreezes `layer4`
(lr 1e-4). AdamW, cross-entropy (class weights supported),
ReduceLROnPlateau, early stopping on validation loss (patience 5),
seeds set. Augmentation (±10° rotation, flips, brightness/contrast) is
applied to the **training set only**.

## Data split and the leakage caveat

- **Train/validation:** stratified 80/20 split of the Kaggle `Training`
  folder (4,480 / 1,120 images).
- **Test:** the Kaggle `Testing` folder, held out completely and used
  only once in `evaluate.py` (1,486 images after duplicate removal).
- **The split is NOT patient-level.** The dataset has no patient IDs, so
  scans from the same patient may appear in both training and test.
  Reported metrics are therefore likely **optimistic**. This caveat is
  repeated in the README, in `evaluate.py` output, and inside the
  evaluation summary files shown in the app.

## Limitations (honest assessment)

1. **Possible data leakage / no patient-level split** — see above.
   Accuracy here is not an estimate of real-world performance.
2. **Single dataset, no external validation.** All images come from one
   Kaggle collection. Performance on images from other scanners,
   centers, or MRI sequences is unknown and probably worse.
3. **Dataset quality issues.** Images mix axial, coronal and sagittal
   views; at least one apparent CT scan sits in the `notumor` class;
   some images contain watermark banners. The model demonstrably uses
   at least one of these banners as a "notumor" cue (see Grad-CAM notes)
   — a shortcut that will not generalize.
4. **Overconfident softmax.** Feeding the app a pure noise image
   produced "notumor, 100.0% confidence". The model cannot say "I don't
   know"; the 70% "uncertain" warning rarely triggers because softmax
   confidence stays high even on out-of-distribution inputs. (Future
   work: temperature scaling, OOD detection.)
5. **Grad-CAM is not segmentation.** It highlights regions that
   influenced the class score at 7×7 resolution (upsampled). It is not a
   tumor boundary, not a segmentation mask, and cannot validate that a
   prediction is medically correct.
6. **glioma is the weak class.** Recall 81.8%: 11% of gliomas are called
   meningioma and 6% are missed as "notumor". For a screening tool, the
   missed-tumor number (not overall accuracy) would be the metric to
   improve.
7. **Not a medical device.** No regulatory compliance, no clinical
   validation, no radiologist study. Educational prototype only.
