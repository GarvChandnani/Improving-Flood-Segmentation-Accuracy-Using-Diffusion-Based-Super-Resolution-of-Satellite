# FloodSense — AI Flood Segmentation

SAR-to-flood segmentation using Sentinel-1 SAR with optional synthetic optical imagery generated via Pix2Pix and BBDM diffusion.

## Architecture Overview

```
flood-project/
├── ml/
│   ├── config.py             ← Central path & hyperparameter configuration
│   ├── preprocessing.py      ← SAR normalization (Scheme A + B), loss functions
│   ├── datasets.py           ← FloodDataset, PairedDataset, MultiModalDataset, PairedLatentDataset
│   ├── data_loader.py        ← Catalog discovery, resumable download
│   ├── inference.py          ← load_model(), run_*_inference(), generate_synthetic_eo()
│   ├── train_baseline.py     ← SAR-only U-Net training
│   ├── train_pix2pix.py      ← SAR→SynEO Pix2Pix training
│   ├── train_multimodal.py   ← SAR+SynEO multimodal training
│   ├── train_bbdm.py         ← Latent diffusion BBDM training
│   └── models/
│       ├── unet.py           ← UNet (shared by all segmentation + pix2pix tracks)
│       ├── discriminator.py  ← PatchDiscriminator (pix2pix)
│       ├── bbdm_wrapper.py   ← BBDM/VQGAN wrapper (external repo)
│       └── bbdm_repo/        ← Cloned xuekt98/BBDM repo (gitignored)
│
├── data/
│   ├── raw/                  ← Original catalog ZIPs (gitignored)
│   ├── processed/
│   │   ├── S1/               ← SAR GeoTIFFs (VV/VH, gitignored)
│   │   ├── LabelHand/        ← Hand-labeled flood masks (gitignored)
│   │   ├── S2Hand/           ← Sentinel-2 optical tiles (gitignored)
│   │   ├── WeakLabel/        ← Otsu weak labels (gitignored)
│   │   ├── SynEO/            ← Pix2Pix generated EO .npy (gitignored)
│   │   └── BBDM_latents/     ← Pre-encoded VQGAN latents (gitignored)
│   └── samples/              ← Small demo samples (tracked in git)
│
├── ml/checkpoints/           ← Model weights (NOT in git — see Checkpoint Safety)
│   ├── baseline/             ← baseline_v1_ep20.pt, *.meta.json
│   ├── pix2pix/              ← pix2pix_gen_v1_ep40.pt, pix2pix_disc_v1_ep40.pt
│   ├── multimodal/           ← multimodal_latest.pt, multimodal_v1_ep20.pt
│   └── bbdm/                 ← model.ckpt (VQGAN), bbdm_unet_ep*.pt
│
├── outputs/                  ← Predictions, visualizations (gitignored)
├── scripts/
│   ├── download_checkpoints.py ← Dataset download + hash verification
│   ├── setup_bbdm.py           ← Clone BBDM repository
│   └── generate_syneo.py       ← Generate SynEO .npy for all chips
├── backend/                  ← Backend API (separate task)
├── frontend/                 ← Web frontend (separate task)
├── requirements.txt
└── README.md
```

## Four Model Tracks

| Track | Input | Output | Architecture | Checkpoint |
|-------|-------|--------|-------------|------------|
| Baseline | 2-ch SAR (VV, VH) | 1-ch flood mask | UNet(2→1) | `baseline/baseline_v1_ep20.pt` |
| Pix2Pix Generator | 2-ch SAR | 3-ch synthetic EO | UNet(2→3) + PatchDisc(5) | `pix2pix/pix2pix_gen_v1_ep40.pt` |
| Multimodal | 2-ch SAR + 3-ch SynEO | 1-ch flood mask | UNet(5→1) | `multimodal/multimodal_latest.pt` |
| BBDM Diffusion | 3-ch DSE SAR | 3-ch synthetic EO | LBBDM-f8 + VQGAN-F8 | `bbdm/bbdm_unet_ep*.pt` |

**Reported IoU:** Baseline ~0.4779 | Multimodal: higher (exact value from your training run)

---

## Environment Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# 2. Install PyTorch (GPU recommended — see requirements.txt for CUDA instructions)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Clone BBDM repository (required for BBDM model track)
python scripts/setup_bbdm.py
```

---

## Dataset Structure

The project uses the **Sen1Floods11** dataset with hand-labeled and weak-labeled tiles.

```bash
# Download the dataset catalog (one-time setup)
# 1. Download catalog.zip from your Google Drive (flood_project_v2/catalog.zip)
# 2. Extract to data/raw/catalog/

# 3. Download all tiles
python scripts/download_checkpoints.py --download-data --catalog data/raw/catalog/catalog
```

After download, `data/processed/` will contain:
- `S1/` — SAR GeoTIFFs, ~4831 files
- `LabelHand/` — 447 hand-labeled flood masks
- `S2Hand/` — Sentinel-2 optical (some may be missing — HTTPError handled)
- `WeakLabel/` — ~4385 Otsu weak labels

---

## Checkpoint Safety

> **The trained model weights are NOT stored in Git.**
> They live in `ml/checkpoints/` which is gitignored.
> Your primary backup is **Google Drive** (`flood_project_v2/checkpoints/`).

### Restoring checkpoints from Google Drive

```
Google Drive path → Local path
─────────────────────────────────────────────────────────────────
flood_project_v2/checkpoints/baseline_ep20.pt
    → ml/checkpoints/baseline/baseline_v1_ep20.pt

flood_project_v2/checkpoints/pix2pix_gen_ep40.pt
    → ml/checkpoints/pix2pix/pix2pix_gen_v1_ep40.pt

flood_project_v2/checkpoints/multimodal_latest.pt
    → ml/checkpoints/multimodal/multimodal_latest.pt

flood_project_v2/BBDM_VQGAN/extracted/model.ckpt
    → ml/checkpoints/bbdm/model.ckpt

(BBDM UNet if trained)
    → ml/checkpoints/bbdm/bbdm_unet_ep80.pt
```

### Verifying checkpoint integrity

```bash
# Compute and save SHA256 hashes for all checkpoint files
python scripts/download_checkpoints.py --verify
# Hashes saved to: ml/checkpoints/checkpoint_manifest.json
```

---

## Preprocessing Schemes

Two SAR normalization schemes are used. **Do NOT mix them across model tracks.**

**Scheme A** (baseline / pix2pix / multimodal):
```
SAR → clip(-30, 0) → shift by +30 → divide by 30 → [0, 1]
     then nan_to_num(nan=0, posinf=1, neginf=0)
Output: 2 channels (VV, VH)
```

**Scheme B — DSE** (BBDM):
```
VV:   clip to [-24.3028, -2.4026], scale to [0, 1]
VH:   clip to [-34.4817, -9.5345], scale to [0, 1]
Ch3:  (VV_norm + VH_norm) / 2      (mean channel)
NoData pixels set to 0
Output: 3 channels (VV_norm, VH_norm, mean)
```

---

## Running Inference

### Baseline (SAR-only) inference
```python
from ml.inference import run_baseline_inference

prob_map, binary_mask = run_baseline_inference(
    sar_path="data/processed/S1/Bolivia_103757_S1Hand.tif",
    checkpoint_path="ml/checkpoints/baseline/baseline_v1_ep20.pt",
    device="cuda",
    threshold=0.5,
)
```

### Generate Synthetic EO (Pix2Pix)
```python
from ml.inference import generate_synthetic_eo

syneo = generate_synthetic_eo(
    sar_path="data/processed/S1/Bolivia_103757_S1Hand.tif",
    checkpoint_path="ml/checkpoints/pix2pix/pix2pix_gen_v1_ep40.pt",
    device="cuda",
)
# syneo: (3, 256, 256) float32
```

### Multimodal inference (SAR + SynEO)
```python
from ml.inference import run_multimodal_inference
import numpy as np

syneo = np.load("data/processed/SynEO/Bolivia_103757_SynEO.npy")
prob_map, binary_mask = run_multimodal_inference(
    sar_path="data/processed/S1/Bolivia_103757_S1Hand.tif",
    syneo=syneo,
    checkpoint_path="ml/checkpoints/multimodal/multimodal_latest.pt",
    device="cuda",
)
```

### BBDM diffusion inference
```python
from ml.models.bbdm_wrapper import load_vqgan, load_bbdm, run_bbdm_inference
from ml.preprocessing import preprocess_sar_dse, load_tif
import numpy as np, torch

sar = load_tif("data/processed/S1/Bolivia_103757_S1Hand.tif").astype("float32")
dse_sar, _ = preprocess_sar_dse(sar)
dse_t = torch.from_numpy(dse_sar).unsqueeze(0)

vqgan      = load_vqgan(device="cuda")
latent_bbdm = load_bbdm(device="cuda")
synthetic_eo = run_bbdm_inference(dse_t, vqgan, latent_bbdm, device="cuda")
```

---

## Training

### Step 1 — Baseline
```bash
python -m ml.train_baseline
# Resume: python -m ml.train_baseline --resume ml/checkpoints/baseline/baseline_v1_ep20.pt
```

### Step 2 — Pix2Pix
```bash
python -m ml.train_pix2pix
# Resume: python -m ml.train_pix2pix --resume ml/checkpoints/pix2pix/pix2pix_gen_v1_ep40.pt
```

### Step 3 — Generate SynEO (required before multimodal training)
```bash
python scripts/generate_syneo.py --checkpoint ml/checkpoints/pix2pix/pix2pix_gen_v1_ep40.pt
```

### Step 4 — Multimodal
```bash
# Auto-resumes from multimodal_latest.pt if it exists
python -m ml.train_multimodal

# Force fresh start
python -m ml.train_multimodal --no_auto_resume --epochs 20
```

### Step 5 — BBDM (optional)
```bash
# Requires: BBDM repo cloned, VQGAN checkpoint, pre-encoded latents
python scripts/setup_bbdm.py
python -m ml.train_bbdm --epochs 80
```

---

## Evaluation

```python
from ml.inference import calculate_metrics, evaluate_iou
import numpy as np

# Per-sample metrics
metrics = calculate_metrics(pred_mask, ground_truth)
print(metrics)  # {"iou": 0.xx, "precision": 0.xx, "recall": 0.xx, "f1": 0.xx}

# Dataset-level IoU (matches notebook evaluation)
from ml.preprocessing import evaluate_iou
from ml.models.unet import UNet
import torch

model = UNet(in_ch=2, out_ch=1)
model.load_state_dict(torch.load("ml/checkpoints/baseline/baseline_v1_ep20.pt", weights_only=False))
iou = evaluate_iou(model, val_loader, device="cuda")
```

---

## GPU Requirements

- **Training:** NVIDIA GPU with ≥ 8 GB VRAM (Colab T4 used)
- **BBDM:** ≥ 16 GB VRAM recommended for full training
- **Inference:** Works on CPU (slow) or any CUDA GPU
- **PyTorch version:** ≥ 2.0.0 (BBDM requires `weights_only=False` patch for checkpoints)

---

## Known Limitations

1. **SynEO quality:** Pix2Pix produces 256×256 synthetic optical imagery. Quality is bounded by training data and the simple UNet architecture.
2. **BBDM latent training:** The BBDM pipeline requires pre-encoding all SAR/EO pairs into VQGAN latents (~4×32×32), which needs a separate preprocessing step not automated here.
3. **S2Hand availability:** Some Sentinel-2 tiles return HTTPError during download (few dozen chips). The dataset remains usable without them.
4. **Weak labels:** The Otsu-based weak labels (`S1OtsuLabelWeak`) are noisy; the baseline IoU on the hand-labeled validation set reflects this.
5. **Colab environment:** The original training used Colab-mounted Google Drive paths. All paths are now project-relative via `ml/config.py`.

---

## Backend Integration

The `ml/inference.py` module exposes clean functions for backend use:

```python
from ml.inference import (
    load_model,
    run_baseline_inference,
    generate_synthetic_eo,
    run_multimodal_inference,
    calculate_metrics,
    save_prediction,
)
```

These functions accept file paths and return numpy arrays. No notebook-specific code remains in the inference path.
