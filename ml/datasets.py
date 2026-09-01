"""
ml/datasets.py — PyTorch Dataset classes extracted from the FloodSense notebook.

Source cells:
    Cell 16  → FloodDataset    (SAR-only, baseline training)
    Cell 20  → PairedDataset   (SAR + S2Hand, pix2pix training)
    Cell 10  → MultiModalDataset (SAR + SynEO, multimodal training)
    Cell 95  → PairedLatentDataset (pre-encoded latents, BBDM training)

All datasets preserve the EXACT preprocessing behavior from the notebook.
The normalization schemes are NOT interchangeable between model tracks.

Data layout expected:
    data/processed/
        S1/           <country>_<chip_id>_S1Hand.tif   (SAR hand-labeled)
                      <country>_<chip_id>_S1Weak.tif   (SAR weak-labeled)
        LabelHand/    <country>_<chip_id>_LabelHand.tif
        S2Hand/       <country>_<chip_id>_S2Hand.tif
        WeakLabel/    <country>_<chip_id>_S1OtsuLabelWeak.tif
        SynEO/        <country>_<chip_id>_SynEO.npy
        BBDM_latents/
            train/    <sample_id>_dse.pt  and  <sample_id>_eo.pt
            test/     <sample_id>_dse.pt  and  <sample_id>_eo.pt
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ml.preprocessing import (
    load_tif,
    preprocess_sar_simple,
    preprocess_s2_simple,
)
from ml.config import cfg


# ---------------------------------------------------------------------------
# FloodDataset — SAR-only segmentation (Cell 16)
# ---------------------------------------------------------------------------

class FloodDataset(Dataset):
    """
    SAR-only dataset for baseline U-Net training.

    Exact replica of notebook Cell 16:

        class FloodDataset(Dataset):
            def __init__(self, id_list, s1_suffix, label_dir, label_suffix, patch_size=256):
                ...
            def __getitem__(self, idx):
                s1  = load_tif(...).astype(np.float32)
                lbl = load_tif(...).astype(np.float32)[0]
                s1  = np.nan_to_num((np.clip(s1, -30, 0) + 30) / 30)
                s1  = F.interpolate(..., mode='bilinear').squeeze(0)
                lbl = F.interpolate(..., mode='nearest').squeeze(0).squeeze(0)
                return s1, lbl

    Returns:
        (s1_tensor, label_tensor)
            s1_tensor:    float32 (2, patch_size, patch_size) — normalized SAR
            label_tensor: float32 (patch_size, patch_size) — flood mask
                          Values: 0 = non-flood, 1 = flood, -1 = NoData
    """

    def __init__(
        self,
        id_list: List[Tuple[str, str]],
        s1_suffix: str,
        label_dir: str,
        label_suffix: str,
        patch_size: int = 256,
        s1_root: Optional[Path] = None,
        label_root: Optional[Path] = None,
    ):
        """
        Args:
            id_list:      List of (country, chip_id) tuples.
            s1_suffix:    Filename suffix for SAR tiles, e.g. "S1Hand" or "S1Weak".
            label_dir:    Subdirectory name containing label files.
            label_suffix: Filename suffix for label tiles.
            patch_size:   Spatial size after interpolation (default 256).
            s1_root:      Root directory for S1 files. Defaults to cfg.S1_DIR.
            label_root:   Root directory for label files. Defaults to cfg.PROCESSED_DIR.
        """
        self.id_list      = id_list
        self.s1_suffix    = s1_suffix
        self.label_dir    = label_dir
        self.label_suffix = label_suffix
        self.patch_size   = patch_size
        self.s1_root      = Path(s1_root) if s1_root else cfg.S1_DIR
        self.label_root   = Path(label_root) if label_root else cfg.PROCESSED_DIR

    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        country, chip_id = self.id_list[idx]

        # Load SAR (2, H, W) — Scheme A normalization
        s1_path = self.s1_root / f"{country}_{chip_id}_{self.s1_suffix}.tif"
        s1 = load_tif(str(s1_path)).astype(np.float32)
        s1 = preprocess_sar_simple(s1)  # (clip(-30,0)+30)/30 + nan_to_num

        # Load label (H, W)
        lbl_path = self.label_root / self.label_dir / f"{country}_{chip_id}_{self.label_suffix}.tif"
        lbl = load_tif(str(lbl_path)).astype(np.float32)[0]  # take band 0

        # Resize to patch_size
        s1  = F.interpolate(
            torch.from_numpy(s1).unsqueeze(0),
            size=(self.patch_size,) * 2,
            mode="bilinear",
        ).squeeze(0)

        lbl = F.interpolate(
            torch.from_numpy(lbl).unsqueeze(0).unsqueeze(0),
            size=(self.patch_size,) * 2,
            mode="nearest",
        ).squeeze(0).squeeze(0)

        return s1, lbl


# ---------------------------------------------------------------------------
# PairedDataset — SAR + S2 for Pix2Pix training (Cell 20)
# ---------------------------------------------------------------------------

class PairedDataset(Dataset):
    """
    Paired SAR + Sentinel-2 dataset for Pix2Pix (SAR → synthetic EO) training.

    Exact replica of notebook Cell 20:

        class PairedDataset(Dataset):
            def __getitem__(self, idx):
                s1 = load_tif(...).astype(np.float32)
                s2 = load_tif(...).astype(np.float32)
                s1 = np.nan_to_num((np.clip(s1, -30, 0) + 30) / 30)
                s2 = np.nan_to_num(np.clip(s2, 0, 3000) / 3000)
                s2 = s2[[3,2,1],:,:] if s2.shape[0] >= 4 else s2[:3,:,:]
                s1 = F.interpolate(..., mode='bilinear').squeeze(0)
                s2 = F.interpolate(..., mode='bilinear').squeeze(0)
                return s1, s2

    Returns:
        (s1_tensor, s2_tensor)
            s1_tensor: float32 (2, patch_size, patch_size) — SAR input
            s2_tensor: float32 (3, patch_size, patch_size) — EO target (RGB)
    """

    def __init__(
        self,
        id_list: List[Tuple[str, str]],
        patch_size: int = 256,
        s1_root: Optional[Path] = None,
        s2_root: Optional[Path] = None,
    ):
        self.id_list    = id_list
        self.patch_size = patch_size
        self.s1_root    = Path(s1_root) if s1_root else cfg.S1_DIR
        self.s2_root    = Path(s2_root) if s2_root else cfg.S2_HAND

    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        country, chip_id = self.id_list[idx]

        # SAR (2, H, W)
        s1_path = self.s1_root / f"{country}_{chip_id}_S1Hand.tif"
        s1 = load_tif(str(s1_path)).astype(np.float32)
        s1 = preprocess_sar_simple(s1)

        # Sentinel-2 optical (3, H, W) — bands [3,2,1] (NIR, Red, Green)
        s2_path = self.s2_root / f"{country}_{chip_id}_S2Hand.tif"
        s2 = load_tif(str(s2_path)).astype(np.float32)
        s2 = preprocess_s2_simple(s2)

        # Resize
        s1 = F.interpolate(
            torch.from_numpy(s1).unsqueeze(0),
            size=(self.patch_size,) * 2,
            mode="bilinear",
        ).squeeze(0)

        s2 = F.interpolate(
            torch.from_numpy(s2).unsqueeze(0),
            size=(self.patch_size,) * 2,
            mode="bilinear",
        ).squeeze(0)

        return s1, s2


# ---------------------------------------------------------------------------
# MultiModalDataset — SAR + SynEO for multimodal training (Cell 10)
# ---------------------------------------------------------------------------

class MultiModalDataset(Dataset):
    """
    Multimodal dataset combining SAR with pix2pix-generated SynEO.

    Exact replica of notebook Cell 10:

        class MultiModalDataset(Dataset):
            def __getitem__(self, idx):
                s1    = load_tif(...).astype(np.float32)
                lbl   = load_tif(...).astype(np.float32)[0]
                syneo = np.load(..._SynEO.npy)

                s1 = np.nan_to_num((np.clip(s1, -30, 0) + 30) / 30)

                s1_t  = F.interpolate(..., mode='bilinear').squeeze(0)
                lbl_t = F.interpolate(..., mode='nearest').squeeze(0).squeeze(0)

                return torch.cat([s1_t, torch.from_numpy(syneo)], dim=0), lbl_t

    SynEO .npy files are expected shape: (3, 256, 256) as saved by the pix2pix
    generation step (notebook Cell 24 / scripts/generate_syneo.py).

    Returns:
        (x_tensor, label_tensor)
            x_tensor:     float32 (5, patch_size, patch_size) — [SAR(2) | SynEO(3)]
            label_tensor: float32 (patch_size, patch_size)
    """

    def __init__(
        self,
        id_list: List[Tuple[str, str]],
        s1_suffix: str,
        label_dir: str,
        label_suffix: str,
        patch_size: int = 256,
        s1_root: Optional[Path] = None,
        label_root: Optional[Path] = None,
        syneo_root: Optional[Path] = None,
    ):
        self.id_list      = id_list
        self.s1_suffix    = s1_suffix
        self.label_dir    = label_dir
        self.label_suffix = label_suffix
        self.patch_size   = patch_size
        self.s1_root      = Path(s1_root)    if s1_root    else cfg.S1_DIR
        self.label_root   = Path(label_root) if label_root else cfg.PROCESSED_DIR
        self.syneo_root   = Path(syneo_root) if syneo_root else cfg.SYNEO_DIR

    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        country, chip_id = self.id_list[idx]

        # SAR (2, H, W)
        s1_path = self.s1_root / f"{country}_{chip_id}_{self.s1_suffix}.tif"
        s1 = load_tif(str(s1_path)).astype(np.float32)
        s1 = preprocess_sar_simple(s1)

        # Label (H, W)
        lbl_path = self.label_root / self.label_dir / f"{country}_{chip_id}_{self.label_suffix}.tif"
        lbl = load_tif(str(lbl_path)).astype(np.float32)[0]

        # SynEO (3, 256, 256) — already at patch_size, saved by generate_syneo.py
        syneo_path = self.syneo_root / f"{country}_{chip_id}_SynEO.npy"
        syneo = np.load(str(syneo_path))

        # Resize SAR and label
        s1_t = F.interpolate(
            torch.from_numpy(s1).unsqueeze(0),
            size=(self.patch_size,) * 2,
            mode="bilinear",
        ).squeeze(0)

        lbl_t = F.interpolate(
            torch.from_numpy(lbl).unsqueeze(0).unsqueeze(0),
            size=(self.patch_size,) * 2,
            mode="nearest",
        ).squeeze(0).squeeze(0)

        # Concatenate SAR (2 ch) + SynEO (3 ch) → 5 channels
        x = torch.cat([s1_t, torch.from_numpy(syneo)], dim=0)

        return x, lbl_t


# ---------------------------------------------------------------------------
# PairedLatentDataset — Pre-encoded VQGAN latents for BBDM training (Cell 95)
# ---------------------------------------------------------------------------

class PairedLatentDataset(Dataset):
    """
    Loads pre-encoded VQGAN latents (4×32×32) for BBDM training.

    Exact replica of notebook Cell 95:

        class PairedLatentDataset(Dataset):
            def __init__(self, directory):
                dse_files = glob.glob(os.path.join(directory, "*_dse.pt"))
                eo_files  = glob.glob(os.path.join(directory, "*_eo.pt"))
                # match by sample_id = filename[:-8]  (strip "_dse.pt" or "_eo.pt")
                common_ids = sorted(set(dse_map.keys()) & set(eo_map.keys()))
                self.samples = [(sample_id, dse_map[...], eo_map[...])]

    Expected filenames:
        <sample_id>_dse.pt  — DSE SAR latent, shape (4, 32, 32)
        <sample_id>_eo.pt   — EO latent,      shape (4, 32, 32)

    Returns:
        {"dse": tensor, "eo": tensor, "id": str}
    """

    def __init__(self, directory: str):
        self.directory = directory

        dse_files = glob.glob(os.path.join(directory, "*_dse.pt"))
        eo_files  = glob.glob(os.path.join(directory, "*_eo.pt"))

        dse_map: dict = {}
        eo_map:  dict = {}

        for path in dse_files:
            filename  = os.path.basename(path)
            sample_id = filename[:-7]   # strip "_dse.pt" (7 chars)
            dse_map[sample_id] = path

        for path in eo_files:
            filename  = os.path.basename(path)
            sample_id = filename[:-6]   # strip "_eo.pt" (6 chars)
            eo_map[sample_id] = path

        common_ids = sorted(set(dse_map.keys()) & set(eo_map.keys()))

        if len(common_ids) == 0:
            raise RuntimeError(
                f"No paired latent files found in: {directory}\n"
                "Expected files named: <id>_dse.pt and <id>_eo.pt"
            )

        self.samples = [
            (sample_id, dse_map[sample_id], eo_map[sample_id])
            for sample_id in common_ids
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_id, dse_path, eo_path = self.samples[idx]

        dse = torch.load(dse_path, map_location="cpu", weights_only=True)
        eo  = torch.load(eo_path,  map_location="cpu", weights_only=True)

        return {
            "dse": dse,
            "eo":  eo,
            "id":  sample_id,
        }


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def make_train_val_ids(
    weak_source: list,
    hand_source: list,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Build train/val ID lists from catalog asset lists.

    Exact replica of notebook Cell 16:
        train_ids = [(c, cid) for c, cid, _ in weak_source]
        val_ids   = [(c, cid) for c, cid, _ in hand_source]

    Args:
        weak_source: Output of collect_all_assets() for weak-labeled source.
        hand_source: Output of collect_all_assets() for hand-labeled source.

    Returns:
        (train_ids, val_ids) — lists of (country, chip_id) tuples.
    """
    train_ids = [(c, cid) for c, cid, _ in weak_source]
    val_ids   = [(c, cid) for c, cid, _ in hand_source]
    return train_ids, val_ids


def make_paired_ids(
    val_ids: List[Tuple[str, str]],
    s2_root: Optional[Path] = None,
) -> List[Tuple[str, str]]:
    """
    Filter val_ids to those with an existing S2Hand file (for pix2pix training).

    Exact replica of notebook Cell 20:
        paired_ids = [(c, cid) for c, cid in val_ids
                      if os.path.exists(f"{LOCAL}/data/S2Hand/{c}_{cid}_S2Hand.tif")]

    Args:
        val_ids: List of (country, chip_id) tuples.
        s2_root: Directory containing S2Hand .tif files. Defaults to cfg.S2_HAND.

    Returns:
        Filtered list of (country, chip_id) tuples that have an S2 file.
    """
    root = Path(s2_root) if s2_root else cfg.S2_HAND
    return [
        (c, cid) for c, cid in val_ids
        if (root / f"{c}_{cid}_S2Hand.tif").exists()
    ]
