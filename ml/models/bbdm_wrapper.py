"""
ml/models/bbdm_wrapper.py — Clean wrapper around the external BBDM repository
for the FloodSense project.

Source: Notebook Cells 35, 41, 44-50, 57-60, 68.

The BBDM library is NOT reimplemented here.
This module wraps the external `xuekt98/BBDM` repository, which must be
cloned to the path configured in ml/config.py (BBDM_REPO_DIR).

To set up the repository:
    python scripts/setup_bbdm.py

Architecture confirmed from notebook Cell 57 and Cell 58:
    VQGAN-F8:
        Input:   B × 3 × 256 × 256  (3-ch DSE SAR or EO)
        Latent:  B × 4 ×  32 ×  32  (VQGAN F8 bottleneck)
        Output:  B × 3 × 256 × 256  (reconstructed)

    BBDM UNet (operates in latent space):
        in_channels  = 4     (latent channels)
        out_channels = 4
        image_size   = 32    (latent spatial size)
        model_channels = 128
        channel_mult   = (1, 4, 8)
        num_res_blocks = 2
        attention_resolutions = (32, 16, 8)
        num_heads = 8
        num_head_channels = 64
        use_scale_shift_norm = True
        resblock_updown = True

    Brownian Bridge:
        num_timesteps = 1000
        mt_type       = "linear"
        objective     = "grad"
        loss_type     = "l1"
        sample_step   = 200
        eta           = 1.0
        max_var       = 1.0

IMPORTANT WEIGHT-SAFETY NOTES:
    - The VQGAN checkpoint (model.ckpt) is loaded with weights_only=False
      because it contains non-tensor objects (PyTorch 2.6+ requires explicit opt-in).
    - The VQModel.init_from_ckpt is patched to use weights_only=False
      (from notebook Cell 44).
    - No architecture changes are made. Only the checkpoint path is updated.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# BBDMConfig: SimpleNamespace with __contains__ support
# ---------------------------------------------------------------------------
# Required because BrownianBridgeModel.py uses both:
#   vars(model_params.UNetParams)   — needs attribute access
#   "max_var" in model_params       — needs __contains__
# (from notebook Cell 57)
# ---------------------------------------------------------------------------

class BBDMConfig(SimpleNamespace):
    """
    Configuration container compatible with the BBDM repository.

    Supports both attribute access (config.foo) and membership test
    ('foo' in config), as required by BrownianBridgeModel internals.
    """

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


# ---------------------------------------------------------------------------
# BBDM repository path management
# ---------------------------------------------------------------------------

def _ensure_bbdm_on_path(bbdm_dir: Optional[Path] = None) -> Path:
    """
    Ensure the BBDM repository is on sys.path.
    Returns the repository root directory.
    """
    from ml.config import cfg
    repo = Path(bbdm_dir) if bbdm_dir else cfg.BBDM_REPO_DIR

    if not repo.exists():
        raise RuntimeError(
            f"BBDM repository not found at: {repo}\n"
            "Run:  python scripts/setup_bbdm.py\n"
            "to clone the repository."
        )

    repo_str = str(repo)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)

    # Clear any stale cached imports of BBDM's 'datasets' package
    # so it doesn't conflict with HuggingFace's 'datasets' package
    for name in list(sys.modules.keys()):
        if name == "datasets" or name.startswith("datasets."):
            del sys.modules[name]

    return repo


# ---------------------------------------------------------------------------
# VQGAN-F8 configuration (from notebook Cell 57)
# ---------------------------------------------------------------------------

def _make_vqgan_config(vqgan_ckpt_path: str) -> BBDMConfig:
    """
    Build VQGAN-F8 configuration matching the model.ckpt checkpoint.
    Values are exact replicas of notebook Cell 57.
    """
    ddconfig = BBDMConfig(
        double_z=False,
        z_channels=4,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(32,),
        dropout=0.0,
    )

    lossconfig = BBDMConfig(
        target="torch.nn.Identity",
    )

    return BBDMConfig(
        ckpt_path=str(vqgan_ckpt_path),
        embed_dim=4,
        n_embed=16384,
        ddconfig=ddconfig,
        lossconfig=lossconfig,
    )


# ---------------------------------------------------------------------------
# BBDM UNet + Brownian Bridge configuration (from notebook Cell 57)
# ---------------------------------------------------------------------------

def _make_model_config(vqgan_ckpt_path: str) -> BBDMConfig:
    """
    Build the complete LatentBrownianBridgeModel configuration.
    All values are exact replicas of notebook Cell 57.

    VQGAN-F8 latent shape: B × 4 × 32 × 32
    UNet operates in this latent space.
    """
    unet_params = BBDMConfig(
        image_size=32,
        in_channels=4,
        model_channels=128,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=(32, 16, 8),
        channel_mult=(1, 4, 8),
        conv_resample=True,
        dims=2,
        num_heads=8,
        num_head_channels=64,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_spatial_transformer=False,
        context_dim=None,
        condition_key="nocond",
    )

    bb_params = BBDMConfig(
        mt_type="linear",
        objective="grad",
        loss_type="l1",
        skip_sample=True,
        sample_type="linear",
        sample_step=200,
        num_timesteps=1000,
        eta=1.0,
        max_var=1.0,
        UNetParams=unet_params,
    )

    ema_config = BBDMConfig(
        use_ema=True,
        ema_decay=0.995,
        update_ema_interval=8,
        start_ema_step=30000,
    )

    cond_stage_params = BBDMConfig(
        n_stages=3,
        in_channels=3,
        out_channels=4,
    )

    vqgan_params = _make_vqgan_config(vqgan_ckpt_path)

    return BBDMConfig(
        model_name="LBBDM-f8",
        model_type="LBBDM",
        latent_before_quant_conv=False,
        normalize_latent=False,
        only_load_latent_mean_std=False,
        CondStageParams=cond_stage_params,
        VQGAN=BBDMConfig(params=vqgan_params),
        BB=BBDMConfig(params=bb_params),
        EMA=ema_config,
    )


# ---------------------------------------------------------------------------
# VQGAN checkpoint loader patch (from notebook Cell 44)
# ---------------------------------------------------------------------------

def _patch_vqmodel_loader():
    """
    Patch VQModel.init_from_ckpt to use weights_only=False.

    Required for PyTorch >= 2.6 compatibility with the VQGAN checkpoint,
    which contains non-tensor objects that are blocked by the new default
    weights_only=True behaviour.

    This is taken verbatim from notebook Cell 44.
    """
    try:
        from taming.models.vqgan import VQModel
    except ImportError:
        return  # BBDM repo not available yet

    def _compatible_init_from_ckpt(self, path, ignore_keys=None):
        if ignore_keys is None:
            ignore_keys = []
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"[bbdm_wrapper] Restored VQGAN from {path}")

    VQModel.init_from_ckpt = _compatible_init_from_ckpt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_vqgan(
    vqgan_ckpt: Optional[str] = None,
    device: Optional[str] = None,
    bbdm_repo: Optional[Path] = None,
) -> nn.Module:
    """
    Load the VQGAN-F8 encoder/decoder.

    Args:
        vqgan_ckpt: Path to the VQGAN checkpoint (model.ckpt).
                    Defaults to cfg.VQGAN_CKPT.
        device:     Target device. Defaults to cfg.DEVICE.
        bbdm_repo:  Path to BBDM repository. Defaults to cfg.BBDM_REPO_DIR.

    Returns:
        Loaded VQModel in eval mode on the specified device.

    Raises:
        RuntimeError if BBDM repo is missing.
        FileNotFoundError if checkpoint is missing.
    """
    from ml.config import cfg

    ckpt = Path(vqgan_ckpt) if vqgan_ckpt else cfg.VQGAN_CKPT
    dev  = device or cfg.DEVICE

    if not ckpt.exists():
        raise FileNotFoundError(
            f"VQGAN checkpoint not found: {ckpt}\n"
            "Copy model.ckpt to ml/checkpoints/bbdm/model.ckpt"
        )

    _ensure_bbdm_on_path(bbdm_repo)
    _patch_vqmodel_loader()

    from taming.models.vqgan import VQModel

    vqgan_params = _make_vqgan_config(str(ckpt))

    vqgan = VQModel(
        ddconfig=vqgan_params.ddconfig,
        lossconfig=vqgan_params.lossconfig,
        n_embed=vqgan_params.n_embed,
        embed_dim=vqgan_params.embed_dim,
        ckpt_path=str(ckpt),
    )

    vqgan = vqgan.to(dev).eval()

    # Freeze all VQGAN parameters — it is a fixed feature extractor
    for p in vqgan.parameters():
        p.requires_grad_(False)

    return vqgan


def load_bbdm(
    vqgan_ckpt: Optional[str] = None,
    bbdm_ckpt: Optional[str] = None,
    device: Optional[str] = None,
    bbdm_repo: Optional[Path] = None,
) -> nn.Module:
    """
    Instantiate the LatentBrownianBridgeModel with VQGAN and BBDM UNet.

    Args:
        vqgan_ckpt: Path to VQGAN checkpoint. Defaults to cfg.VQGAN_CKPT.
        bbdm_ckpt:  Optional path to a trained BBDM UNet checkpoint.
                    If None, returns a freshly initialised model.
        device:     Target device. Defaults to cfg.DEVICE.
        bbdm_repo:  Path to BBDM repository.

    Returns:
        LatentBrownianBridgeModel in eval mode.
    """
    from ml.config import cfg

    ckpt_v = Path(vqgan_ckpt) if vqgan_ckpt else cfg.VQGAN_CKPT
    dev    = device or cfg.DEVICE

    _ensure_bbdm_on_path(bbdm_repo)
    _patch_vqmodel_loader()

    from model.BrownianBridge.LatentBrownianBridgeModel import (
        LatentBrownianBridgeModel,
    )

    model_config = _make_model_config(str(ckpt_v))

    latent_bbdm = LatentBrownianBridgeModel(model_config).to(dev)
    latent_bbdm.eval()

    if bbdm_ckpt is not None:
        bbdm_ckpt = Path(bbdm_ckpt)
        if not bbdm_ckpt.exists():
            raise FileNotFoundError(f"BBDM checkpoint not found: {bbdm_ckpt}")
        state = torch.load(str(bbdm_ckpt), map_location=dev, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        latent_bbdm.load_state_dict(state, strict=False)
        print(f"[bbdm_wrapper] Loaded BBDM checkpoint: {bbdm_ckpt}")

    return latent_bbdm


def encode_sar_to_latent(
    vqgan: nn.Module,
    dse_sar: torch.Tensor,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Encode a 3-channel DSE SAR tensor into VQGAN latent space.

    Args:
        vqgan:   Loaded VQModel (from load_vqgan()).
        dse_sar: B × 3 × 256 × 256 DSE-preprocessed SAR tensor.
                 (VV-norm, VH-norm, mean-channel)
        device:  Target device.

    Returns:
        Latent tensor: B × 4 × 32 × 32  (pre-quantized, continuous latent)

    Note: Uses the encoder + quant_conv path (without full quantization),
    matching notebook Cell 56 and the BBDM training pipeline.
    """
    from ml.config import cfg
    dev = device or cfg.DEVICE

    vqgan.eval()
    dse_sar = dse_sar.to(dev)

    with torch.no_grad():
        latent = vqgan.encoder(dse_sar)
        latent = vqgan.quant_conv(latent)

    return latent  # B × 4 × 32 × 32


def decode_latent_to_eo(
    vqgan: nn.Module,
    latent: torch.Tensor,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Decode a VQGAN latent back to pixel space.

    Args:
        vqgan:  Loaded VQModel.
        latent: B × 4 × 32 × 32 latent tensor (continuous or quantized).
        device: Target device.

    Returns:
        Reconstructed image: B × 3 × 256 × 256

    Note: Applies quantization before decoding (full encode→quantize→decode path),
    matching notebook Cell 56.
    """
    from ml.config import cfg
    dev = device or cfg.DEVICE

    vqgan.eval()
    latent = latent.to(dev)

    with torch.no_grad():
        quantized, _, _ = vqgan.quantize(latent)
        reconstruction  = vqgan.decode(quantized)

    return reconstruction  # B × 3 × 256 × 256


def run_bbdm_inference(
    sar_dse: torch.Tensor,
    vqgan: nn.Module,
    latent_bbdm: nn.Module,
    num_samples: int = 1,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Run full BBDM inference: DSE SAR → Synthetic EO via latent diffusion.

    Args:
        sar_dse:      B × 3 × 256 × 256 DSE-preprocessed SAR tensor.
        vqgan:        Loaded VQModel.
        latent_bbdm:  Loaded LatentBrownianBridgeModel.
        num_samples:  Number of diffusion samples to average.
        device:       Target device.

    Returns:
        Synthetic EO: B × 3 × 256 × 256 (pixel-space, float32, [0, 1] range)
    """
    from ml.config import cfg
    dev = device or cfg.DEVICE

    vqgan.eval()
    latent_bbdm.eval()

    sar_dse = sar_dse.to(dev)

    with torch.no_grad():
        # Encode conditioning SAR into latent space
        dse_latent = encode_sar_to_latent(vqgan, sar_dse, dev)

        # Run Brownian Bridge diffusion sampling
        # The BBDM model's sample() interface takes the conditioning latent
        outputs = []
        for _ in range(num_samples):
            sample_output = latent_bbdm.sample(
                y=dse_latent,
                clip_denoised=True,
            )
            # sample_output may be a tuple or tensor depending on BBDM version
            if isinstance(sample_output, (tuple, list)):
                sample_latent = sample_output[0]
            else:
                sample_latent = sample_output

            # Decode the sampled latent back to pixel space
            synthetic_eo = decode_latent_to_eo(vqgan, sample_latent, dev)
            outputs.append(synthetic_eo)

        if num_samples > 1:
            synthetic_eo = torch.stack(outputs, dim=0).mean(dim=0)
        else:
            synthetic_eo = outputs[0]

    # Clamp to valid image range
    synthetic_eo = synthetic_eo.clamp(0.0, 1.0)

    return synthetic_eo


def get_bbdm_config_summary() -> dict:
    """Return a human-readable summary of the BBDM configuration."""
    return {
        "model_type": "LBBDM (Latent Brownian Bridge Diffusion Model)",
        "vqgan": {
            "architecture": "VQGAN-F8",
            "input_channels": 3,
            "output_channels": 3,
            "latent_channels": 4,
            "latent_size": 32,
            "input_size": 256,
            "embed_dim": 4,
            "n_embed": 16384,
        },
        "bbdm_unet": {
            "in_channels": 4,
            "out_channels": 4,
            "image_size": 32,
            "model_channels": 128,
            "channel_mult": [1, 4, 8],
            "num_res_blocks": 2,
            "num_heads": 8,
        },
        "brownian_bridge": {
            "num_timesteps": 1000,
            "mt_type": "linear",
            "objective": "grad",
            "loss_type": "l1",
            "sample_step": 200,
            "eta": 1.0,
            "max_var": 1.0,
        },
    }
