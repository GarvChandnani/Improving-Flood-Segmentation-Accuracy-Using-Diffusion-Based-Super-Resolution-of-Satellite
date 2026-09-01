"""
ml/models/discriminator.py — PatchDiscriminator architecture extracted verbatim
from the FloodSense notebook.

Source: Notebook Cell 6 (THE definitions cell).

IMPORTANT — DO NOT MODIFY this architecture without verifying against the
existing pix2pix discriminator checkpoint. The state_dict keys depend on
the Sequential indexing produced by this exact definition.

Architecture:
    Input:  B × 5 × H × W   (2-ch SAR concatenated with 3-ch EO or SynEO)
    Layer 0: Conv2d(5→64,  4×4, stride=2, pad=1) + LeakyReLU(0.2)       no BN
    Layer 1: Conv2d(64→128, 4×4, stride=2, pad=1) + BN + LeakyReLU(0.2)
    Layer 2: Conv2d(128→256,4×4, stride=2, pad=1) + BN + LeakyReLU(0.2)
    Layer 3: Conv2d(256→1,  4×4, pad=1)                                   raw logits

State dict keys (net.X.weight / net.X.bias):
    net.0.weight  [64,  5, 4, 4]
    net.0.bias    [64]
    net.1.weight  [64]   (BatchNorm — ABSENT because first block has norm=False)
    ... The actual indexing follows nn.Sequential flattening of the block lists.
"""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator.

    Exact replica of notebook Cell 6:

        def block(i, o, norm=True):
            layers = [nn.Conv2d(i, o, 4, stride=2, padding=1)]
            if norm: layers.append(nn.BatchNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.net = nn.Sequential(
            *block(in_ch, 64, False),
            *block(64, 128),
            *block(128, 256),
            nn.Conv2d(256, 1, 4, padding=1)
        )

    Note: `in_ch=5` matches the discriminator input used during Pix2Pix training,
    which concatenates the 2-channel SAR input with the 3-channel real/fake EO:

        torch.cat([s1, s2_real_or_fake], dim=1)  →  B × 5 × H × W

    Returns raw logits (no sigmoid). Loss uses BCEWithLogitsLoss.
    """

    def __init__(self, in_ch: int = 5):
        super().__init__()

        def block(in_c: int, out_c: int, norm: bool = True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_ch, 64, False),   # no BatchNorm on first layer
            *block(64, 128),
            *block(128, 256),
            nn.Conv2d(256, 1, 4, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
