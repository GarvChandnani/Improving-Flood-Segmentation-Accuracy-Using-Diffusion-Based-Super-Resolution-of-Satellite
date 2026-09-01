"""
ml/models/unet.py — UNet architecture extracted verbatim from the FloodSense notebook.

Source: Notebook Cell 6 (THE definitions cell — "Run this every session")
Confirmed compatible with: Cell 277 checkpoint architecture audit.

IMPORTANT — DO NOT MODIFY this architecture without verifying against
existing checkpoints. The state_dict key names are fixed by this definition:

    d1.net.0.weight    [64, in_ch, 3, 3]
    d1.net.0.bias      [64]
    d1.net.1.weight    [64]  (BatchNorm)
    d1.net.3.weight    [64, 64, 3, 3]
    ...
    d2.net.0.weight    [128, 64, 3, 3]
    d3.net.0.weight    [256, 128, 3, 3]
    bottleneck.net.0.weight  [512, 256, 3, 3]
    up3.weight         [512, 256, 2, 2]
    u3.net.0.weight    [256, 512, 3, 3]
    up2.weight         [256, 128, 2, 2]
    u2.net.0.weight    [128, 256, 3, 3]
    up1.weight         [128, 64, 2, 2]
    u1.net.0.weight    [64, 128, 3, 3]
    out.weight         [out_ch, 64, 1, 1]
    out.bias           [out_ch]

Supported instantiations (all use the SAME class):
    UNet(in_ch=2, out_ch=1) — SAR-only baseline segmentation
    UNet(in_ch=2, out_ch=3) — Pix2Pix generator (SAR → synthetic EO, RGB)
    UNet(in_ch=5, out_ch=1) — Multimodal segmentation (2 SAR + 3 SynEO)
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d→BatchNorm2d→ReLU blocks.

    Exact replica of notebook Cell 6:

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """
    3-level encoder–decoder U-Net with skip connections.

    Exact replica of notebook Cell 6. Architecture:

        Encoder:
            d1: DoubleConv(in_ch → 64)
            d2: DoubleConv(64 → 128)
            d3: DoubleConv(128 → 256)
            pool: MaxPool2d(2)
            bottleneck: DoubleConv(256 → 512)

        Decoder:
            up3: ConvTranspose2d(512 → 256, stride=2)  + cat(c3) → u3: DoubleConv(512 → 256)
            up2: ConvTranspose2d(256 → 128, stride=2)  + cat(c2) → u2: DoubleConv(256 → 128)
            up1: ConvTranspose2d(128 → 64,  stride=2)  + cat(c1) → u1: DoubleConv(128 → 64)
            out: Conv2d(64 → out_ch, 1×1)

    Input/Output dimensions (for 256×256 inputs):
        Encoder: 256 → 128 → 64 → 32
        Bottleneck: 32×32
        Decoder: 32 → 64 → 128 → 256
        Output: B × out_ch × 256 × 256
    """

    def __init__(self, in_ch: int = 2, out_ch: int = 1):
        super().__init__()
        # Encoder
        self.d1 = DoubleConv(in_ch, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.u3  = DoubleConv(512, 256)   # 256+256 after cat
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.u2  = DoubleConv(256, 128)   # 128+128 after cat
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.u1  = DoubleConv(128, 64)    # 64+64 after cat
        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self.d1(x)
        p1 = self.pool(c1)
        c2 = self.d2(p1)
        p2 = self.pool(c2)
        c3 = self.d3(p2)
        p3 = self.pool(c3)
        b  = self.bottleneck(p3)
        # Decoder with skip connections
        u3 = self.u3(torch.cat([self.up3(b),  c3], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.u1(torch.cat([self.up1(u2), c1], dim=1))
        return self.out(u1)
