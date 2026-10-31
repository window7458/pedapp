"""Frozen foundation-model (FM) encoders that return a dense feature map at stride 16.

Every encoder takes a batch of RGB images in [0, 1] with shape (B, 3, H, W), where H and W are
multiples of 32 (true for every YOLO batch), and returns (B, D, H/16, W/16).
Patch-14 models (DINOv2) are resized internally to (H/16*14, W/16*14) so all grids line up with
YOLO's P4 (stride 16).

Encoders are kept in a process-wide registry, NOT inside the YOLO nn.Module, so that
  * Ultralytics' trainer never re-enables their gradients or puts them in the optimizer,
  * they are not written into every checkpoint / EMA copy (86M-400M params each).

Usage:
    enc = get_encoder("dinov3_b+siglip2_b", device)    # "+" = channel-concat fusion
    feat = enc(x01)                                     # (B, D, H/16, W/16), float32
    python encoders.py --prefetch all                   # download all weights ahead of time
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# name -> (loader kind, source id, feature dim)
SPECS = {
    "siglip2_b": ("siglip", "google/siglip2-base-patch16-512", 768),
    "dinov2_b": ("dinov2", "facebook/dinov2-base", 768),
    "dinov3_b": ("dinov3", "facebook/dinov3-vitb16-pretrain-lvd1689m", 768),  # gated: accept license on HF
    "radio_v2.5-b": ("radio", "radio_v2.5-b", 768),  # NVIDIA non-commercial license
    "c-radio_v3-b": ("radio", "c-radio_v3-b", 768),  # NVIDIA Open Model License (commercial ok)
    "c-radio_v4-so400m": ("radio", "c-radio_v4-so400m", 1152),
    # test-only encoder: random-init small ViT, no download (used by selftest.py)
    "dummy": ("dummy", "", 64),
}

_REGISTRY: dict[str, "FMEncoder"] = {}
RANDOM_INIT = os.environ.get("FM_RANDOM_INIT", "0") == "1"  # selftest: build from config, no weights
RADIO_HF = {"radio_v2.5-b": "nvidia/RADIO-B", "c-radio_v3-b": "nvidia/C-RADIOv3-B",
            "c-radio_v4-so400m": "nvidia/C-RADIOv4-SO400M"}


def feat_dim(name: str) -> int:
    return sum(SPECS[n][2] for n in name.split("+"))


class _Single(nn.Module):
    """One backbone + its preprocessing. forward(x01) -> (B, D, H/16, W/16)."""

    def __init__(self, name: str):
        super().__init__()
        kind, src, self.dim = SPECS[name]
        self.name, self.kind = name, kind
        self.patch = 16
        self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        self.n_prefix = 0

        if kind == "siglip":
            from transformers import SiglipVisionConfig, SiglipVisionModel

            self.model = (
                SiglipVisionModel(SiglipVisionConfig(hidden_size=768, num_hidden_layers=2, num_attention_heads=12,
                                                     intermediate_size=512, image_size=512, patch_size=16))
                if RANDOM_INIT
                else SiglipVisionModel.from_pretrained(src)
            )
            self.mean, self.std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        elif kind == "dinov2":
            from transformers import Dinov2Config, Dinov2Model

            self.model = (
                Dinov2Model(Dinov2Config(hidden_size=768, num_hidden_layers=2, num_attention_heads=12,
                                         image_size=518, patch_size=14))
                if RANDOM_INIT
                else Dinov2Model.from_pretrained(src)
            )
            self.patch = 14
            self.n_prefix = 1  # CLS
        elif kind == "dinov3":
            from transformers import DINOv3ViTConfig, DINOv3ViTModel

            self.model = (
                DINOv3ViTModel(DINOv3ViTConfig(hidden_size=768, num_hidden_layers=2, num_attention_heads=12,
                                               intermediate_size=512, patch_size=16, num_register_tokens=4))
                if RANDOM_INIT
                else DINOv3ViTModel.from_pretrained(src)
            )
            self.n_prefix = 1 + int(getattr(self.model.config, "num_register_tokens", 0))  # CLS + registers
        elif kind == "radio":
            if RANDOM_INIT:
                raise RuntimeError("RADIO has no random-init path; run it in sanity.py on the GPU server.")
            # RADIO normalises internally and expects [0, 1] input.
            try:
                self.model = torch.hub.load("NVlabs/RADIO", "radio_model", version=src, progress=True,
                                            skip_validation=True, trust_repo=True)
            except Exception as e:  # e.g. licence-gated download: fall back to the HF copy (uses HF_TOKEN)
                from transformers import AutoModel

                repo = RADIO_HF[src]
                print(f"[FM] torch.hub failed for {src} ({type(e).__name__}); trying HF {repo}", flush=True)
                self.model = AutoModel.from_pretrained(repo, trust_remote_code=True).radio_model
            self.mean = self.std = None
        elif kind == "dummy":
            self.model = nn.Conv2d(3, self.dim, 16, 16)  # patchify only
            self.mean = self.std = None
        else:
            raise ValueError(kind)

        self.model.eval().requires_grad_(False)
        if self.mean is not None:
            self.register_buffer("_m", torch.tensor(self.mean).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("_s", torch.tensor(self.std).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x01.shape
        gh, gw = H // 16, W // 16  # target grid (YOLO P4)
        x = x01
        if self.patch != 16:
            x = F.interpolate(x, size=(gh * self.patch, gw * self.patch), mode="bilinear", align_corners=False)
        if self.mean is not None:
            x = (x - self._m.to(x.dtype)) / self._s.to(x.dtype)

        if self.kind == "siglip":
            tok = self.model(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state  # no CLS token
        elif self.kind in ("dinov2", "dinov3"):
            tok = self.model(pixel_values=x).last_hidden_state[:, self.n_prefix:]
        elif self.kind == "radio":
            return self.model(x, feature_fmt="NCHW")[1].float()  # RadioOutput(summary, features)
        else:  # dummy
            return self.model(x).float()

        return tok.transpose(1, 2).reshape(B, tok.shape[-1], gh, gw).float()


class FMEncoder(nn.Module):
    """One encoder, or several concatenated along channels ("a+b")."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.parts = nn.ModuleList(_Single(n) for n in name.split("+"))
        self.dim = sum(p.dim for p in self.parts)

    @torch.no_grad()
    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        feats = [p(x01) for p in self.parts]
        if len(feats) == 1:
            return feats[0]
        size = feats[0].shape[-2:]
        feats = [f if f.shape[-2:] == size else F.interpolate(f, size=size, mode="bilinear") for f in feats]
        return torch.cat(feats, 1)


def get_encoder(name: str, device: torch.device | str) -> FMEncoder:
    """Load once per process, cache, and return an eval-mode frozen encoder on `device`."""
    device = torch.device(device)
    enc = _REGISTRY.get(name)
    if enc is None:
        t = time.time()
        enc = FMEncoder(name).eval()
        _REGISTRY[name] = enc
        print(f"[FM] loaded {name} (D={enc.dim}) in {time.time() - t:.1f}s", flush=True)
    if next(enc.parameters()).device != device:
        enc.to(device)
    return enc


def encode(name: str, x01: torch.Tensor) -> torch.Tensor:
    """Run the named encoder on a YOLO batch. Uses fp16 autocast on CUDA."""
    enc = get_encoder(name, x01.device)
    with torch.no_grad(), torch.autocast(device_type=x01.device.type, dtype=torch.float16,
                                         enabled=x01.device.type == "cuda"):
        return enc(x01.float()).float()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefetch", default="", help="'all' or comma list of encoder names")
    a = ap.parse_args()
    names = [n for n in SPECS if n != "dummy"] if a.prefetch == "all" else a.prefetch.split(",")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for n in names:
        try:
            f = encode(n, torch.rand(1, 3, 640, 640, device=dev))
            print(f"  OK  {n:20s} -> {tuple(f.shape)}")
        except Exception as e:  # keep going so one gated model does not hide the others
            print(f"  FAIL {n:20s} -> {type(e).__name__}: {e}")
        _REGISTRY.pop(n, None)
        if dev == "cuda":
            torch.cuda.empty_cache()
