"""YOLO11-seg + frozen foundation-model features, for Ultralytics 8.3/8.4.

Two ways of using the FM (set per run with env vars, see train_one.py):

  fm_mode="fusion"   Track A: FM features are ADDED into the backbone outputs P3/P4/P5 (layers 4, 6, 10)
                     through a zero-initialised 1x1 adapter, so training starts exactly at the
                     pretrained YOLO11s-seg and the FM runs at inference too.
  fm_mode="distill"  Track B: the FM is only a teacher. Small 1x1 heads project P3/P4/P5 into the FM
                     space and a cosine loss pulls them toward the FM features. At inference the model
                     is a plain YOLO11s-seg (export_plain.py drops the heads).
  fm_mode="none"     Baseline, identical to SegmentationModel.

The FM itself lives in encoders._REGISTRY (outside this nn.Module), see encoders.py for why.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel

from encoders import encode, feat_dim

TAPS = (4, 6, 10)  # YOLO11 backbone outputs: P3/8, P4/16, P5/32


def _resize(f: torch.Tensor, size) -> torch.Tensor:
    if tuple(f.shape[-2:]) == tuple(size):
        return f
    if f.shape[-2] > size[0]:  # downsample: average pooling keeps the signal
        return F.adaptive_avg_pool2d(f, size)
    return F.interpolate(f, size=size, mode="bilinear", align_corners=False)


class FuseAdapter(nn.Module):
    """FM (D) -> YOLO (C) residual. Last conv is zero-initialised: output starts at 0."""

    def __init__(self, d: int, c: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, d)  # scale-normalise FM features (their magnitudes differ a lot per model)
        self.proj = nn.Sequential(nn.Conv2d(d, c, 1), nn.SiLU(), nn.Conv2d(c, c, 1))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, fm: torch.Tensor) -> torch.Tensor:
        w = self.proj[0].weight
        fm = self.norm(_resize(fm, x.shape[-2:]).to(w.dtype))
        return x + self.proj(fm).to(x.dtype)


class DistillHead(nn.Module):
    """YOLO (C) -> FM space (D). Training-only."""

    def __init__(self, c: int, d: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(c, d, 1), nn.SiLU(), nn.Conv2d(d, d, 1))

    def forward(self, x):
        return self.proj(x)


class FMSegModel(SegmentationModel):
    def __init__(self, cfg="yolo11s-seg.yaml", ch=3, nc=None, verbose=True,
                 fm_mode: str = "none", fm_name: str = "", fm_lambda: float = 1.0):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        assert fm_mode in ("none", "fusion", "distill"), fm_mode
        self.fm_mode, self.fm_name, self.fm_lambda = fm_mode, fm_name, float(fm_lambda)
        self._taps: list[torch.Tensor] = []
        if fm_mode != "none":
            chans = self._tap_channels(ch)
            d = feat_dim(fm_name)
            mods = [FuseAdapter(d, c) if fm_mode == "fusion" else DistillHead(c, d) for c in chans]
            self.fm_adapt = nn.ModuleList(mods)

    # ---------------------------------------------------------------- helpers
    @torch.no_grad()
    def _tap_channels(self, ch):
        out = {}
        hooks = [self.model[i].register_forward_hook(lambda m, a, o, i=i: out.__setitem__(i, o.shape[1]))
                 for i in TAPS]
        mode = self.training
        self.eval()
        super()._predict_once(torch.zeros(1, ch, 64, 64))
        self.train(mode)
        for h in hooks:
            h.remove()
        return [out[i] for i in TAPS]

    # ---------------------------------------------------------------- forward
    def _predict_once(self, x, profile=False, embed=None):
        mode = getattr(self, "fm_mode", "none")  # attribute does not exist yet during super().__init__
        if mode == "none" or embed:
            return super()._predict_once(x, profile=profile, embed=embed)

        fm = encode(self.fm_name, x) if mode == "fusion" else None
        self._taps = []
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            if m.i in TAPS:
                k = TAPS.index(m.i)
                if mode == "fusion":
                    x = self.fm_adapt[k](x, fm)
                else:
                    self._taps.append(x)
            y.append(x if m.i in self.save else None)
        return x

    # ---------------------------------------------------------------- loss
    def loss(self, batch, preds=None):
        loss, items = super().loss(batch, preds)
        if getattr(self, "fm_mode", "none") != "distill":
            return loss, items
        if not self._taps:  # preds came from somewhere that bypassed our forward: re-run it to get the taps
            self.predict(batch["img"])
        target = encode(self.fm_name, batch["img"])
        d = 0.0
        for head, tap in zip(self.fm_adapt, self._taps):
            s = head(tap).float()
            t = _resize(target, s.shape[-2:])
            d = d + (1 - F.cosine_similarity(s, t, dim=1)).mean()
        d = self.fm_lambda * d / len(self._taps)
        self._taps = []
        bs = batch["img"].shape[0]
        loss = torch.cat([loss, (d * bs).reshape(1).to(loss.dtype)])
        items = {**items, "distill": d.detach()}
        return loss, items


def fm_config_from_env():
    return (os.environ.get("FM_MODE", "none"), os.environ.get("FM_ENCODER", ""),
            float(os.environ.get("FM_LAMBDA", "1.0")))


class FMSegTrainer(SegmentationTrainer):
    """SegmentationTrainer that builds FMSegModel. FM settings come from env vars so that
    Ultralytics' strict argument checking and `resume=True` both keep working."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        mode, name, lam = fm_config_from_env()
        model = FMSegModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose,
                           fm_mode=mode, fm_name=name, fm_lambda=lam)
        model = self.set_model_names_for_load(model)
        if weights:
            model.load(weights)
        return model
