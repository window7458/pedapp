"""Image-only tactile-paving segmentation with the public GuideTWSI checkpoint.

Tested on the four supplied images. No training, text prompts, manual points,
image-specific coordinates, yellow-color threshold, or hand-drawn masks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageOps

WEIGHTS_URL = "https://raw.githubusercontent.com/DARoSLab/GuideTWSI/master/model_weights/yolo11n_tactile.pt"
WEIGHTS_SHA256 = "79f0bb39a354dd66d367bb5b97184df626c85f91568456e099e49a8cdbfcd878"
SUPPORTED = {".jpg", ".jpeg", ".jpe", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_weights(path=None):
    """Download the verified official weight if absent; reject changed files."""
    path = Path(path) if path is not None else ROOT / "weights" / "yolo11n_tactile.pt"
    if path.is_file():
        if sha256(path) != WEIGHTS_SHA256:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading GuideTWSI weights (about 6 MB)...", flush=True)
    descriptor, temporary = tempfile.mkstemp(prefix="download_", suffix=".pt", dir=path.parent)
    os.close(descriptor)
    try:
        request = urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "TactileImageDemo/2.0"})
        with urllib.request.urlopen(request, timeout=90) as response, open(temporary, "wb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
        if sha256(temporary) != WEIGHTS_SHA256:
            raise ValueError("Downloaded checkpoint differs from the tested checkpoint; download aborted.")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def read_image(path):
    with Image.open(path) as original:
        return ImageOps.exif_transpose(original).convert("RGB")


def overlay_mask(image, mask, alpha=0.4):
    """Display only the predicted mask, with no bounding box or direction label."""
    rgb = np.asarray(image).copy()
    if mask.shape != (image.height, image.width):
        raise ValueError("Mask dimensions do not match the EXIF-corrected image")
    if not 0 <= alpha <= 1:
        raise ValueError("Overlay alpha must be between 0 and 1")
    color = np.array([0, 220, 190], dtype=np.float32)
    rgb[mask] = np.rint(rgb[mask] * (1 - alpha) + color * alpha).astype(np.uint8)
    return Image.fromarray(rgb)


def side_by_side(original, overlay):
    scale = min(1.0, 850 / original.width)
    size = (max(1, round(original.width * scale)), max(1, round(original.height * scale)))
    left = original.resize(size, Image.Resampling.LANCZOS)
    right = overlay.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size[0] * 2 + 12, size[1] + 40), "#f1f5f9")
    canvas.paste(left, (0, 40))
    canvas.paste(right, (size[0] + 12, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), "Original", fill="#10242b")
    draw.text((size[0] + 22, 12), "Predicted tactile paving", fill="#10242b")
    return canvas


def merge_consistent_masks(first, mirrored, shape, confidence=0.7, match_iou=0.5):
    """Align mirrored predictions, match masks, and retain their intersection.

    Pair score is the arithmetic mean of the two detector scores, not a
    calibrated probability. Missing detections in either view are rejected.
    """
    second_masks = mirrored["masks"][:, :, ::-1]
    matches = []
    first_areas = first["masks"].sum(axis=(1, 2))
    second_areas = second_masks.sum(axis=(1, 2))
    for i, a in enumerate(first["masks"]):
        for j, b in enumerate(second_masks):
            score = (first["scores"][i] + mirrored["scores"][j]) / 2
            if score < confidence:
                continue
            intersection = int(np.count_nonzero(a & b))
            overlap = intersection / max(int(first_areas[i] + second_areas[j]) - intersection, 1)
            if overlap >= match_iou:
                matches.append((overlap, score, i, j))
    used_first, used_second = set(), set()
    masks, scores, overlaps = [], [], []
    for overlap, score, i, j in sorted(matches, reverse=True):
        if i in used_first or j in used_second:
            continue
        used_first.add(i)
        used_second.add(j)
        masks.append(first["masks"][i] & second_masks[j])
        scores.append(score)
        overlaps.append(overlap)
    masks = np.stack(masks) if masks else np.zeros((0, *shape), dtype=bool)
    return masks, scores, overlaps


class TactileSegmenter:
    def __init__(self, weights=None, device=None, threads=4):
        import torch
        from ultralytics import YOLO
        self.torch = torch
        self.device = str(device if device is not None else ("0" if torch.cuda.is_available() else "cpu"))
        if self.device == "cpu":
            if threads < 1:
                raise ValueError("CPU threads must be positive")
            torch.set_num_threads(threads)
        self.weights = ensure_weights(weights)
        self.model = YOLO(str(self.weights))
        if self.model.task != "segment":
            raise ValueError("The checkpoint must be a segmentation model")
        if self.model.names != {0: "braille"}:
            raise ValueError(f"Unexpected classes: {self.model.names}")

    def _predict_view(self, image, imgsz, conf, iou):
        result = self.model.predict(
            source=image, imgsz=imgsz, conf=conf, iou=iou,
            retina_masks=True, device=self.device, verbose=False,
            classes=[0], max_det=50, save=False,
        )[0]
        if result.masks is None:
            return {"masks": np.zeros((0, image.height, image.width), dtype=bool), "scores": []}
        masks = result.masks.data.detach().cpu().numpy().astype(bool)
        scores = result.boxes.conf.detach().cpu().tolist()
        if masks.shape[1:] != (image.height, image.width):
            raise RuntimeError("retina_masks=True did not return native-resolution masks")
        if len(masks) != len(scores):
            raise RuntimeError("The mask count and detection count differ")
        return {"masks": masks, "scores": scores}

    def predict(self, image, imgsz=960, conf=0.7, iou=0.7, mode="consistent"):
        if not 0 <= conf <= 1 or not 0 <= iou <= 1:
            raise ValueError("Confidence and IoU thresholds must be between 0 and 1")
        if imgsz < 32 or imgsz % 32:
            raise ValueError("imgsz must be a positive multiple of 32, e.g. 640 or 960")
        if mode not in {"single", "consistent"}:
            raise ValueError("mode must be single or consistent")
        if image.mode != "RGB":
            image = image.convert("RGB")
        start = time.perf_counter()
        if mode == "consistent":
            original = self._predict_view(image, imgsz, min(0.25, conf), iou)
            mirrored = self._predict_view(ImageOps.mirror(image), imgsz, min(0.25, conf), iou)
            masks, scores, overlaps = merge_consistent_masks(
                original, mirrored, (image.height, image.width), confidence=conf, match_iou=0.5)
        else:
            original = self._predict_view(image, imgsz, conf, iou)
            mirrored = {"masks": np.zeros((0, image.height, image.width), dtype=bool), "scores": []}
            masks, scores, overlaps = original["masks"], original["scores"], []
        boxes = []
        for mask in masks:
            ys, xs = np.nonzero(mask)
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
        union = masks.any(axis=0) if len(masks) else np.zeros((image.height, image.width), dtype=bool)
        elapsed = time.perf_counter() - start
        return {
            "mask": union, "instance_masks": masks, "scores": scores,
            "boxes_xyxy": boxes, "seconds": elapsed, "match_ious": overlaps,
            "original": original, "mirrored": mirrored,
        }


def run_images(paths, output_dir, segmenter, imgsz=960, conf=0.7, iou=0.7, alpha=0.4, mode="consistent"):
    paths = [Path(path).expanduser().resolve() for path in paths]
    if not paths:
        raise ValueError("No images selected")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "model": "GuideTWSI YOLO11n-seg", "class": "braille (tactile paving)",
        "weights_sha256": WEIGHTS_SHA256, "weights_source": WEIGHTS_URL,
        "device": segmenter.device, "imgsz": imgsz, "confidence_threshold": conf,
        "nms_iou_threshold": iou, "retina_masks": True, "overlay_alpha": alpha,
        "mode": mode, "candidate_threshold": min(0.25, conf) if mode == "consistent" else conf,
        "view_matching_mask_iou": 0.5 if mode == "consistent" else None,
        "fusion": "intersection of matched original/mirrored masks" if mode == "consistent" else "single-view mask",
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "ultralytics", "numpy", "Pillow", "opencv-python")},
        "note": "Masks are predictions. No ground-truth accuracy is calculated. Timing excludes model loading; first inference includes warm-up.",
    }
    (output_dir / "run_info.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    records = []
    fields = ["image", "status", "instance_count", "max_confidence", "mask_pixels", "image_pixels", "seconds"]
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, path in enumerate(paths, 1):
            image = read_image(path)
            prediction = segmenter.predict(image, imgsz, conf, iou, mode)
            stem = f"image_{index:02d}"
            mask = prediction["mask"]
            overlay = overlay_mask(image, mask, alpha)
            Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / f"{stem}_mask.png")
            overlay.save(output_dir / f"{stem}_overlay.png")
            side_by_side(image, overlay).save(output_dir / f"{stem}_comparison.png")
            np.savez_compressed(output_dir / f"{stem}_instances.npz",
                                masks=prediction["instance_masks"], scores=np.array(prediction["scores"]),
                                original_masks=prediction["original"]["masks"],
                                original_scores=np.array(prediction["original"]["scores"]),
                                mirrored_masks_unaligned=prediction["mirrored"]["masks"],
                                mirrored_scores=np.array(prediction["mirrored"]["scores"]))
            row = {
                "image": path.name, "status": "detected" if mask.any() else "not_detected",
                "instance_count": len(prediction["scores"]),
                "max_confidence": max(prediction["scores"], default=0.0),
                "mask_pixels": int(mask.sum()), "image_pixels": mask.size,
                "seconds": round(prediction["seconds"], 4),
            }
            record = {**row, "width": image.width, "height": image.height,
                      "input_sha256": sha256(path), "scores": prediction["scores"],
                      "match_ious": prediction["match_ious"],
                      "boxes_xyxy_for_audit_only": prediction["boxes_xyxy"],
                      "mask_file": f"{stem}_mask.png", "overlay_file": f"{stem}_overlay.png",
                      "comparison_file": f"{stem}_comparison.png"}
            (output_dir / f"{stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            writer.writerow(row)
            handle.flush()
            records.append(record)
            print(f"[{index}/{len(paths)}] {path.name}: {row['instance_count']} region(s), {row['seconds']:.3f}s", flush=True)
    (output_dir / "summary.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def main():
    parser = argparse.ArgumentParser(description="Standalone tactile-paving image segmentation")
    parser.add_argument("--input", nargs="+", required=True, help="Image files or directories")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--weights", default=None, help="Optional path to the same verified checkpoint")
    parser.add_argument("--device", default=None, help="cpu, 0, or cuda:0; defaults to available CUDA or CPU")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--mode", choices=["consistent", "single"], default="consistent")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.4)
    args = parser.parse_args()
    paths = []
    for source in args.input:
        path = Path(source).expanduser()
        if path.is_dir():
            paths.extend(p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED)
        else:
            paths.append(path)
    segmenter = TactileSegmenter(args.weights, args.device)
    run_images(paths, args.output, segmenter, args.imgsz, args.conf, args.iou, args.alpha, args.mode)


if __name__ == "__main__":
    main()
