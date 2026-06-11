from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image


def _collect_images(folder: Path) -> Dict[str, np.ndarray]:
    images = {}
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        with Image.open(path) as im:
            images[path.stem] = np.asarray(im).astype(np.float32) / 255.0
    if not images:
        raise RuntimeError(f"No images found in {folder}")
    return images


def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred - gt) ** 2)
    if mse <= 1e-10:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def evaluate_scene(renders_dir: Path, ground_truth_dir: Path) -> Dict[str, float]:
    """Compute PSNR on matching renders vs. ground truth images."""
    pred_images = _collect_images(renders_dir)
    gt_images = _collect_images(ground_truth_dir)

    scores = []
    for name, pred in pred_images.items():
        if name not in gt_images:
            continue
        gt = gt_images[name]
        if pred.shape != gt.shape:
            gt = np.array(Image.fromarray((gt * 255).astype(np.uint8)).resize(pred.shape[1::-1], Image.BICUBIC)) / 255.0
        scores.append(_psnr(pred, gt))
    if not scores:
        raise RuntimeError("No overlapping image names between renders and ground truth.")
    return {"psnr": float(np.mean(scores))}

