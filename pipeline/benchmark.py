from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from pipeline.backends import build_backend
from pipeline.metrics import AccuracyMetrics, TimingMetrics, merge_report
from pipeline.postprocess import postprocess_predictions
from pipeline.preprocess import preprocess_image

logger = logging.getLogger(__name__)


def load_image(path: Path) -> np.ndarray:
    """Load a single image from path."""
    return cv2.imread(str(path))


def iter_images(directory: str | Path) -> Iterable[Path]:
    root = Path(directory)
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for path in root.glob(suffix):
            yield path


def load_images_parallel(paths: list[Path], num_workers: int | None = None) -> list[np.ndarray]:
    """Load images in parallel using thread pool."""
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        images = list(executor.map(load_image, paths))
    return images


def run_benchmark(config: dict, backend_name: str, model_path: str, image_dir: str) -> dict:
    backend = build_backend(backend_name, model_path)
    timing = TimingMetrics()

    image_paths = list(iter_images(image_dir))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    sample = cv2.imread(str(image_paths[0]))
    warm = preprocess_image(sample, config)
    backend.warmup(warm.tensor, int(config["benchmark"]["warmup"]))

    runs = int(config["benchmark"]["runs"])
    batch_size = int(config["benchmark"].get("batch_size", 1))

    if batch_size > 1:
        # Batch processing mode
        batch_images = []
        for idx in range(runs):
            path = image_paths[idx % len(image_paths)]
            if len(batch_images) < batch_size:
                batch_images.append(cv2.imread(str(path)))
            if len(batch_images) == batch_size or idx == runs - 1:
                # Process batch
                tensors = [preprocess_image(img, config).tensor for img in batch_images]
                batch_tensor = np.concatenate(tensors, axis=0)
                raw, infer_ms = backend.infer(batch_tensor)
                for i, img in enumerate(batch_images):
                    pre = preprocess_image(img, config)
                    single_raw = [raw[0][i:i+1] if isinstance(raw, list) else raw[i:i+1]]
                    post = postprocess_predictions(single_raw, pre.meta, config)
                    timing.preprocess_ms.append(pre.elapsed_ms)
                    timing.inference_ms.append(infer_ms / len(batch_images))
                    timing.postprocess_ms.append(post.elapsed_ms)
                    timing.total_ms.append(pre.elapsed_ms + infer_ms / len(batch_images) + post.elapsed_ms)
                batch_images = []
    else:
        # Sequential processing mode
        for idx in range(runs):
            image = cv2.imread(str(image_paths[idx % len(image_paths)]))
            pre = preprocess_image(image, config)
            raw, infer_ms = backend.infer(pre.tensor)
            post = postprocess_predictions(raw, pre.meta, config)

            timing.preprocess_ms.append(pre.elapsed_ms)
            timing.inference_ms.append(infer_ms)
            timing.postprocess_ms.append(post.elapsed_ms)
            timing.total_ms.append(pre.elapsed_ms + infer_ms + post.elapsed_ms)

    return merge_report(backend_name, model_path, timing)


def run_ultralytics_accuracy(config: dict, model_path: str) -> AccuracyMetrics:
    from ultralytics import YOLO

    model = YOLO(model_path)
    result = model.val(
        data=config["dataset"]["data"],
        split=config["accuracy"].get("split", "val"),
        imgsz=int(config["model"]["imgsz"]),
        conf=float(config["model"]["conf"]),
        iou=float(config["model"]["iou"]),
        verbose=False,
    )
    box = result.box
    return AccuracyMetrics(
        map50=float(box.map50),
        map50_95=float(box.map),
        precision=float(box.mp),
        recall=float(box.mr),
    )
