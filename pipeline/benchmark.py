from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2

from pipeline.backends import build_backend
from pipeline.metrics import AccuracyMetrics, TimingMetrics, merge_report
from pipeline.postprocess import postprocess_predictions
from pipeline.preprocess import preprocess_image


def iter_images(directory: str | Path) -> Iterable[Path]:
    root = Path(directory)
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        for path in root.glob(suffix):
            yield path


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
