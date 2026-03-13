from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VARIANT_MAP: dict[tuple[str, str], str] = {
    ("pt", "fp32"): "pt",
    ("onnx", "fp16"): "onnx_fp16",
    ("onnx", "int8"): "onnx_int8",
    ("tensorrt", "fp16"): "tensorrt_fp16",
    ("tensorrt", "int8"): "tensorrt_int8",
    ("rknn", "int8"): "rknn_int8",
}


@dataclass(frozen=True)
class ModelZooCase:
    name: str
    model_name: str
    backend: str
    precision: str
    quantized: bool
    model_path: str


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def normalize_filters(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {value.lower() for value in values}


def load_model_map(config: dict) -> dict[str, dict]:
    models = config.get("benchmark_matrix", {}).get("models", {})
    if not models:
        raise ValueError("Missing benchmark_matrix.models in config")
    return models


def select_model_names(models: dict[str, dict], selected_models: set[str] | None) -> list[str]:
    return [name for name in models.keys() if not selected_models or name.lower() in selected_models]


def build_cases(
    models: dict[str, dict],
    project_root: Path,
    selected_models: set[str] | None = None,
    selected_backends: set[str] | None = None,
    selected_precisions: set[str] | None = None,
) -> list[ModelZooCase]:
    cases: list[ModelZooCase] = []
    for model_name, model_cfg in models.items():
        if selected_models and model_name.lower() not in selected_models:
            continue

        for (backend, precision), field_name in VARIANT_MAP.items():
            if selected_backends and backend not in selected_backends:
                continue
            if selected_precisions and precision not in selected_precisions:
                continue

            value = model_cfg.get(field_name)
            if not value:
                continue

            if backend == "pt":
                model_path = str(value)
            else:
                model_path = str(resolve_path(project_root, str(value)))

            cases.append(
                ModelZooCase(
                    name=f"{model_name}_{backend}_{precision}",
                    model_name=model_name,
                    backend=backend,
                    precision=precision,
                    quantized=precision == "int8",
                    model_path=model_path,
                )
            )
    return cases
