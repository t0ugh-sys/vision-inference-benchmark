from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Dict, List, Optional


@dataclass
class TimingMetrics:
    preprocess_ms: List[float] = field(default_factory=list)
    inference_ms: List[float] = field(default_factory=list)
    postprocess_ms: List[float] = field(default_factory=list)
    total_ms: List[float] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        total = mean(self.total_ms) if self.total_ms else 0.0
        return {
            "preprocess_ms": mean(self.preprocess_ms) if self.preprocess_ms else 0.0,
            "inference_ms": mean(self.inference_ms) if self.inference_ms else 0.0,
            "postprocess_ms": mean(self.postprocess_ms) if self.postprocess_ms else 0.0,
            "total_ms": total,
            "fps": 1000.0 / total if total else 0.0,
        }


@dataclass
class AccuracyMetrics:
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def compare_accuracy(baseline: AccuracyMetrics, candidate: AccuracyMetrics) -> Dict[str, float]:
    return {
        "map50_drop": baseline.map50 - candidate.map50,
        "map50_95_drop": baseline.map50_95 - candidate.map50_95,
        "precision_drop": baseline.precision - candidate.precision,
        "recall_drop": baseline.recall - candidate.recall,
    }


def merge_report(
    backend: str,
    model_path: str,
    timing: TimingMetrics,
    accuracy: Optional[AccuracyMetrics] = None,
    quantized: bool = False,
    model_name: str = "",
    precision: str = "unknown",
) -> Dict[str, object]:
    report: Dict[str, object] = {
        "backend": backend,
        "model_name": model_name,
        "model_path": model_path,
        "precision": precision,
        "quantized": quantized,
        "timing": timing.summary(),
    }
    if accuracy is not None:
        report["accuracy"] = accuracy.to_dict()
    return report
