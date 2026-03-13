from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import resolve_path
from pipeline.metrics import AccuracyMetrics, compare_accuracy
from pipeline.report import load_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare benchmark reports.")
    parser.add_argument("--baseline", required=True, help="Baseline report json")
    parser.add_argument("--candidate", required=True, help="Candidate report json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_report(resolve_path(ROOT, args.baseline))
    candidate = load_report(resolve_path(ROOT, args.candidate))

    output = {
        "baseline_backend": baseline["backend"],
        "candidate_backend": candidate["backend"],
        "timing_delta_ms": {
            key: candidate["timing"][key] - baseline["timing"][key]
            for key in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms")
        },
        "fps_delta": candidate["timing"]["fps"] - baseline["timing"]["fps"],
    }

    if "accuracy" in baseline and "accuracy" in candidate:
        base = AccuracyMetrics(**baseline["accuracy"])
        cand = AccuracyMetrics(**candidate["accuracy"])
        output["accuracy_drop"] = compare_accuracy(base, cand)

    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
