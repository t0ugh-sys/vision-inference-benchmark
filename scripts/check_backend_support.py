from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import ensure_directory, resolve_path
from pipeline.report import save_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ONNX operator compatibility for a target backend.")
    parser.add_argument("--input", required=True, help="Input ONNX model path")
    parser.add_argument("--backend", required=True, choices=["onnxruntime", "tensorrt", "rknn"])
    parser.add_argument("--report", help="Optional JSON report output path")
    parser.add_argument("--fail-on-blocked", action="store_true", help="Return non-zero exit code when the graph is blocked")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from pipeline.graph_rewrite import _load_model, build_compatibility_report, get_opset_version, histogram_ops

    input_path = resolve_path(ROOT, args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    model = _load_model(input_path)
    compatibility = build_compatibility_report(model, args.backend)
    report = {
        "status": "blocked" if compatibility.get("blocked", False) else "ok",
        "stage": "support_check",
        "error": "Compatibility check blocked the graph" if compatibility.get("blocked", False) else None,
        "input_model": str(input_path),
        "backend": args.backend,
        "opset": get_opset_version(model),
        "op_histogram": histogram_ops(model),
        "compatibility": compatibility,
        "artifacts": {
            "input_model": str(input_path),
        },
    }

    if args.report:
        report_path = resolve_path(ROOT, args.report)
        ensure_directory(report_path.parent)
        save_report(report_path, report)
        print(f"Support report saved to {report_path}")

    print(json.dumps(report, indent=2, ensure_ascii=True))

    if compatibility.get("blocked", False) and args.fail_on_blocked:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
