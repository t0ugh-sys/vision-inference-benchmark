from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite and adapt an ONNX model for a target backend.")
    parser.add_argument("--config", required=True, help="Path to pipeline yaml")
    parser.add_argument("--input", required=True, help="Input ONNX model path")
    parser.add_argument("--backend", required=True, choices=["onnxruntime", "tensorrt", "rknn"])
    parser.add_argument("--output", help="Output ONNX model path")
    parser.add_argument("--report", help="Output JSON report path")
    parser.add_argument("--opset", type=int, help="Override target opset")
    parser.add_argument("--skip-infer-shapes", action="store_true", help="Disable ONNX shape inference")
    parser.add_argument("--skip-strip-identity", action="store_true", help="Disable Identity node removal")
    parser.add_argument("--simplify", action="store_true", help="Enable onnxsim simplify pass")
    parser.add_argument("--skip-compatibility-check", action="store_true", help="Disable backend compatibility warnings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from pipeline.config import load_config
    from pipeline.graph_rewrite import rewrite_onnx_model

    config = load_config(args.config)
    adapt_cfg = config.get("adapt", {})

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    default_output_dir = Path(adapt_cfg.get("output_dir", "weights/adapted"))
    output_path = Path(args.output) if args.output else default_output_dir / f"{input_path.stem}.{args.backend}.onnx"
    report_path = Path(args.report) if args.report else Path(config["project"]["report_dir"]) / f"{output_path.stem}.rewrite.json"

    target_opset = args.opset if args.opset is not None else adapt_cfg.get("target_opset")
    if isinstance(target_opset, str):
        target_opset = int(target_opset)

    result = rewrite_onnx_model(
        input_path=input_path,
        output_path=output_path,
        backend=args.backend,
        target_opset=target_opset,
        infer_shapes_first=not args.skip_infer_shapes and bool(adapt_cfg.get("infer_shapes", True)),
        strip_identity=not args.skip_strip_identity and bool(adapt_cfg.get("strip_identity", True)),
        simplify_graph=bool(args.simplify or adapt_cfg.get("simplify", False)),
        simplify_check_n=int(adapt_cfg.get("simplify_check_n", 0)),
        compatibility_check=not args.skip_compatibility_check and bool(adapt_cfg.get("compatibility_check", True)),
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Adapted model saved to {output_path}")
    print(f"Rewrite report saved to {report_path}")
    if result.warnings:
        print("Warnings:")
        for item in result.warnings:
            print(f"- {item}")


if __name__ == "__main__":
    main()
