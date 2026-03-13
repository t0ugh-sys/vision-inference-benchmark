from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import ensure_directory, load_project_config, resolve_path
from pipeline.model_zoo import build_cases, load_model_map, normalize_filters
from pipeline.summary import summarize_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a benchmark matrix for multiple YOLO model families.")
    parser.add_argument("--config", required=True, help="Path to pipeline yaml")
    parser.add_argument("--reports-dir", default="reports/model_zoo", help="Directory for generated benchmark reports")
    parser.add_argument("--summary-prefix", default="reports/model_zoo/summary", help="Output prefix for generated summary files")
    parser.add_argument("--models", nargs="*", help="Optional model keys from benchmark_matrix.models")
    parser.add_argument("--backends", nargs="*", choices=["pt", "onnx", "tensorrt", "rknn"], help="Optional backend filter")
    parser.add_argument("--precisions", nargs="*", choices=["fp32", "fp16", "int8"], help="Optional precision filter")
    parser.add_argument("--images", help="Validation images directory override")
    parser.add_argument("--skip-missing", action="store_true", help="Skip cases whose model files are missing")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately when one case fails")
    return parser.parse_args()


def run_case(config_path: Path, reports_dir: Path, images: str | None, case) -> Path:
    report_path = reports_dir / f"{case.name}.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_model.py"),
        "--config",
        str(config_path),
        "--backend",
        case.backend,
        "--model",
        case.model_path,
        "--model-name",
        case.model_name,
        "--precision",
        case.precision,
        "--report-dir",
        str(reports_dir),
        "--report-name",
        report_path.name,
    ]
    if case.quantized:
        command.append("--quantized")
    if images:
        command.extend(["--images", images])

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    return report_path


def run_summary(reports_dir: Path, summary_prefix: Path) -> None:
    summarize_reports(
        reports_dir=reports_dir,
        pattern="*.json",
        output_prefix=summary_prefix,
        sort_by="inference_ms",
    )


def main() -> None:
    args = parse_args()
    config_path, config, project_root = load_project_config(resolve_path(ROOT, args.config))
    reports_dir = ensure_directory(resolve_path(project_root, args.reports_dir))
    summary_prefix = resolve_path(project_root, args.summary_prefix)

    models = load_model_map(config)
    cases = build_cases(
        models=models,
        project_root=project_root,
        selected_models=normalize_filters(args.models),
        selected_backends=normalize_filters(args.backends),
        selected_precisions=normalize_filters(args.precisions),
    )
    if not cases:
        raise ValueError("No benchmark cases matched the requested filters")

    completed = 0
    skipped: list[str] = []
    failures: list[str] = []

    for case in cases:
        model_path = Path(case.model_path)
        if case.backend != "pt" and not model_path.exists():
            message = f"Skip {case.name}: missing model {model_path}"
            if args.skip_missing:
                print(message)
                skipped.append(case.name)
                continue
            raise FileNotFoundError(message)

        try:
            run_case(config_path, reports_dir, args.images, case)
            completed += 1
        except Exception as exc:
            message = f"{case.name}: {exc}"
            failures.append(message)
            print(f"Failed {message}")
            if args.fail_fast:
                raise

    if completed:
        run_summary(reports_dir, summary_prefix)

    print(f"Completed cases: {completed}")
    if skipped:
        print(f"Skipped cases: {len(skipped)}")
    if failures:
        print(f"Failed cases: {len(failures)}")
        for message in failures:
            print(f"- {message}")


if __name__ == "__main__":
    main()
