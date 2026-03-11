from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.report import load_report, save_report


TIMING_KEYS = ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms", "fps")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multiple benchmark reports.")
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory containing benchmark report json files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern used to discover report files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="reports/summary",
        help="Output prefix for generated summary files without extension.",
    )
    parser.add_argument(
        "--baseline",
        help="Optional baseline report path. If set, delta columns are generated relative to it.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("backend", "precision", "inference_ms", "total_ms", "fps"),
        default="inference_ms",
        help="Field used to sort the summary rows.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order.",
    )
    return parser.parse_args()


def discover_reports(reports_dir: Path, pattern: str) -> list[Path]:
    if not reports_dir.exists():
        raise FileNotFoundError(f"Reports directory not found: {reports_dir}")
    return sorted(path for path in reports_dir.glob(pattern) if path.is_file())


def report_row(path: Path, report: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    timing = report.get("timing", {})
    row: dict[str, Any] = {
        "report_name": path.name,
        "backend": report.get("backend", ""),
        "precision": report.get("precision", "unknown"),
        "quantized": bool(report.get("quantized", False)),
        "model_path": report.get("model_path", ""),
    }

    for key in TIMING_KEYS:
        row[key] = float(timing.get(key, 0.0))

    accuracy = report.get("accuracy", {})
    row["map50"] = float(accuracy.get("map50", 0.0))
    row["map50_95"] = float(accuracy.get("map50_95", 0.0))
    row["precision_metric"] = float(accuracy.get("precision", 0.0))
    row["recall"] = float(accuracy.get("recall", 0.0))

    if baseline is not None:
        baseline_timing = baseline.get("timing", {})
        row["baseline_report"] = baseline.get("_report_name", "")
        for key in ("inference_ms", "total_ms", "fps"):
            row[f"{key}_delta"] = row[key] - float(baseline_timing.get(key, 0.0))
        baseline_accuracy = baseline.get("accuracy", {})
        if accuracy and baseline_accuracy:
            row["map50_drop"] = float(baseline_accuracy.get("map50", 0.0)) - row["map50"]
            row["map50_95_drop"] = float(baseline_accuracy.get("map50_95", 0.0)) - row["map50_95"]
    return row


def sort_rows(rows: list[dict[str, Any]], sort_by: str, descending: bool) -> list[dict[str, Any]]:
    if sort_by in {"backend", "precision"}:
        key_fn = lambda item: str(item.get(sort_by, ""))
    else:
        key_fn = lambda item: float(item.get(sort_by, 0.0))
    return sorted(rows, key=key_fn, reverse=descending)


def to_markdown(rows: list[dict[str, Any]], baseline_name: str | None) -> str:
    headers = [
        "report_name",
        "backend",
        "precision",
        "quantized",
        "inference_ms",
        "total_ms",
        "fps",
    ]
    if baseline_name is not None:
        headers.extend(["inference_ms_delta", "total_ms_delta", "fps_delta"])

    lines = []
    lines.append("# Benchmark Summary")
    lines.append("")
    if baseline_name is not None:
        lines.append(f"Baseline: `{baseline_name}`")
        lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        values: list[str] = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    report_paths = discover_reports(reports_dir, args.pattern)
    if not report_paths:
        raise FileNotFoundError(f"No reports matched {args.pattern} in {reports_dir}")

    baseline_report: dict[str, Any] | None = None
    baseline_name: str | None = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        baseline_report = load_report(baseline_path)
        baseline_report["_report_name"] = baseline_path.name
        baseline_name = baseline_path.name

    rows = []
    for path in report_paths:
        report = load_report(path)
        rows.append(report_row(path, report, baseline_report))

    sorted_rows = sort_rows(rows, args.sort_by, args.descending)

    output_prefix = Path(args.output_prefix)
    save_report(output_prefix.with_suffix(".json"), {"baseline": baseline_name, "rows": sorted_rows})
    write_csv(output_prefix.with_suffix(".csv"), sorted_rows)
    output_prefix.with_suffix(".md").write_text(to_markdown(sorted_rows, baseline_name), encoding="utf-8")

    print(f"Saved {output_prefix.with_suffix('.json')}")
    print(f"Saved {output_prefix.with_suffix('.csv')}")
    print(f"Saved {output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
