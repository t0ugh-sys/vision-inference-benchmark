from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from pipeline.config import ensure_directory
from pipeline.report import load_report, save_report

TIMING_KEYS = ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms", "fps")


def discover_reports(reports_dir: str | Path, pattern: str) -> list[Path]:
    root = Path(reports_dir)
    if not root.exists():
        raise FileNotFoundError(f"Reports directory not found: {root}")
    return sorted(path for path in root.glob(pattern) if path.is_file())


def report_row(path: Path, report: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    timing = report.get("timing", {})
    row: dict[str, Any] = {
        "report_name": path.name,
        "model_name": report.get("model_name", ""),
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
    if sort_by in {"backend", "precision", "model_name"}:
        return sorted(rows, key=lambda item: str(item.get(sort_by, "")), reverse=descending)
    return sorted(rows, key=lambda item: float(item.get(sort_by, 0.0)), reverse=descending)


def to_markdown(rows: list[dict[str, Any]], baseline_name: str | None) -> str:
    headers = [
        "report_name",
        "model_name",
        "backend",
        "precision",
        "quantized",
        "inference_ms",
        "total_ms",
        "fps",
    ]
    if baseline_name is not None:
        headers.extend(["inference_ms_delta", "total_ms_delta", "fps_delta"])

    lines = ["# Benchmark Summary", ""]
    if baseline_name is not None:
        lines.append(f"Baseline: `{baseline_name}`")
        lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        values: list[str] = []
        for header in headers:
            value = row.get(header, "")
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_directory(target.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_reports(
    reports_dir: str | Path,
    pattern: str,
    output_prefix: str | Path,
    baseline_path: str | Path | None = None,
    sort_by: str = "inference_ms",
    descending: bool = False,
) -> dict[str, Any]:
    report_paths = discover_reports(reports_dir, pattern)
    if not report_paths:
        raise FileNotFoundError(f"No reports matched {pattern} in {reports_dir}")

    baseline_report: dict[str, Any] | None = None
    baseline_name: str | None = None
    if baseline_path:
        baseline = Path(baseline_path)
        baseline_report = load_report(baseline)
        baseline_report["_report_name"] = baseline.name
        baseline_name = baseline.name

    rows = [report_row(path, load_report(path), baseline_report) for path in report_paths]
    sorted_rows = sort_rows(rows, sort_by, descending)

    prefix = Path(output_prefix)
    save_report(prefix.with_suffix(".json"), {"baseline": baseline_name, "rows": sorted_rows})
    write_csv(prefix.with_suffix(".csv"), sorted_rows)
    prefix.with_suffix(".md").write_text(to_markdown(sorted_rows, baseline_name), encoding="utf-8")
    return {"baseline": baseline_name, "rows": sorted_rows}
