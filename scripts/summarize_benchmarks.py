from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.summary import summarize_reports


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
        choices=("model_name", "backend", "precision", "inference_ms", "total_ms", "fps"),
        default="inference_ms",
        help="Field used to sort the summary rows.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    summarize_reports(
        reports_dir=args.reports_dir,
        pattern=args.pattern,
        output_prefix=output_prefix,
        baseline_path=args.baseline,
        sort_by=args.sort_by,
        descending=args.descending,
    )

    print(f"Saved {output_prefix.with_suffix('.json')}")
    print(f"Saved {output_prefix.with_suffix('.csv')}")
    print(f"Saved {output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
