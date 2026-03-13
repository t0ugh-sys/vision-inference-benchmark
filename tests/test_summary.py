import json
import shutil
import unittest
from pathlib import Path

from pipeline.report import save_report
from pipeline.summary import summarize_reports


class SummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path("tests/tmp") / self._testMethodName
        if self.base.exists():
            shutil.rmtree(self.base)
        self.base.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.base.exists():
            shutil.rmtree(self.base)

    def test_summarize_reports_keeps_model_name_and_deltas(self) -> None:
        report_a = self.base / "a.json"
        report_b = self.base / "b.json"
        save_report(
            report_a,
            {
                "model_name": "yolo11",
                "backend": "onnx",
                "precision": "fp16",
                "quantized": False,
                "model_path": "weights/a.onnx",
                "timing": {"preprocess_ms": 1.0, "inference_ms": 2.0, "postprocess_ms": 1.0, "total_ms": 4.0, "fps": 250.0},
            },
        )
        save_report(
            report_b,
            {
                "model_name": "yolov8",
                "backend": "onnx",
                "precision": "int8",
                "quantized": True,
                "model_path": "weights/b.onnx",
                "timing": {"preprocess_ms": 1.0, "inference_ms": 3.0, "postprocess_ms": 1.0, "total_ms": 5.0, "fps": 200.0},
            },
        )

        summary = summarize_reports(
            reports_dir=self.base,
            pattern="*.json",
            output_prefix=self.base / "summary",
            baseline_path=report_a,
            sort_by="inference_ms",
        )

        self.assertEqual(len(summary["rows"]), 2)
        self.assertEqual(summary["rows"][0]["model_name"], "yolo11")
        self.assertEqual(summary["rows"][1]["model_name"], "yolov8")
        self.assertEqual(summary["rows"][1]["inference_ms_delta"], 1.0)

        summary_json = json.loads((self.base / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary_json["baseline"], "a.json")


if __name__ == "__main__":
    unittest.main()
