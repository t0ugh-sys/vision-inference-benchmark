import unittest
from pathlib import Path

from pipeline.model_zoo import build_cases


class ModelZooTests(unittest.TestCase):
    def test_build_cases_filters_and_paths(self) -> None:
        models = {
            "yolov8": {
                "pt": "yolov8n.pt",
                "onnx_fp16": "weights/adapted/yolov8n.onnxruntime.onnx",
                "onnx_int8": "weights/adapted/yolov8n_int8.onnxruntime.onnx",
            },
            "yolo11": {
                "pt": "yolo11n.pt",
            },
        }
        cases = build_cases(
            models=models,
            project_root=Path("D:/workspace/vision-inference-benchmark"),
            selected_models={"yolov8"},
            selected_backends={"onnx"},
            selected_precisions={"fp16", "int8"},
        )

        self.assertEqual(len(cases), 2)
        self.assertTrue(all(case.model_name == "yolov8" for case in cases))
        self.assertTrue(all(case.backend == "onnx" for case in cases))
        self.assertTrue(any(case.precision == "fp16" for case in cases))
        self.assertTrue(any(case.precision == "int8" for case in cases))
        self.assertTrue(all(Path(case.model_path).is_absolute() for case in cases))


if __name__ == "__main__":
    unittest.main()

