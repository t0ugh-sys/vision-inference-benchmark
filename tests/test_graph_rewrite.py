import shutil
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from pipeline.graph_rewrite import analyze_yolo_patterns, build_compatibility_report, rewrite_onnx_model, _load_model


class GraphRewriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path('tests/tmp') / self._testMethodName
        if self.base.exists():
            shutil.rmtree(self.base)
        self.base.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.base.exists():
            shutil.rmtree(self.base)

    def _save_model(self, model, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, path.as_posix())

    def test_remove_identity_node(self) -> None:
        input_path = self.base / 'identity.onnx'
        output_path = self.base / 'identity.out.onnx'

        x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        identity = helper.make_node('Identity', ['input'], ['id_out'], name='identity0')
        sigmoid = helper.make_node('Sigmoid', ['id_out'], ['sig_out'], name='sigmoid0')
        mul = helper.make_node('Mul', ['id_out', 'sig_out'], ['output'], name='mul0')
        graph = helper.make_graph([identity, sigmoid, mul], 'identity_graph', [x], [y])
        model = helper.make_model(graph, producer_name='test', opset_imports=[helper.make_opsetid('', 13)])
        self._save_model(model, input_path)

        result = rewrite_onnx_model(input_path, output_path, 'tensorrt')
        rewritten = _load_model(output_path)

        self.assertEqual(result.removed_identity_count, 1)
        self.assertEqual([node.op_type for node in rewritten.graph.node], ['Sigmoid', 'Mul'])

    def test_rewrite_silu_node(self) -> None:
        input_path = self.base / 'silu.onnx'
        output_path = self.base / 'silu.out.onnx'

        x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 32, 32])
        y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 32, 32])
        silu = helper.make_node('SiLU', ['input'], ['output'], name='silu0')
        graph = helper.make_graph([silu], 'silu_graph', [x], [y])
        model = helper.make_model(graph, producer_name='test', opset_imports=[helper.make_opsetid('', 13)])
        self._save_model(model, input_path)

        result = rewrite_onnx_model(input_path, output_path, 'tensorrt')
        rewritten = _load_model(output_path)

        self.assertEqual(result.rewrites_applied['rewrite_silu'], 1)
        self.assertEqual([node.op_type for node in rewritten.graph.node], ['Sigmoid', 'Mul'])

    def test_strip_trailing_nms_node(self) -> None:
        input_path = self.base / 'nms.onnx'
        output_path = self.base / 'nms.out.onnx'

        boxes = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, 10, 4])
        scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1, 1, 10])
        selected = helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [10, 3])
        max_output = helper.make_tensor('max_output_boxes_per_class', TensorProto.INT64, [1], [10])
        iou_threshold = helper.make_tensor('iou_threshold', TensorProto.FLOAT, [1], [0.5])
        score_threshold = helper.make_tensor('score_threshold', TensorProto.FLOAT, [1], [0.25])
        nms = helper.make_node(
            'NonMaxSuppression',
            ['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
            ['selected_indices'],
            name='nms0',
        )
        graph = helper.make_graph([nms], 'nms_graph', [boxes, scores], [selected], [max_output, iou_threshold, score_threshold])
        model = helper.make_model(graph, producer_name='test', opset_imports=[helper.make_opsetid('', 13)])
        self._save_model(model, input_path)

        result = rewrite_onnx_model(input_path, output_path, 'tensorrt', fail_on_blocked=True)
        rewritten = _load_model(output_path)

        self.assertEqual(result.rewrites_applied['strip_nms'], 1)
        self.assertEqual([node.op_type for node in rewritten.graph.node], [])
        self.assertEqual([output.name for output in rewritten.graph.output], ['boxes', 'scores'])
        self.assertEqual(result.compatibility['blocked'], False)
        self.assertEqual(result.model_analysis['embedded_nms'], False)

    def test_support_report_blocks_nms_for_tensorrt(self) -> None:
        x = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, 10, 4])
        s = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1, 1, 10])
        y = helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [10, 3])
        max_output = helper.make_tensor('max_output_boxes_per_class', TensorProto.INT64, [1], [10])
        iou_threshold = helper.make_tensor('iou_threshold', TensorProto.FLOAT, [1], [0.5])
        score_threshold = helper.make_tensor('score_threshold', TensorProto.FLOAT, [1], [0.25])
        nms = helper.make_node(
            'NonMaxSuppression',
            ['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
            ['selected_indices'],
            name='nms0',
        )
        graph = helper.make_graph([nms], 'nms_graph', [x, s], [y], [max_output, iou_threshold, score_threshold])
        model = helper.make_model(graph, producer_name='test', opset_imports=[helper.make_opsetid('', 13)])

        report = build_compatibility_report(model, 'tensorrt')

        self.assertTrue(report['blocked'])
        self.assertEqual(report['sensitive_ops'][0]['op_type'], 'NonMaxSuppression')

    def test_analyze_dfl_like_head_pattern(self) -> None:
        x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 64, 20, 20])
        y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 84, 8400])
        conv_w = helper.make_tensor('conv_w', TensorProto.FLOAT, [64, 64, 1, 1], [0.0] * (64 * 64))
        conv_b = helper.make_tensor('conv_b', TensorProto.FLOAT, [64], [0.0] * 64)
        conv = helper.make_node('Conv', ['input', 'conv_w', 'conv_b'], ['conv_out'], name='conv0')
        reshape_shape = helper.make_tensor('reshape_shape', TensorProto.INT64, [4], [1, 4, 16, 400])
        reshape = helper.make_node('Reshape', ['conv_out', 'reshape_shape'], ['reshape_out'], name='reshape0')
        softmax = helper.make_node('Softmax', ['reshape_out'], ['softmax_out'], name='softmax0', axis=2)
        graph = helper.make_graph([conv, reshape, softmax], 'dfl_graph', [x], [y], [conv_w, conv_b, reshape_shape])
        model = helper.make_model(graph, producer_name='test', opset_imports=[helper.make_opsetid('', 13)])

        analysis = analyze_yolo_patterns(model)

        self.assertTrue(analysis['dfl_like_head'])
        self.assertTrue(any('DFL-like head pattern' in note for note in analysis['notes']))


if __name__ == '__main__':
    unittest.main()
