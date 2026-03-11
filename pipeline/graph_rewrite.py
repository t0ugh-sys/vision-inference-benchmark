from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BACKEND_WARN_OPS: dict[str, set[str]] = {
    "tensorrt": {
        "NonMaxSuppression",
        "GridSample",
        "DeformConv",
        "Mod",
        "CumSum",
        "Range",
    },
    "rknn": {
        "NonMaxSuppression",
        "GatherND",
        "GridSample",
        "Resize",
        "ScatterND",
        "LayerNormalization",
    },
    "onnxruntime": set(),
}


@dataclass
class RewriteResult:
    input_model: str
    output_model: str
    backend: str
    opset_before: int | None
    opset_after: int | None
    node_count_before: int
    node_count_after: int
    removed_identity_count: int = 0
    inferred_shapes: bool = False
    simplified: bool = False
    warnings: list[str] = field(default_factory=list)
    op_histogram: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_model": self.input_model,
            "output_model": self.output_model,
            "backend": self.backend,
            "opset_before": self.opset_before,
            "opset_after": self.opset_after,
            "node_count_before": self.node_count_before,
            "node_count_after": self.node_count_after,
            "removed_identity_count": self.removed_identity_count,
            "inferred_shapes": self.inferred_shapes,
            "simplified": self.simplified,
            "warnings": self.warnings,
            "op_histogram": self.op_histogram,
        }


def _require_onnx():
    try:
        import onnx
    except ImportError as exc:
        raise ImportError("onnx is required for graph rewrite. Install with: pip install onnx") from exc
    return onnx


def _load_model(path: str | Path):
    onnx = _require_onnx()
    return onnx.load(Path(path).as_posix())


def _save_model(model: Any, path: str | Path) -> None:
    onnx = _require_onnx()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, target.as_posix())


def get_opset_version(model: Any) -> int | None:
    for opset in getattr(model, "opset_import", []):
        if getattr(opset, "domain", "") in {"", "ai.onnx"}:
            return int(opset.version)
    return None


def histogram_ops(model: Any) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for node in model.graph.node:
        histogram[node.op_type] = histogram.get(node.op_type, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: (-item[1], item[0])))


def remove_identity_nodes(model: Any) -> tuple[Any, int]:
    removed = 0
    graph = model.graph
    nodes = list(graph.node)
    for node in nodes:
        if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
            continue
        source_name = node.input[0]
        target_name = node.output[0]
        for consumer in graph.node:
            for index, value in enumerate(consumer.input):
                if value == target_name:
                    consumer.input[index] = source_name
        for output in graph.output:
            if output.name == target_name:
                output.name = source_name
        graph.node.remove(node)
        removed += 1
    return model, removed


def infer_shapes(model: Any) -> Any:
    onnx = _require_onnx()
    return onnx.shape_inference.infer_shapes(model)


def convert_opset(model: Any, target_opset: int) -> Any:
    onnx = _require_onnx()
    return onnx.version_converter.convert_version(model, target_opset)


def simplify_model(model: Any, check_n: int = 0) -> tuple[Any, bool]:
    try:
        from onnxsim import simplify
    except ImportError as exc:
        raise ImportError("onnxsim is required for simplify pass. Install with: pip install onnxsim") from exc
    return simplify(model, check_n=check_n)


def check_backend_compatibility(model: Any, backend: str) -> list[str]:
    warn_ops = BACKEND_WARN_OPS.get(backend.lower(), set())
    warnings: list[str] = []
    histogram = histogram_ops(model)
    for op_name in warn_ops:
        count = histogram.get(op_name, 0)
        if count > 0:
            warnings.append(f"{backend} compatibility warning: op `{op_name}` appears {count} time(s)")
    return warnings


def rewrite_onnx_model(
    input_path: str | Path,
    output_path: str | Path,
    backend: str,
    *,
    target_opset: int | None = None,
    infer_shapes_first: bool = True,
    strip_identity: bool = True,
    simplify_graph: bool = False,
    simplify_check_n: int = 0,
    compatibility_check: bool = True,
) -> RewriteResult:
    model = _load_model(input_path)
    opset_before = get_opset_version(model)
    node_count_before = len(model.graph.node)

    inferred_shapes = False
    if infer_shapes_first:
        model = infer_shapes(model)
        inferred_shapes = True

    removed_identity_count = 0
    if strip_identity:
        model, removed_identity_count = remove_identity_nodes(model)

    if target_opset is not None:
        current = get_opset_version(model)
        if current is None or current != target_opset:
            model = convert_opset(model, target_opset)

    simplified = False
    if simplify_graph:
        model, simplified = simplify_model(model, check_n=simplify_check_n)

    _save_model(model, output_path)

    warnings: list[str] = []
    if compatibility_check:
        warnings.extend(check_backend_compatibility(model, backend))

    return RewriteResult(
        input_model=str(Path(input_path)),
        output_model=str(Path(output_path)),
        backend=backend,
        opset_before=opset_before,
        opset_after=get_opset_version(model),
        node_count_before=node_count_before,
        node_count_after=len(model.graph.node),
        removed_identity_count=removed_identity_count,
        inferred_shapes=inferred_shapes,
        simplified=simplified,
        warnings=warnings,
        op_histogram=histogram_ops(model),
    )
