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
        "ScatterND",
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
    rewrites_applied: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    op_histogram: dict[str, int] = field(default_factory=dict)
    compatibility: dict[str, Any] = field(default_factory=dict)

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
            "rewrites_applied": self.rewrites_applied,
            "warnings": self.warnings,
            "op_histogram": self.op_histogram,
            "compatibility": self.compatibility,
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


def _next_tensor_name(base_name: str, suffix: str, used_names: set[str]) -> str:
    candidate = f"{base_name}_{suffix}"
    index = 1
    while candidate in used_names:
        candidate = f"{base_name}_{suffix}_{index}"
        index += 1
    used_names.add(candidate)
    return candidate


def _collect_value_names(model: Any) -> set[str]:
    names: set[str] = set()
    for item in model.graph.input:
        names.add(item.name)
    for item in model.graph.output:
        names.add(item.name)
    for item in model.graph.value_info:
        names.add(item.name)
    for item in model.graph.initializer:
        names.add(item.name)
    for node in model.graph.node:
        names.update(node.input)
        names.update(node.output)
    names.discard("")
    return names


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


def rewrite_custom_activation_ops(model: Any) -> tuple[Any, dict[str, int]]:
    onnx = _require_onnx()
    helper = onnx.helper
    graph = model.graph
    used_names = _collect_value_names(model)
    rewrites = {
        "rewrite_silu": 0,
        "rewrite_hardswish": 0,
    }

    new_nodes = []
    for node in graph.node:
        if node.op_type in {"SiLU", "Swish"} and len(node.input) == 1 and len(node.output) == 1:
            input_name = node.input[0]
            output_name = node.output[0]
            sigmoid_name = _next_tensor_name(output_name, "sigmoid", used_names)
            sigmoid_node = helper.make_node("Sigmoid", [input_name], [sigmoid_name], name=f"{node.name or output_name}_sigmoid")
            mul_node = helper.make_node("Mul", [input_name, sigmoid_name], [output_name], name=f"{node.name or output_name}_mul")
            new_nodes.extend([sigmoid_node, mul_node])
            rewrites["rewrite_silu"] += 1
            continue

        if node.op_type == "HardSwish" and len(node.input) == 1 and len(node.output) == 1:
            input_name = node.input[0]
            output_name = node.output[0]
            hsigmoid_name = _next_tensor_name(output_name, "hsigmoid", used_names)
            hsigmoid_node = helper.make_node(
                "HardSigmoid",
                [input_name],
                [hsigmoid_name],
                name=f"{node.name or output_name}_hsigmoid",
                alpha=1.0 / 6.0,
                beta=0.5,
            )
            mul_node = helper.make_node("Mul", [input_name, hsigmoid_name], [output_name], name=f"{node.name or output_name}_mul")
            new_nodes.extend([hsigmoid_node, mul_node])
            rewrites["rewrite_hardswish"] += 1
            continue

        new_nodes.append(node)

    graph.ClearField("node")
    graph.node.extend(new_nodes)
    return model, rewrites


def normalize_resize_ops(model: Any, backend: str) -> tuple[Any, dict[str, int]]:
    if backend.lower() not in {"rknn", "tensorrt"}:
        return model, {"normalize_resize": 0}

    onnx = _require_onnx()
    helper = onnx.helper
    normalized = 0
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if "coordinate_transformation_mode" not in attrs:
            preferred = b"asymmetric" if backend.lower() == "rknn" else b"half_pixel"
            node.attribute.extend([helper.make_attribute("coordinate_transformation_mode", preferred)])
            normalized += 1
        mode = attrs.get("mode", b"nearest")
        if isinstance(mode, str):
            mode = mode.encode("utf-8")
        if mode == b"nearest" and "nearest_mode" not in attrs:
            node.attribute.extend([helper.make_attribute("nearest_mode", b"floor")])
            normalized += 1
    return model, {"normalize_resize": normalized}


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


def _standard_onnx_ops() -> set[str]:
    onnx = _require_onnx()
    return {schema.name for schema in onnx.defs.get_all_schemas_with_history()}


def build_compatibility_report(model: Any, backend: str) -> dict[str, Any]:
    histogram = histogram_ops(model)
    warn_ops = BACKEND_WARN_OPS.get(backend.lower(), set())
    standard_ops = _standard_onnx_ops()

    sensitive_ops = []
    custom_ops = []
    warnings: list[str] = []

    for op_name, count in histogram.items():
        if op_name in warn_ops:
            sensitive_ops.append({
                "op_type": op_name,
                "count": count,
                "status": "needs_review",
            })
            warnings.append(f"{backend} compatibility warning: op `{op_name}` appears {count} time(s)")
        if op_name not in standard_ops:
            custom_ops.append({
                "op_type": op_name,
                "count": count,
                "status": "custom_op",
            })
            warnings.append(f"Custom ONNX op detected: `{op_name}` appears {count} time(s)")

    return {
        "backend": backend,
        "total_nodes": sum(histogram.values()),
        "sensitive_ops": sensitive_ops,
        "custom_ops": custom_ops,
        "blocked": bool(sensitive_ops or custom_ops),
        "warnings": warnings,
    }


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
    rewrite_custom_ops: bool = True,
    normalize_resize: bool = True,
) -> RewriteResult:
    model = _load_model(input_path)
    opset_before = get_opset_version(model)
    node_count_before = len(model.graph.node)
    rewrites_applied: dict[str, int] = {}

    inferred_shapes = False
    if infer_shapes_first:
        model = infer_shapes(model)
        inferred_shapes = True

    removed_identity_count = 0
    if strip_identity:
        model, removed_identity_count = remove_identity_nodes(model)

    if rewrite_custom_ops:
        model, activation_rewrites = rewrite_custom_activation_ops(model)
        rewrites_applied.update(activation_rewrites)

    if normalize_resize:
        model, resize_rewrites = normalize_resize_ops(model, backend)
        rewrites_applied.update(resize_rewrites)

    if target_opset is not None:
        current = get_opset_version(model)
        if current is None or current != target_opset:
            model = convert_opset(model, target_opset)
            rewrites_applied["convert_opset"] = 1

    simplified = False
    if simplify_graph:
        model, simplified = simplify_model(model, check_n=simplify_check_n)

    compatibility = build_compatibility_report(model, backend) if compatibility_check else {
        "backend": backend,
        "total_nodes": len(model.graph.node),
        "sensitive_ops": [],
        "custom_ops": [],
        "blocked": False,
        "warnings": [],
    }

    _save_model(model, output_path)

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
        rewrites_applied=rewrites_applied,
        warnings=list(compatibility.get("warnings", [])),
        op_histogram=histogram_ops(model),
        compatibility=compatibility,
    )
