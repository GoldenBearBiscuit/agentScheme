"""Seed capabilities and execution plans shared by every agent style."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .compare import compare
from .topology import (
    Capability,
    CapabilityRef,
    ExecutionPlan,
    ExecutionStep,
    FunctionalTopology,
    KernelError,
)


def text_normalize(text: str) -> str:
    return " ".join(text.split())


def text_word_count(text: str) -> int:
    return len(text.split())


def context_plan(capability_id: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"capability_id": capability_id, "args": args}


def context_observe(value: Any) -> dict[str, Any]:
    return {"source": "internal_trace", "status": "observed", "value": value}


@dataclass(frozen=True)
class PythonSource:
    path: str
    symbol: str
    source: str
    function: Callable[..., Any]


@dataclass(frozen=True)
class ExpressionCandidate:
    expression: str
    summary: str = ""

    def __post_init__(self) -> None:
        if not self.expression.strip():
            raise KernelError("candidate expression cannot be empty")


def load_python(path: str, symbol: str) -> PythonSource:
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise KernelError(f"source file not found: {source_path}")
    package_parts = [source_path.stem]
    package_root = source_path.parent
    while (package_root / "__init__.py").is_file():
        package_parts.append(package_root.name)
        package_root = package_root.parent
    if len(package_parts) > 1:
        module_name = ".".join(reversed(package_parts))
        added = str(package_root) not in sys.path
        if added:
            sys.path.insert(0, str(package_root))
        try:
            module = importlib.import_module(module_name)
        finally:
            if added:
                sys.path.remove(str(package_root))
    else:
        digest = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:16]
        module_name = f"_agentschema_source_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise KernelError(f"cannot load Python source: {source_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    function = getattr(module, symbol, None)
    if not callable(function) or inspect.iscoroutinefunction(function):
        raise KernelError(f"source symbol is not a synchronous callable: {symbol}")
    return PythonSource(
        path=str(source_path),
        symbol=symbol,
        source=inspect.getsource(function),
        function=function,
    )


def candidate_from_expression(expression: str) -> ExpressionCandidate:
    return ExpressionCandidate(expression=expression, summary="manual candidate")


def propose_expression(
    source: PythonSource,
    inputs: list[dict[str, Any]],
    llm: Any,
) -> ExpressionCandidate:
    if llm is None or not getattr(llm, "available", False):
        raise KernelError("clone proposal requires an available LLM provider")
    system = (
        "Clone the supplied pure Python function as one restricted Python expression. "
        "Return JSON with keys expression and summary. Do not return expected outputs. "
        "Use only argument names, literals, arithmetic, boolean logic, comparisons, "
        "conditional expressions, list/tuple/dict values, indexing, and these functions: "
        "abs, bool, float, int, int_or, len, max, min, round, str, sum. No attributes, imports, "
        "assignments, comprehensions, or other calls."
    )
    user = json.dumps(
        {"path": source.path, "symbol": source.symbol, "source": source.source, "inputs": inputs},
        ensure_ascii=False,
    )
    response = llm.complete_json(system, user)
    expression = response.get("expression")
    if not isinstance(expression, str):
        raise KernelError("LLM response must contain a string expression")
    return ExpressionCandidate(
        expression=expression,
        summary=str(response.get("summary") or "LLM candidate"),
    )


def repair_expression(
    source: PythonSource,
    inputs: list[dict[str, Any]],
    candidate: ExpressionCandidate,
    verification: dict[str, Any],
    llm: Any,
) -> ExpressionCandidate:
    if verification.get("accepted"):
        return candidate
    if llm is None or not getattr(llm, "available", False):
        raise KernelError("clone repair requires an available LLM provider")
    system = (
        "Repair the restricted Python expression using the deterministic differential report. "
        "Return JSON with keys expression and summary. Do not return expected outputs. "
        "Use only argument names, literals, arithmetic, boolean logic, comparisons, "
        "conditional expressions, list/tuple/dict values, indexing, and these functions: "
        "abs, bool, float, int, int_or, len, max, min, round, str, sum. No attributes, imports, "
        "assignments, comprehensions, or other calls."
    )
    user = json.dumps(
        {
            "source": source.source,
            "inputs": inputs,
            "candidate": candidate.expression,
            "verification": verification,
        },
        ensure_ascii=False,
    )
    response = llm.complete_json(system, user)
    expression = response.get("expression")
    if not isinstance(expression, str):
        raise KernelError("LLM repair response must contain a string expression")
    return ExpressionCandidate(
        expression=expression,
        summary=str(response.get("summary") or "LLM repaired candidate"),
    )


def propose_inputs(
    source: PythonSource,
    count: int,
    llm: Any,
) -> list[dict[str, Any]]:
    if llm is None or not getattr(llm, "available", False):
        raise KernelError("test generation requires an available LLM provider")
    if not 1 <= count <= 32:
        raise KernelError("test case count must be between 1 and 32")
    system = (
        "Generate diverse valid inputs for testing the supplied Python function. "
        "Return JSON with one key inputs containing a list of argument objects. "
        "Cover normal, boundary, and invalid values accepted by the signature. "
        "Do not return expected outputs."
    )
    user = json.dumps(
        {"symbol": source.symbol, "source": source.source, "count": count},
        ensure_ascii=False,
    )
    response = llm.complete_json(system, user)
    inputs = response.get("inputs")
    if not isinstance(inputs, list) or not inputs or any(
        not isinstance(item, dict) for item in inputs
    ):
        raise KernelError("LLM test response must contain a non-empty inputs list")
    try:
        for item in inputs:
            inspect.signature(source.function).bind(**item)
    except TypeError as exc:
        raise KernelError(f"generated test inputs do not match the signature: {exc}") from exc
    return inputs


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_SAFE_FUNCTIONS = {
    "abs": abs,
    "bool": bool,
    "float": float,
    "int": int,
    "int_or": _int_or,
    "len": len,
    "max": max,
    "min": min,
    "round": round,
    "str": str,
    "sum": sum,
}
_SAFE_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Subscript,
    ast.Slice,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
)


def evaluate_expression(expression: str, args: dict[str, Any]) -> Any:
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            raise KernelError(f"unsupported candidate syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS:
                raise KernelError("candidate calls must use an allowed pure function")
        if isinstance(node, ast.Name) and node.id not in args and node.id not in _SAFE_FUNCTIONS:
            raise KernelError(f"unknown candidate name: {node.id}")
    namespace = {**_SAFE_FUNCTIONS, **args}
    return eval(compile(tree, "<candidate>", "eval"), {"__builtins__": {}}, namespace)


def _outcome(run: Callable[[], Any]) -> dict[str, Any]:
    try:
        return {"ok": True, "value": run()}
    except Exception as exc:  # noqa: BLE001 - errors are part of the behavior contract
        return {"ok": False, "error": type(exc).__name__}


def _same_outcome(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    if expected["ok"] != actual["ok"]:
        return False
    if not expected["ok"]:
        return expected["error"] == actual["error"]
    return compare(expected["value"], actual["value"])


def _provenance(source: PythonSource) -> dict[str, str]:
    return {"kind": "external_code", "path": source.path, "symbol": source.symbol}


def differential_verify(
    source: PythonSource,
    inputs: list[dict[str, Any]],
    candidate: ExpressionCandidate,
) -> dict[str, Any]:
    if not inputs or any(not isinstance(item, dict) for item in inputs):
        raise KernelError("verification inputs must be a non-empty list of argument objects")
    cases = []
    for args in inputs:
        expected = _outcome(lambda: source.function(**args))
        actual = _outcome(lambda: evaluate_expression(candidate.expression, args))
        cases.append({
            "args": args,
            "expected": expected,
            "actual": actual,
            "match": _same_outcome(expected, actual),
        })
    return {
        "harness": "differential-v1",
        "accepted": all(item["match"] for item in cases),
        "candidate": {"expression": candidate.expression, "summary": candidate.summary},
        "provenance": _provenance(source),
        "cases": cases,
    }


def sqlite_verify(
    source: PythonSource,
    inputs: list[dict[str, Any]],
    candidate: ExpressionCandidate,
) -> dict[str, Any]:
    report = differential_verify(source, inputs, candidate)
    mutants = ("None", "0", "1", "False", "True", '"__mutation__"')
    survivors = [
        expression
        for expression in mutants
        if differential_verify(
            source,
            inputs,
            ExpressionCandidate(expression, "negative control"),
        )["accepted"]
    ]
    sensitive = not survivors
    report["harness"] = "sqlite-commit-v1"
    report["mutation"] = {
        "checked": len(mutants),
        "survivors": survivors,
        "sensitive": sensitive,
    }
    report["accepted"] = bool(report["accepted"] and sensitive)
    return report


def _expression_capability(record: dict[str, Any]) -> Capability:
    expression = str(record["expression"])
    defaults = dict(record.get("defaults") or {})

    def run(**args: Any) -> Any:
        return evaluate_expression(expression, {**defaults, **args})

    return Capability(
        capability_id=str(record["capability_id"]),
        run=run,
        summary=str(record.get("summary") or "Absorbed expression"),
        inputs={name: "any" for name in record["inputs"]},
        required=tuple(record["required"]),
        output="any",
        provenance=tuple(record.get("provenance") or ()),
        oracle=str(record.get("oracle") or "differential"),
    )


class ExpressionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self, functional: FunctionalTopology) -> None:
        if not self.path.is_file():
            return
        records = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise KernelError("expression registry must contain a list")
        for record in records:
            functional.register(_expression_capability(record))

    def save(self, record: dict[str, Any]) -> None:
        records = []
        if self.path.is_file():
            records = json.loads(self.path.read_text(encoding="utf-8"))
        records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.path)


def register_candidate(
    functional: FunctionalTopology,
    source: PythonSource,
    candidate: ExpressionCandidate,
    verification: dict[str, Any],
    capability_id: str,
) -> dict[str, Any]:
    report = {**verification, "capability_id": capability_id, "registered": False}
    if not verification.get("accepted"):
        return report
    if not capability_id:
        raise KernelError("absorbed capability_id cannot be empty")
    parameters = inspect.signature(source.function).parameters
    if any(
        parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        for parameter in parameters.values()
    ):
        raise KernelError("cannot register a candidate with variadic parameters")
    record = {
        "capability_id": capability_id,
        "expression": candidate.expression,
        "summary": candidate.summary,
        "inputs": list(parameters),
        "required": [
            name for name, parameter in parameters.items()
            if parameter.default is parameter.empty
        ],
        "defaults": {
            name: parameter.default
            for name, parameter in parameters.items()
            if parameter.default is not parameter.empty
        },
        "provenance": [f"{source.path}:{source.symbol}"],
        "oracle": str(verification.get("harness") or "differential"),
    }
    functional.register(_expression_capability(record))
    if functional.expression_store is not None:
        functional.expression_store.save(record)
    report["registered"] = True
    return report


def seed_functional_topology() -> FunctionalTopology:
    functional = FunctionalTopology()
    capabilities = (
        Capability(
            "text.normalize", text_normalize, "Collapse whitespace",
            {"text": "string"}, ("text",), "string",
            oracle="tests.test_agentschema.KernelTest.test_styles_preserve_functional_output",
        ),
        Capability(
            "text.word_count", text_word_count, "Count whitespace-delimited words",
            {"text": "string"}, ("text",), "integer",
            oracle="tests.test_agentschema.KernelTest.test_styles_preserve_functional_output",
        ),
        Capability(
            "context.plan", context_plan, "Prepare a capability invocation",
            {"capability_id": "string", "args": "dict"},
            ("capability_id", "args"), "dict",
        ),
        Capability(
            "context.observe", context_observe, "Record observed output without claiming success",
            {"value": "any"}, ("value",), "dict",
        ),
        Capability(
            "source.load_python", load_python, "Load a trusted pure Python reference",
            {"path": "string", "symbol": "string"}, ("path", "symbol"), "source",
            provenance=("legacy Python source",),
        ),
        Capability(
            "clone.propose_expression", propose_expression, "Ask a profile LLM for a restricted candidate",
            {"source": "any", "inputs": "list", "llm": "any"},
            ("source", "inputs", "llm"), "candidate",
        ),
        Capability(
            "clone.repair_expression", repair_expression,
            "Repair a rejected candidate once using deterministic feedback",
            {"source": "any", "inputs": "list", "candidate": "any", "verification": "dict", "llm": "any"},
            ("source", "inputs", "candidate", "verification", "llm"), "candidate",
        ),
        Capability(
            "clone.manual_expression", candidate_from_expression, "Wrap a manually supplied candidate",
            {"expression": "string"}, ("expression",), "candidate",
        ),
        Capability(
            "clone.register_expression", register_candidate,
            "Register only an independently accepted expression capability",
            {"functional": "any", "source": "any", "candidate": "any", "verification": "dict", "capability_id": "string"},
            ("functional", "source", "candidate", "verification", "capability_id"), "dict",
            effect="write",
        ),
        Capability(
            "test.propose_inputs", propose_inputs,
            "Generate test inputs without inventing expected outcomes",
            {"source": "any", "count": "integer", "llm": "any"},
            ("source", "count", "llm"), "list",
        ),
        Capability(
            "verify.differential", differential_verify,
            "Compare reference and candidate outcomes on independent inputs",
            {"source": "any", "inputs": "list", "candidate": "any"},
            ("source", "inputs", "candidate"), "dict",
            provenance=("agentschema.compare",),
            oracle="reference implementation outcomes",
        ),
        Capability(
            "verify.sqlite_commit", sqlite_verify,
            "Differential verification plus negative-control mutation sensitivity",
            {"source": "any", "inputs": "list", "candidate": "any"},
            ("source", "inputs", "candidate"), "dict",
            provenance=("docs/SQLITE_TESTING_PLAYBOOK.md",),
            oracle="reference outcomes and mutation rejection",
        ),
    )
    for capability in capabilities:
        functional.register(capability)
    return functional


def builtin_plans() -> tuple[ExecutionPlan, ...]:
    return (
        ExecutionPlan(
            "kiss",
            (ExecutionStep(
                "act",
                CapabilityRef("$task.capability_id"),
                args_from="$task.args",
            ),),
            "act",
            "Direct execution of a shared capability",
        ),
        ExecutionPlan(
            "pi",
            (
                ExecutionStep(
                    "prepare",
                    CapabilityRef("context.plan"),
                    {"capability_id": "$task.capability_id", "args": "$task.args"},
                ),
                ExecutionStep(
                    "act",
                    CapabilityRef("$steps.prepare.capability_id"),
                    args_from="$steps.prepare.args",
                ),
                ExecutionStep(
                    "observe",
                    CapabilityRef("context.observe"),
                    {"value": "$steps.act"},
                ),
            ),
            "act",
            "Prepare, act, and observe using the same shared capability",
        ),
        ExecutionPlan(
            "clone",
            (
                ExecutionStep(
                    "load",
                    CapabilityRef("source.load_python"),
                    {"path": "$task.args.path", "symbol": "$task.args.symbol"},
                ),
                ExecutionStep(
                    "propose",
                    CapabilityRef("clone.propose_expression"),
                    {
                        "source": "$steps.load",
                        "inputs": "$task.args.inputs",
                        "llm": "$provider",
                    },
                ),
                ExecutionStep(
                    "verify",
                    CapabilityRef("verify.sqlite_commit"),
                    {
                        "source": "$steps.load",
                        "inputs": "$task.args.inputs",
                        "candidate": "$steps.propose",
                    },
                ),
                ExecutionStep(
                    "repair",
                    CapabilityRef("clone.repair_expression"),
                    {
                        "source": "$steps.load",
                        "inputs": "$task.args.inputs",
                        "candidate": "$steps.propose",
                        "verification": "$steps.verify",
                        "llm": "$provider",
                    },
                ),
                ExecutionStep(
                    "verify_repair",
                    CapabilityRef("verify.sqlite_commit"),
                    {
                        "source": "$steps.load",
                        "inputs": "$task.args.inputs",
                        "candidate": "$steps.repair",
                    },
                ),
                ExecutionStep(
                    "register",
                    CapabilityRef("clone.register_expression"),
                    {
                        "functional": "$functional",
                        "source": "$steps.load",
                        "candidate": "$steps.repair",
                        "verification": "$steps.verify_repair",
                        "capability_id": "$task.args.capability_id",
                    },
                ),
            ),
            "register",
            "Clone, independently verify, repair at most once, and register one pure function",
            provider="deepseek",
        ),
        ExecutionPlan(
            "test",
            (
                ExecutionStep(
                    "load",
                    CapabilityRef("source.load_python"),
                    {"path": "$task.args.path", "symbol": "$task.args.symbol"},
                ),
                ExecutionStep(
                    "cases",
                    CapabilityRef("test.propose_inputs"),
                    {
                        "source": "$steps.load",
                        "count": "$task.args.count",
                        "llm": "$provider",
                    },
                ),
                ExecutionStep(
                    "candidate",
                    CapabilityRef("clone.manual_expression"),
                    {"expression": "$task.args.expression"},
                ),
                ExecutionStep(
                    "verify",
                    CapabilityRef("verify.sqlite_commit"),
                    {
                        "source": "$steps.load",
                        "inputs": "$steps.cases",
                        "candidate": "$steps.candidate",
                    },
                ),
            ),
            "verify",
            "Generate inputs with a profile LLM and verify only against reference outcomes",
            provider="deepseek",
        ),
        ExecutionPlan(
            "verification",
            (
                ExecutionStep(
                    "load",
                    CapabilityRef("source.load_python"),
                    {"path": "$task.args.path", "symbol": "$task.args.symbol"},
                ),
                ExecutionStep(
                    "candidate",
                    CapabilityRef("clone.manual_expression"),
                    {"expression": "$task.args.expression"},
                ),
                ExecutionStep(
                    "verify",
                    CapabilityRef("verify.sqlite_commit"),
                    {
                        "source": "$steps.load",
                        "inputs": "$task.args.inputs",
                        "candidate": "$steps.candidate",
                    },
                ),
            ),
            "verify",
            "SQLite-inspired differential and mutation-sensitivity commit checks",
        ),
    )
