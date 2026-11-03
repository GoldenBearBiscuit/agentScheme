"""Three topology contracts and one profile-driven loop engine."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


class KernelError(ValueError):
    pass


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    capability_id: str
    args: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.task_id or not self.capability_id:
            raise KernelError("task_id and capability_id are required")
        if not isinstance(self.args, dict):
            raise KernelError("task args must be an object")


@dataclass(frozen=True)
class CapabilityRef:
    capability_id: str

    def __post_init__(self) -> None:
        if not self.capability_id:
            raise KernelError("capability reference cannot be empty")


@dataclass(frozen=True)
class Capability:
    capability_id: str
    run: Callable[..., Any]
    summary: str
    inputs: dict[str, str]
    required: tuple[str, ...]
    output: str
    effect: str = "read"
    provenance: tuple[str, ...] = ()
    oracle: str = ""


_KINDS: dict[str, Callable[[Any], bool]] = {
    "any": lambda value: True,
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "list": lambda value: isinstance(value, list),
    "dict": lambda value: isinstance(value, dict),
}


class FunctionalTopology:
    """Static, addressable capabilities; execution plans never own these functions."""

    def __init__(self, *, expression_store: Any = None) -> None:
        self._capabilities: dict[str, Capability] = {}
        self.expression_store = expression_store

    def register(self, capability: Capability) -> None:
        if capability.capability_id in self._capabilities:
            raise KernelError(f"duplicate capability: {capability.capability_id}")
        if not callable(capability.run):
            raise KernelError(f"capability is not callable: {capability.capability_id}")
        unknown = set(capability.inputs.values()) - set(_KINDS)
        if unknown:
            raise KernelError(f"unknown input kinds: {sorted(unknown)}")
        missing = set(capability.required) - set(capability.inputs)
        if missing:
            raise KernelError(f"required inputs have no contract: {sorted(missing)}")
        self._capabilities[capability.capability_id] = capability

    def resolve(self, reference: CapabilityRef | str) -> Capability:
        capability_id = reference.capability_id if isinstance(reference, CapabilityRef) else reference
        try:
            return self._capabilities[capability_id]
        except KeyError as exc:
            raise KernelError(f"unknown capability: {capability_id}") from exc

    def invoke(self, reference: CapabilityRef | str, args: dict[str, Any]) -> Any:
        capability = self.resolve(reference)
        unknown = set(args) - set(capability.inputs)
        missing = set(capability.required) - set(args)
        if unknown or missing:
            raise KernelError(
                f"contract mismatch for {capability.capability_id}: "
                f"unknown={sorted(unknown)}, missing={sorted(missing)}"
            )
        for name, value in args.items():
            kind = capability.inputs[name]
            if not _KINDS[kind](value):
                raise KernelError(
                    f"invalid argument {name} for {capability.capability_id}: expected {kind}"
                )
        return capability.run(**args)

    def descriptors(self) -> list[dict[str, Any]]:
        return [
            {
                "capability_id": item.capability_id,
                "summary": item.summary,
                "inputs": item.inputs,
                "required": list(item.required),
                "output": item.output,
                "effect": item.effect,
                "provenance": list(item.provenance),
                "oracle": item.oracle,
            }
            for item in sorted(self._capabilities.values(), key=lambda value: value.capability_id)
        ]


@dataclass(frozen=True)
class ExecutionStep:
    name: str
    capability: CapabilityRef
    args: dict[str, Any] = field(default_factory=dict)
    args_from: str | None = None


@dataclass(frozen=True)
class ExecutionPlan:
    profile: str
    steps: tuple[ExecutionStep, ...]
    result_step: str
    summary: str = ""
    provider: str | None = None


@dataclass
class ContextTopology:
    run_id: str
    profile: str
    task: TaskSpec
    steps: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunReport:
    run_id: str
    task_id: str
    profile: str
    status: str
    output: Any = None
    error: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "profile": self.profile,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "trace": self.trace,
        }


def _resolve(
    value: Any,
    context: ContextTopology,
    provider: Any,
    functional: FunctionalTopology,
) -> Any:
    if not isinstance(value, str) or not value.startswith("$"):
        return value
    if value == "$provider":
        return provider
    if value == "$functional":
        return functional
    parts = value[1:].split(".")
    if parts[0] == "task":
        current: Any = context.task
        parts = parts[1:]
    elif parts[0] == "steps":
        if len(parts) < 2:
            raise KernelError("step reference requires a name")
        try:
            current = context.steps[parts[1]]
        except KeyError as exc:
            raise KernelError(f"unknown step reference: {value}") from exc
        parts = parts[2:]
    else:
        raise KernelError(f"unknown context reference: {value}")
    for part in parts:
        current = current[part] if isinstance(current, dict) else getattr(current, part)
    return current


class LoopEngine:
    def __init__(self, functional: FunctionalTopology) -> None:
        self.functional = functional

    def run(self, plan: ExecutionPlan, task: TaskSpec, *, provider: Any = None) -> RunReport:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        context = ContextTopology(run_id=run_id, profile=plan.profile, task=task)
        report = RunReport(run_id, task.task_id, plan.profile, "running")
        for step in plan.steps:
            started = time.perf_counter()
            capability_id = ""
            try:
                capability_id = _resolve(
                    step.capability.capability_id, context, provider, self.functional
                )
                if not isinstance(capability_id, str):
                    raise KernelError(f"dynamic capability reference did not resolve: {step.name}")
                args = {
                    name: _resolve(value, context, provider, self.functional)
                    for name, value in step.args.items()
                }
                if step.args_from is not None:
                    dynamic_args = _resolve(
                        step.args_from, context, provider, self.functional
                    )
                    if not isinstance(dynamic_args, dict):
                        raise KernelError(f"dynamic args must be an object: {step.name}")
                    args.update(dynamic_args)
                output = self.functional.invoke(capability_id, args)
                context.steps[step.name] = output
                report.trace.append({
                    "step": step.name,
                    "capability_id": capability_id,
                    "status": "ok",
                    "ms": max(1, int((time.perf_counter() - started) * 1000)),
                })
            except Exception as exc:  # noqa: BLE001 - stage failures become run evidence
                report.status = "failed"
                report.error = f"{step.name}: {exc}"
                report.trace.append({
                    "step": step.name,
                    "capability_id": capability_id or step.capability.capability_id,
                    "status": "failed",
                    "error": str(exc),
                    "ms": max(1, int((time.perf_counter() - started) * 1000)),
                })
                return report
        report.status = "ok"
        report.output = context.steps[plan.result_step]
        return report


class MainAgent:
    """Registration and invocation control plane for interchangeable execution plans."""

    def __init__(
        self,
        functional: FunctionalTopology,
        *,
        providers: dict[str, Any] | None = None,
    ) -> None:
        self.functional = functional
        self.providers = dict(providers or {})
        self.engine = LoopEngine(functional)
        self._plans: dict[str, ExecutionPlan] = {}

    def register(self, plan: ExecutionPlan) -> None:
        if plan.profile in self._plans:
            raise KernelError(f"duplicate profile: {plan.profile}")
        if not plan.steps or len(plan.steps) > 32:
            raise KernelError("execution plan must contain 1..32 steps")
        names = [step.name for step in plan.steps]
        if len(set(names)) != len(names):
            raise KernelError(f"duplicate step name in {plan.profile}")
        if plan.result_step not in names:
            raise KernelError(f"unknown result step: {plan.result_step}")
        for step in plan.steps:
            if not step.capability.capability_id.startswith("$"):
                self.functional.resolve(step.capability)
        self._plans[plan.profile] = plan

    def agents(self) -> list[dict[str, Any]]:
        return [
            {
                "profile": plan.profile,
                "summary": plan.summary,
                "provider": plan.provider,
                "available": self._available(plan),
                "steps": [step.name for step in plan.steps],
            }
            for plan in sorted(self._plans.values(), key=lambda value: value.profile)
        ]

    def _available(self, plan: ExecutionPlan) -> bool:
        if plan.provider is None:
            return True
        provider = self.providers.get(plan.provider)
        return bool(provider is not None and getattr(provider, "available", False))

    def run(self, profile: str, task: TaskSpec) -> RunReport:
        if profile not in self._plans:
            raise KernelError(f"unknown profile: {profile}")
        plan = self._plans[profile]
        if not self._available(plan):
            run_id = f"run_{uuid.uuid4().hex[:12]}"
            return RunReport(
                run_id,
                task.task_id,
                profile,
                "unsupported",
                error=f"provider unavailable: {plan.provider}",
            )
        provider = self.providers.get(plan.provider) if plan.provider else None
        return self.engine.run(plan, task, provider=provider)
