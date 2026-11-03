"""主循环：解析引用、执行单个计划、注册与调度多个计划。"""

from __future__ import annotations

import time
import uuid
from typing import Any

from .context_model import ContextModel, KernelError, RunReport, TaskSpec
from .functional_model import FunctionalModel
from .plan_model import PlanModel


def _resolve(
    value: Any,
    context: ContextModel,
    provider: Any,
    functional: FunctionalModel,
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
    def __init__(self, functional: FunctionalModel) -> None:
        self.functional = functional

    def run(self, plan: PlanModel, task: TaskSpec, *, provider: Any = None) -> RunReport:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        context = ContextModel(run_id=run_id, profile=plan.profile, task=task)
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
        functional: FunctionalModel,
        *,
        providers: dict[str, Any] | None = None,
    ) -> None:
        self.functional = functional
        self.providers = dict(providers or {})
        self.engine = LoopEngine(functional)
        self._plans: dict[str, PlanModel] = {}

    def register(self, plan: PlanModel) -> None:
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

    def _available(self, plan: PlanModel) -> bool:
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
