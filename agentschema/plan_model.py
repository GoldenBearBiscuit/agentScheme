"""执行模型：一个 Profile 的阶段计划（PlanModel）与步骤（ExecutionStep）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .functional_model import CapabilityRef


@dataclass(frozen=True)
class ExecutionStep:
    name: str
    capability: CapabilityRef
    args: dict[str, Any] = field(default_factory=dict)
    args_from: str | None = None


@dataclass(frozen=True)
class PlanModel:
    profile: str
    steps: tuple[ExecutionStep, ...]
    result_step: str
    summary: str = ""
    provider: str | None = None
