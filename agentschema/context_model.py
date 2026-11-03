"""上下文模型：一次运行的任务、阶段结果、证据、产物与报告。

`KernelError` 也放在这里：它是四个核心模块共享的最底层错误类型，
放在依赖根上可以避免再引入一个 errors 模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


@dataclass
class ContextModel:
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
