"""上下文模型：一次会话的消息历史、工具结果、证据与产物。

`KernelError` 也放在这里：它是核心模块共享的最底层错误类型，
放在依赖根上可以避免再引入一个 errors 模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class KernelError(ValueError):
    pass


@dataclass
class ContextModel:
    """一次会话的上下文：messages 是对话历史，steps 记工具结果。"""

    run_id: str
    style: str
    # 对话历史：{"role": "system"|"user"|"assistant"|"tool", "content": ...}，
    # 兼容 OpenAI messages 协议，可直接发给 LLM。
    messages: list[dict[str, Any]] = field(default_factory=list)
    # 工具调用结果的记账：工具名 → 最近一次结果
    steps: dict[str, Any] = field(default_factory=dict)
    # 证据链：每轮动作的轻量记录（谁、用什么工具、结果状态）
    evidence: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


# 上下文模型注册表：名字 → 类。记忆框架的选择收在上下文域内部——
# new_context() 在域内解析（当前默认唯一），风格与装配层不感知。
# 加新上下文模型只改这一行。
CONTEXT_MODELS: dict[str, type] = {
    "default": ContextModel,
}


# 按名字解析上下文模型类。
def resolve_context(name: str | None = None) -> type:
    """按名字解析上下文模型类；None 用 default。"""
    name = name or "default"
    context_model = CONTEXT_MODELS.get(name)
    if context_model is None:
        known = ", ".join(sorted(CONTEXT_MODELS))
        raise KernelError(f"unknown context model: {name} (known: {known})")
    return context_model
