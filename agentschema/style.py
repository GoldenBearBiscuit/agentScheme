"""执行模型：一个会话风格（Style）及其行为参数。

风格是静态侧的收束点：自带系统提示词（人设）、行为参数与功能注册表。
MainAgent 顶层只认 style 和 provider，两者平级——模型是内核标配，
provider 对象不进 style。

功能不是扁平一张表：按「持久时长 × 影响范围」二维分层——有的固定在
引擎环节里（API 级），有的进系统提示词，有的是 MCP / skill 级。
分层挂载机制待设计，暂以单表占位。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context_model import ContextModel, resolve_context
from .functional_model import FunctionalModel


@dataclass(frozen=True)
class BehaviorStyle:
    """行为风格：会话循环的参数。"""

    # 一次用户输入最多连续的工具调用轮数；超限强制结束本轮并提示。
    max_tool_rounds: int = 8


@dataclass(frozen=True)
class Style:
    style: str
    summary: str = ""
    # 系统提示词：这个风格的人设与硬约束（个人助手）。
    system_prompt: str = ""
    behavior: BehaviorStyle = BehaviorStyle()
    # 静态内核机制：怎么执行（见 loop.ENGINES）。
    engine: type | None = None
    # 功能注册表随风格一起进来（分层见模块注记）。
    functional: FunctionalModel | None = None

    def new_context(self, run_id: str) -> ContextModel:
        """为一次会话开启上下文。

        记忆框架的选择收在上下文域内部（resolve_context），风格不感知。
        """
        context_model = resolve_context()
        return context_model(run_id=run_id, style=self.style)
