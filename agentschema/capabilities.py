"""内置能力与会话风格：个人助手的最小工具集。

能力按「持久时长 × 影响范围」二维分层（见 functional_model 模块注记）。
当前是助手的最小可用工具集：文本处理 + 终端输入输出。
clone / verification 等质量方法论能力已移除，后续以工具形态回归。
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .context_model import KernelError
from .functional_model import Aspect, Capability, FunctionalModel
from .prompt import DEFAULT_SYSTEM_PROMPT
from .style import BehaviorStyle, Style


def text_normalize(text: str) -> str:
    return " ".join(text.split())


def text_word_count(text: str) -> int:
    return len(text.split())


def io_input() -> str:
    """从终端读一行用户输入；EOF / Ctrl+D 视为退出。"""
    try:
        return input("> ").strip()
    except EOFError:
        return "quit"


def io_render(content: str) -> dict[str, Any]:
    """把助手输出渲染到终端。"""
    print(content)
    return {"rendered": True}


class LoggingAspect(Aspect):
    """观测切面：把「哪个 LLM、什么风格、调了什么能力」落成一行 jsonl。

    与 llm_calls.jsonl（模型侧）对称，capability_calls.jsonl 记工具侧——
    一次会话两侧都有账。拒绝发生在 before 阶段，不经过本切面。
    """

    def __init__(self, workspace: Path) -> None:
        self._path = workspace / "capability_calls.jsonl"

    def after(
        self,
        invocation: dict[str, Any],
        output: Any,
        error: str | None,
        ms: int,
    ) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "style": invocation.get("style"),
            "model": invocation.get("model"),
            "capability": invocation.get("capability_id"),
            "effect": invocation.get("effect"),
            "ok": error is None,
            "ms": ms,
            "error": error or None,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # 日志通道自己坏了只能闭嘴


def seed_functional_model() -> FunctionalModel:
    functional = FunctionalModel()
    capabilities = (
        Capability(
            "text.normalize", text_normalize, "Collapse whitespace",
            {"text": "string"}, ("text",), "string",
        ),
        Capability(
            "text.word_count", text_word_count, "Count whitespace-delimited words",
            {"text": "string"}, ("text",), "integer",
        ),
        Capability(
            "io.input", io_input, "Read one line of user input from the terminal",
            {}, (), "string", effect="write",
        ),
        Capability(
            "io.render", io_render, "Render assistant output to the terminal",
            {"content": "string"}, ("content",), "dict", effect="write",
        ),
    )
    for capability in capabilities:
        functional.register(capability)
    return functional


# 功能模型注册表：名字 → 种子函数。加新能力库只改这一行。
FUNCTIONAL_SEEDS: dict[str, Callable[[], FunctionalModel]] = {
    "builtin": seed_functional_model,
}


# 按名字解析功能模型。
def resolve_functional(name: str | None = None) -> FunctionalModel:
    """按名字 seed 出一份功能模型；None 用内置能力表。"""
    name = name or "builtin"
    seed = FUNCTIONAL_SEEDS.get(name)
    if seed is None:
        known = ", ".join(sorted(FUNCTIONAL_SEEDS))
        raise KernelError(f"unknown functional model: {name} (known: {known})")
    return seed()


# 不指定风格时用它。
DEFAULT_STYLE = "assistant"


def builtin_styles() -> tuple[Style, ...]:
    # 功能模型收束到风格里：同一次产出的所有内置风格共享同一张能力注册表。
    # 人设先挂内置默认；装配时 create_agent 会换成项目根读到的那份（见 prompt.py）。
    functional = seed_functional_model()
    return (
        Style(
            "assistant",
            "Conversational personal assistant with tool access",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            behavior=BehaviorStyle(),
            functional=functional,
        ),
    )
