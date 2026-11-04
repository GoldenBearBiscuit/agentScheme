"""主循环：会话式引擎——turn 单轮，loop 死循环。

每轮五个环节：compose（上下文组装）→ req（发请求）→ parse（解析响应）
→ execute（执行工具）→ render（展示）。模型是内核标配：MainAgent 一定
持有 provider，没有就在装配期报错，引擎内不再做可用性门控。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from typing import Any

from .context_model import ContextModel, KernelError
from .functional_model import FunctionalModel
from .style import Style


class LoopEngine:
    """分环节会话引擎。上下文由 MainAgent 开启并持有，引擎只管推进。"""

    # 会话主循环：一直跑到用户退出（quit / exit / Ctrl+D）。
    #
    # 入参：style —— 当前风格；provider —— 模型（必选）；context —— 会话上下文。
    # 出参：结束时的上下文（留证据用）。
    def loop(self, style: Style, provider: Any, context: ContextModel) -> ContextModel:
        """会话主循环，直到用户退出。"""
        functional = self._functional(style)

        # compose：会话开始时注入一次（turn 单独调用时也会兜到这里）
        self._ensure_composed(style, functional, context)
        functional.invoke("io.render", {"content": f"[{style.style}] 助手已启动，quit/exit 退出"})

        while True:
            try:
                user_input = str(functional.invoke("io.input", {}))
            except EOFError:  # Ctrl+D 与 quit 同义
                user_input = "quit"
            if user_input.strip().lower() in ("quit", "exit"):
                functional.invoke("io.render", {"content": "再见。"})
                return context
            if not user_input:
                continue
            self.turn(style, provider, context, user_input)

    # 单轮对话：处理一条用户输入，跑完五个环节。
    #
    # 模型要么直接回复（render 后本轮结束），要么发起工具调用
    # （执行后把结果回填消息，继续请求直到模型回复或超轮数上限）。
    #
    # 出参：本轮助手最终的文本回复。
    def turn(
        self,
        style: Style,
        provider: Any,
        context: ContextModel,
        user_input: str,
    ) -> str:
        """处理一条用户输入，返回助手最终回复。"""
        functional = self._functional(style)
        self._ensure_composed(style, functional, context)
        context.messages.append({"role": "user", "content": user_input})

        for round_index in range(style.behavior.max_tool_rounds):
            # req：带工具表发一轮请求
            message = provider.chat_tools(context.messages, functional.tools_schema())

            # parse：工具调用 or 文本回复
            calls = message.get("tool_calls") or []
            reply = str(message.get("content") or "")

            if not calls:
                # render：直接回复，本轮结束
                context.messages.append({"role": "assistant", "content": reply})
                context.evidence.append({"round": round_index, "action": "reply"})
                functional.invoke("io.render", {"content": reply})
                return reply

            # execute：逐个执行工具调用，结果以 role:"tool" 回填
            context.messages.append({
                "role": "assistant",
                "content": reply,
                "tool_calls": calls,
            })
            for call in calls:
                result = self._invoke_tool(functional, context, call)
                context.messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": result,
                })

        # 超出工具轮数上限：不再让模型继续调工具，如实提示
        notice = f"已达单轮工具调用上限（{style.behavior.max_tool_rounds}），本轮结束。"
        context.messages.append({"role": "assistant", "content": notice})
        context.evidence.append({"action": "tool_round_limit"})
        functional.invoke("io.render", {"content": notice})
        return notice

    # 执行单个工具调用：名字转回能力 id，契约校验与调用都走功能表；
    # 任何失败都转成 JSON 错误载荷回给模型，而不是炸掉整轮。
    def _invoke_tool(
        self,
        functional: FunctionalModel,
        context: ContextModel,
        call: dict[str, Any],
    ) -> str:
        """执行一个工具调用，返回给模型的 JSON 结果文本。"""
        function = call.get("function") or {}
        name = str(function.get("name") or "").replace("__", ".")
        try:
            args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            payload = {"ok": False, "error": f"invalid tool arguments: {exc}"}
        else:
            try:
                output = functional.invoke(name, args)
                context.steps[name] = output
                context.evidence.append({
                    "tool": name,
                    "args": args,
                    "status": "ok",
                })
                payload = {"ok": True, "result": output}
            except Exception as exc:  # noqa: BLE001 - 工具失败是证据，不是崩溃
                payload = {"ok": False, "error": str(exc)}
                context.evidence.append({
                    "tool": name,
                    "args": args,
                    "status": "failed",
                    "error": str(exc),
                })
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _functional(self, style: Style) -> FunctionalModel:
        if style.functional is None:
            raise KernelError(f"style is not bound to a functional model: {style.style}")
        return style.functional

    # compose 环节：系统提示词 + 工具说明，只在会话第一条消息前注入一次。
    def _ensure_composed(
        self,
        style: Style,
        functional: FunctionalModel,
        context: ContextModel,
    ) -> None:
        """确保会话消息已带系统提示词与工具说明。"""
        if context.messages:
            return
        context.messages.append({"role": "system", "content": style.system_prompt})
        context.messages.append({
            "role": "system",
            "content": "可用工具：\n" + json.dumps(
                functional.tools_schema(), ensure_ascii=False, indent=2
            ),
        })


class MainAgent:
    """会话式 agent 的注册与控制面。

    顶层只有两样东西：主 style（自带功能模型）和 provider（模型必选）。
    agent 初始化即开启会话上下文——三域在这里对称齐了：
    functional（功能域）、context（上下文域）、engine（执行域）。
    """

    def __init__(self, style: Style, *, provider: Any) -> None:
        if style.functional is None:
            raise KernelError("primary style must carry a functional model")
        if provider is None:
            raise KernelError("agent requires a provider (model is mandatory)")
        self.functional: FunctionalModel = style.functional
        self.provider = provider
        # 登记观测三要素之二：哪个 LLM、什么风格（capability 由 invoke 时补全）
        self.functional.set_session(
            style=style.style,
            model=str(getattr(provider, "model", "unknown")),
        )
        self.engine = (style.engine or LoopEngine)()
        self.style = style.style
        # 会话上下文：agent 生命周期 = 会话生命周期
        self.context: ContextModel = style.new_context(f"run_{uuid.uuid4().hex[:12]}")
        self._styles: dict[str, Style] = {}
        self.register(style)

    def register(self, style: Style) -> None:
        if style.style in self._styles:
            raise KernelError(f"duplicate style: {style.style}")
        # 绑定到 agent 的注册表：裸 style（functional=None）或自带他表的 style，
        # 都换成主 style 那张，保证动态注册的能力跨风格可见。
        if style.functional is not self.functional:
            style = replace(style, functional=self.functional)
        self._styles[style.style] = style

    def agents(self) -> list[dict[str, Any]]:
        return [
            {
                "style": style.style,
                "summary": style.summary,
                "default": style.style == self.style,
                "tools": len(self.functional.descriptors()),
            }
            for style in sorted(self._styles.values(), key=lambda value: value.style)
        ]

    # 会话主循环入口：一直跑到用户退出。
    def loop(self) -> ContextModel:
        """进入会话循环，返回结束时的上下文。"""
        return self.engine.loop(self._styles[self.style], self.provider, self.context)

    # 单轮入口：处理一条用户输入，返回助手最终回复（测试与编程用）。
    def turn(self, user_input: str) -> str:
        """处理一条用户输入。"""
        return self.engine.turn(self._styles[self.style], self.provider, self.context, user_input)


# 执行引擎注册表：名字 → 引擎类型。加新引擎只改这一行。
ENGINES: dict[str, type] = {
    "loop": LoopEngine,
}


# 按名字解析执行引擎类。
def resolve_engine(name: str | None = None) -> type:
    """按名字解析执行引擎类；None 用标准 LoopEngine。"""
    name = name or "loop"
    engine = ENGINES.get(name)
    if engine is None:
        known = ", ".join(sorted(ENGINES))
        raise KernelError(f"unknown engine: {name} (known: {known})")
    return engine
