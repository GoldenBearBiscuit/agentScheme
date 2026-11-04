"""功能模型：静态、可寻址的能力注册表。

Style 持有同一个注册表的引用作为收束点，但函数只在这里登记，
不在执行环节里复制。

分层注记：功能不是扁平一张表，而是按「持久时长 × 影响范围」二维分层——
引擎环节级（代码绑定）、系统提示词级（描述注入）、MCP 级（协议接入）、
skill 级（可挑选组合），持久时长从单次步骤到永久。当前实现是单层扁平表，
是该二维空间的退化特例；分层挂载机制是后续演进方向，注册接口保持不变。

切面注记：invoke 是所有能力调用的唯一咽喉，横切关注点（观测 / 权限）
以 Aspect 协议挂在这里，运行时可装卸；before 抛异常即拒绝调用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .context_model import KernelError


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


class Aspect:
    """切面：能力调用的横切关注点（观测 / 权限）。

    before 抛异常 = 拒绝本次调用（after 不会再收到通知）；
    after 收到执行结果或失败原因，用于审计与日志。
    """

    def before(self, invocation: dict[str, Any]) -> None:
        """调用前检查；拒绝就抛 KernelError。"""

    def after(
        self,
        invocation: dict[str, Any],
        output: Any,
        error: str | None,
        ms: int,
    ) -> None:
        """调用后通知；error 非 None 表示执行失败。"""


_KINDS: dict[str, Callable[[Any], bool]] = {
    "any": lambda value: True,
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "list": lambda value: isinstance(value, list),
    "dict": lambda value: isinstance(value, dict),
}

# 能力契约类型 → OpenAI JSON schema 类型；any 不约束，直接省略。
_JSON_TYPES: dict[str, str] = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "list": "array",
    "dict": "object",
}


class FunctionalModel:
    """Static, addressable capabilities; aspects wrap the single invoke throat."""

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}
        self._aspects: list[Aspect] = []
        # 会话元数据：观测三要素之二（style / model），随每次 invoke 进入 invocation。
        self._session: dict[str, str] = {}

    # 登记会话元数据：哪个模型在驱动、什么风格在会话。
    # 由 MainAgent 装配时调用；未登记时 invocation 里这两个字段为空串。
    def set_session(self, *, style: str, model: str) -> None:
        """登记观测三要素中的 style 与 model。"""
        self._session = {"style": style, "model": model}

    # 挂一个切面到调用咽喉上；顺序即执行顺序。
    def add_aspect(self, aspect: Aspect) -> None:
        """运行时装卸横切关注点。"""
        self._aspects.append(aspect)

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

    def invoke(self, capability_id: str, args: dict[str, Any]) -> Any:
        capability = self.resolve(capability_id)
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

        # 切面织入：before 可拒绝（拒绝不产生 after 通知）；
        # 执行无论成败都走 after——失败也是观测证据。
        invocation: dict[str, Any] = {
            "capability_id": capability_id,
            "args": args,
            "effect": capability.effect,
            "style": self._session.get("style", ""),
            "model": self._session.get("model", ""),
        }
        for aspect in self._aspects:
            aspect.before(invocation)
        started = time.perf_counter()
        try:
            output = capability.run(**args)
        except Exception as exc:  # noqa: BLE001 - 失败也过切面，再原样抛出
            elapsed = max(1, int((time.perf_counter() - started) * 1000))
            for aspect in self._aspects:
                aspect.after(invocation, None, str(exc), elapsed)
            raise
        elapsed = max(1, int((time.perf_counter() - started) * 1000))
        for aspect in self._aspects:
            aspect.after(invocation, output, None, elapsed)
        return output

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

    # 能力表 → OpenAI tools 数组，给 LLM 的工具说明从这里生成。
    #
    # 能力 id 里的 "." 转成 "__"（OpenAI 工具名只允许字母数字下划线连字符），
    # 引擎执行时再转回来；能力 id 本身不含 "__"，转换无歧义。
    def tools_schema(self) -> list[dict[str, Any]]:
        """生成 OpenAI function calling 的 tools 数组。"""
        tools: list[dict[str, Any]] = []
        for item in sorted(self._capabilities.values(), key=lambda value: value.capability_id):
            properties: dict[str, Any] = {}
            for name, kind in item.inputs.items():
                json_type = _JSON_TYPES.get(kind)
                properties[name] = {"type": json_type} if json_type else {}
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": item.capability_id.replace(".", "__"),
                        "description": item.summary,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": list(item.required),
                        },
                    },
                }
            )
        return tools
