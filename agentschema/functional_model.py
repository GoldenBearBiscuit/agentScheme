"""功能模型：静态、可寻址的能力注册表；执行风格不拥有这些函数。"""

from __future__ import annotations

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


_KINDS: dict[str, Callable[[Any], bool]] = {
    "any": lambda value: True,
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "list": lambda value: isinstance(value, list),
    "dict": lambda value: isinstance(value, dict),
}


class FunctionalModel:
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
