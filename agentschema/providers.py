"""执行拓扑的 LLM Provider；Provider 不属于功能身份。

同一份功能拓扑可以被不同 Provider 驱动：clone / test 两个 Profile 需要模型
生成"候选"或"输入用例"，但期望值永远来自参考实现，不来自模型。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import Config
from .llm import LLM


class DeepSeekProvider:
    def __init__(self, root: Path | None = None, *, offline: bool = False) -> None:
        self._llm = LLM(Config.load(root, offline=offline), tags={"phase": "agentschema"})

    @property
    def available(self) -> bool:
        return self._llm.available

    @property
    def description(self) -> str:
        return self._llm.describe()

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        return self._llm.complete_json(system, user, tags={"phase": "agentschema.clone"})


def execution_providers(root: Path | None = None, *, offline: bool = False) -> dict[str, Any]:
    return {"deepseek": DeepSeekProvider(root, offline=offline)}
