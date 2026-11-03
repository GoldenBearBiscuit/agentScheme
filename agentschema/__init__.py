"""agentschema：三拓扑（上下文 / 功能 / 执行）驱动的可验证 agent 内核。

三条拓扑共享一个 LoopEngine：
- 功能拓扑保存静态能力（稳定 ID + 契约 + Oracle），不属于任何风格；
- 执行拓扑只保存阶段计划（Profile），按 CapabilityRef 引用能力；
- 上下文拓扑保存任务、阶段结果、证据与运行报告。

克隆/测试的期望值来自参考实现实测，模型只能提供候选表达式或输入用例。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .capabilities import ExpressionStore, builtin_plans, seed_functional_topology
from .config import Config
from .topology import (
    Capability,
    CapabilityRef,
    ContextTopology,
    ExecutionPlan,
    ExecutionStep,
    FunctionalTopology,
    KernelError,
    LoopEngine,
    MainAgent,
    RunReport,
    TaskSpec,
)


def create_agent(
    root: Path | None = None,
    *,
    offline: bool = False,
    providers: dict[str, Any] | None = None,
) -> MainAgent:
    """组装一个可运行的 Agent：能力表 + 持久化表达式 + 执行 Profile。

    入参：
        root: 项目根；None 表示当前工作目录。workspace 由它派生。
        offline: 传给默认 Provider，禁止真实 LLM 调用。
        providers: 显式注入的 Provider 表（测试用）；None 时建默认 DeepSeek。

    处理：
        1. 解析配置得到 workspace
        2. 建静态功能拓扑，再从 expressions.json 恢复已固化的克隆能力
        3. 注册内置执行计划（kiss / pi / clone / test / verification）

    出参：已就绪的 MainAgent。
    """
    if providers is None:
        from .providers import execution_providers

        providers = execution_providers(root, offline=offline)
    config = Config.load(root, offline=offline)
    store = ExpressionStore(config.expressions_path)
    functional = seed_functional_topology()
    store.load(functional)
    functional.expression_store = store
    agent = MainAgent(functional, providers=providers)
    for plan in builtin_plans():
        agent.register(plan)
    return agent


__all__ = [
    "Capability",
    "CapabilityRef",
    "ContextTopology",
    "ExecutionPlan",
    "ExecutionStep",
    "FunctionalTopology",
    "KernelError",
    "LoopEngine",
    "MainAgent",
    "RunReport",
    "TaskSpec",
    "create_agent",
]
