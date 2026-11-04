"""agentschema：三个模型（上下文 / 功能 / 执行）驱动的会话式 agent 内核。

LoopEngine 驱动这三个模型实现会话循环（compose → req → parse → execute → render）：
- 功能模型 保存静态能力（稳定 ID + 契约），按「持久时长 × 影响范围」二维
  分层的可扩展注册表；工具说明从这里生成，展示/输入也是能力，可换实现；
- 执行模型（风格 Style）自带系统提示词（人设）与行为参数，MainAgent 顶层
  只认 style 和 provider，两者平级且都必选（模型是内核标配）；
- 上下文模型 保存会话消息历史、工具结果与证据；记忆框架的选择收在域内部。

本文件只做包初始化：把公开符号提到包这一层，组装 agent 的逻辑在 factory.py。
"""

from __future__ import annotations

from .context_model import ContextModel, KernelError
from .factory import create_agent
from .functional_model import Capability, CapabilityRef, FunctionalModel
from .loop import LoopEngine, MainAgent
from .style import BehaviorStyle, Style

__all__ = [
    "BehaviorStyle",
    "Capability",
    "CapabilityRef",
    "ContextModel",
    "Style",
    "FunctionalModel",
    "KernelError",
    "LoopEngine",
    "MainAgent",
    "create_agent",
]
