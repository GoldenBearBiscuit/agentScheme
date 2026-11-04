"""装配厂：把一个可运行的 MainAgent 拼出来。

流程位置：`create_agent()` 是外部唯一入口，`__init__.py` 只是把它再导出一次。
顶层只把两样东西交给 MainAgent：style 和 provider，两者平级且都必选——
模型是内核标配，provider 不可用在装配期就报错，不留到会话里才炸。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .capabilities import (
    DEFAULT_STYLE,
    LoggingAspect,
    builtin_styles,
    resolve_functional,
)
from .config import Config
from .context_model import KernelError
from .loop import MainAgent, resolve_engine
from .prompt import load_system_prompt
from .providers import DEFAULT_PROVIDER, resolve_provider


# 组装一个可运行的 Agent：按 style 名选一个风格，从项目根装入系统提示词，
# 可插拔件按名字解析后挂上，解析 Provider（必选，不可用直接报错），
# 把 style 和 provider 交给 MainAgent。
#
# 入参：
#     rootpath: 项目根路径；None 表示当前工作目录。workspace 与系统提示词都由它派生。
#     style: 必须是内置风格之一；None 用默认的 assistant。
#     provider: 模型系列名，必须在 providers.json 里存在；None 用默认的 deepseek-flash。
#     functional: 功能模型种子名，见 FUNCTIONAL_SEEDS；None 用内置能力表。
#     engine: 执行引擎名，见 ENGINES；None 用标准 loop 引擎。
# 出参：已就绪的 Agent。
def create_agent(
    rootpath: Path | None = None,
    *,
    style: str | None = None,
    provider: str | None = None,
    functional: str | None = None,
    engine: str | None = None,
) -> MainAgent:
    """按 style 装配一个可运行的 agent。"""
    # 1. 校验风格名：内置表里有才放行，报错时把已知名字一起给出。
    #    功能模型随风格一起返回，装配层不再单独 seed 一张能力表。
    style = style or DEFAULT_STYLE
    builtin = {item.style: item for item in builtin_styles()}
    if style not in builtin:
        known = ", ".join(sorted(builtin))
        raise KernelError(f"unknown style: {style} (known: {known})")

    # 2. 收集：先定根目录，再读系统提示词——提示词属于项目配置，装配前就要齐
    rootpath = (rootpath or Path.cwd()).resolve()
    system_prompt = load_system_prompt(rootpath)

    # 3. 处理：把项目根的人设换进风格，可插拔件按名字查注册表，None 用内置
    selected = replace(
        builtin[style],
        system_prompt=system_prompt,
        functional=resolve_functional(functional),
        engine=resolve_engine(engine),
    )

    # 4. 加载配置并解析 Provider：模型是内核标配，没配 key 当场报错
    config = Config.load(rootpath)
    resolved = resolve_provider(provider, config)
    if not getattr(resolved, "available", False):
        raise KernelError(
            f"provider unavailable: {provider or DEFAULT_PROVIDER} "
            f"(missing API key? see .env.example)"
        )

    # 5. 装配：顶层只有 style 和 provider，两者平级；观测切面默认开
    selected.functional.add_aspect(LoggingAspect(config.workspace))
    return MainAgent(selected, provider=resolved)
