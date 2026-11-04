"""执行模型的 LLM Provider；Provider 不属于功能身份。

同一份功能模型可以被不同 Provider 驱动：clone / test 两个 Profile 需要 LLM
生成"候选"或"输入用例"，但期望值永远来自参考实现，不来自 LLM。

Provider 用「模型系列名」寻址：名字到配置的映射写在项目根的 providers.json
（一个对象列表，人可以改）。连接信息（密钥、端点、超时）优先取条目里的字段，
没写的回落到 .env / 环境变量 / 内置默认值。
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import Config
from .context_model import KernelError
from .llm import LLM

# 默认模型系列：不指定 provider 时用它。目前只有 DeepSeek 一家。
DEFAULT_PROVIDER = "deepseek-flash"
PROVIDER_FILE = "providers.json"

# providers.json 缺失时的兜底，保证零配置也能跑。
_BUILTIN_SPEC = {"name": DEFAULT_PROVIDER, "vendor": "deepseek", "model": DEFAULT_PROVIDER}


# DeepSeek 的实现：可用性判定与对话调用都转给 LLM 层，
# 这一层只负责补 phase 标签，方便在 llm_calls.jsonl 里区分来源。
class DeepSeekProvider:
    def __init__(self, config: Config) -> None:
        self._llm = LLM(config, tags={"phase": "agentschema"})

    @property
    def available(self) -> bool:
        return self._llm.available

    @property
    def model(self) -> str:
        """模型名：观测三要素之一（哪个 LLM）。"""
        return self._llm.cfg.model

    @property
    def description(self) -> str:
        return self._llm.describe()

    # 带工具表的对话轮：引擎会话循环的标准请求入口。
    def chat_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """发一轮对话（带工具说明），返回 assistant 消息。"""
        return self._llm.chat_tools(messages, tools)


# 读 providers.json，返回 name → 配置条目的表。
#
# 入参：rootpath —— 项目根；None 用当前工作目录。
# 出参：name → 该条目的原始 dict。
def load_provider_table(rootpath: Path | None = None) -> dict[str, dict[str, Any]]:
    """读 providers.json，返回 name → 条目。"""
    # 1. 文件不存在就给内置兜底，零配置也能跑
    path = (rootpath or Path.cwd()) / PROVIDER_FILE
    if not path.is_file():
        return {DEFAULT_PROVIDER: dict(_BUILTIN_SPEC)}

    # 2. 读不动或不是 JSON 时报错，不悄悄退回默认值——否则改错文件会以为是模型的问题
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KernelError(f"cannot read {PROVIDER_FILE}: {exc}") from exc

    # 3. 顶层必须是对象列表
    if not isinstance(data, list):
        raise KernelError(f"{PROVIDER_FILE} must be a list of objects")

    # 4. 逐条校验 name 是字符串；重名后者覆盖前者
    table: dict[str, dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise KernelError(f"{PROVIDER_FILE} entries need a string name")
        table[item["name"]] = item
    return table


# 按模型系列名取一个可用的 Provider。
#
# 入参：
#     name: providers.json 里的 name；None 用 DEFAULT_PROVIDER。
#     config: 装配时已加载好的 Config；None 时自己加载一份。
# 出参：具备 available 与 complete_json 的对象。
def resolve_provider(name: str | None = None, config: Config | None = None) -> Any:
    """按名字解析出一个 Provider。"""
    # 1. 只认字符串：字典与实例在这里被拒，避免旧写法静默变成"不可用"
    if name is None:
        name = DEFAULT_PROVIDER
    if not isinstance(name, str):
        raise KernelError(f"provider must be a model name, got {type(name).__name__}")

    # 2. 查表；不在表里就把已知名字一起报出来
    config = config or Config.load()
    table = load_provider_table(config.rootpath)
    spec = table.get(name)
    if spec is None:
        known = ", ".join(sorted(table)) or "(empty)"
        raise KernelError(f"unknown provider: {name} (known: {known})")
    return _build(spec, config)


# vendor 注册表：厂商名 → (Provider 类型, 默认密钥环境变量)。
# 判断标准是"这个 vendor 能不能解析出一个 LLM Provider 类型"：
# 在表里才算认识的厂商，不在就报错。加新厂商只改这一行，不动 _build。
_VENDORS: dict[str, tuple[type, str]] = {
    "deepseek": (DeepSeekProvider, "DEEPSEEK_API_KEY"),
}


# 把一条 providers.json 条目变成 Provider。
#
# vendor 决定用哪个 Provider 类型；model 没写就用条目的 name；连接信息
# （端点 / 密钥 / 超时）没写就沿用传入的 config（来自 .env 与环境变量），
# 所以文件可以只声明"有哪些模型"。
def _build(spec: dict[str, Any], config: Config) -> Any:
    """按一条条目造 Provider。"""
    # 1. 查 vendor 注册表：能对上类型才放行，报错时把已知厂商一起给出
    vendor = str(spec.get("vendor") or "deepseek")
    entry = _VENDORS.get(vendor)
    if entry is None:
        known = ", ".join(sorted(_VENDORS))
        raise KernelError(f"unsupported vendor: {vendor} (known: {known})")
    provider_cls, default_key_env = entry

    # 2. 条目字段盖在 config 上，没写的沿用原值
    api_key_env = str(spec.get("api_key_env") or default_key_env)
    config = replace(
        config,
        model=str(spec.get("model") or spec["name"]),
        base_url=str(spec.get("base_url") or config.base_url).rstrip("/"),
        api_key=os.environ.get(api_key_env, "").strip(),
        timeout=int(spec.get("timeout") or config.timeout),
    )
    return provider_cls(config)
