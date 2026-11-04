"""配置解析：把「.env + 环境变量」收敛成一个 Config。

流程位置：`create_agent()` 与 `resolve_provider()` 的第一步。它定下两件事——
用哪个 LLM / 密钥，以及调用记录（llm_calls.jsonl）落在哪个 workspace。

优先级（按实际生效顺序）：
    进程已有环境变量 > .env > .env.local > 内置默认值

注意 `.env.local` 排在 `.env` 之后加载，而 load_dotenv 只补空缺，
所以 `.env` 里已经写了的键，`.env.local` 覆盖不掉——这是当前实现的行为。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# DeepSeek 的 OpenAI 兼容端点；换成别家只要改这两个默认值 + base_url 即可。
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


# 把 .env 文件读进 os.environ（最小的 .env 解析器，零依赖）。
#
# 入参：path —— .env / .env.local 的路径，不存在就直接返回，不报错。
# 出参：无。副作用是修改进程环境变量。
def load_dotenv(path: Path) -> None:
    """把 .env 读进 os.environ。"""
    # 1. 文件不在或读不动就静默返回：.env 本来就是可选的
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return

    # 2. 逐行解析，空行 / # 注释 / 没有等号的行都跳过
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)

        # 3. 去掉两侧空白与成对引号；已存在的键不覆盖，保证进程环境变量优先
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# 一次运行的全部外部配置。
#
# 字段：
#     rootpath: 项目根路径（默认为 cwd），用于找 .env 和相对路径。
#     workspace: 产物目录；llm_calls.jsonl 落在它下面。
#     api_key: DeepSeek 密钥；为空则 Provider 不可用。
#     base_url / model / timeout: OpenAI 兼容端点的三件套。
@dataclass
class Config:
    """一次运行的全部外部配置。"""

    rootpath: Path
    workspace: Path
    api_key: str
    base_url: str
    model: str
    timeout: int

    @property
    def llm_calls_path(self) -> Path:
        """LLM 调用记录（jsonl），排查/复现用。"""
        return self.workspace / "llm_calls.jsonl"

    # 幂等地把 workspace 建出来，已存在不报错。
    def ensure(self) -> None:
        """建 workspace 目录。"""
        self.workspace.mkdir(parents=True, exist_ok=True)

    # 构造 Config：先落 .env，再从环境变量里取值。
    #
    # 入参：rootpath —— 项目根；None 用当前工作目录（解析成绝对路径）。
    # 处理：.env 只补空缺，所以已有环境变量不会被文件盖掉。
    # 出参：Config 实例；此时还没建目录，建目录是 ensure() 的事。
    @classmethod
    def load(cls, rootpath: Path | None = None) -> "Config":
        """从 .env 与环境变量构造 Config。"""
        # 1. 依次落 .env / .env.local
        rootpath = (rootpath or Path.cwd()).resolve()
        load_dotenv(rootpath / ".env")
        load_dotenv(rootpath / ".env.local")

        # 2. workspace 默认 rootpath/workspace；环境变量给的是相对 rootpath 的路径
        env_ws = os.environ.get("AGENTSCHEMA_WORKSPACE", "").strip()
        workspace = (rootpath / env_ws).resolve() if env_ws else rootpath / "workspace"

        # 3. 逐项 strip：.env 里多打的空格会把 key 弄坏  cls≈self cls是类本身再用初始化赋值，self是实例
        return cls(
            rootpath=rootpath,
            workspace=workspace,
            api_key=os.environ.get("DEEPSEEK_API_KEY", "").strip(),
            # 去掉尾部斜杠，后面拼 "/chat/completions" 时才不会出现双斜杠
            base_url=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            # 空字符串也算没配，回落到默认 LLM
            model=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            timeout=int(os.environ.get("AGENTSCHEMA_TIMEOUT", "120")),
        )
