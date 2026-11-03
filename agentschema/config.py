"""配置解析：把「.env + 环境变量」收敛成一个 Config。

流程位置：`create_agent()` 与 `DeepSeekProvider` 的第一步。它定下两件事——
用哪个模型/密钥，以及产物（expressions.json / llm_calls.jsonl）落在哪个 workspace。

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
DEFAULT_MODEL = "deepseek-chat"


def load_dotenv(path: Path) -> None:
    """把 .env 文件读进 os.environ（最小的 .env 解析器，零依赖）。

    入参：
        path: .env / .env.local 的路径，不存在就直接返回，不报错。

    处理：
        1. 逐行去掉首尾空白，跳过空行、`#` 注释行、没有 `=` 的行
        2. 按第一个 `=` 切成 key/value，去掉 value 两侧成对的引号
        3. 只在 os.environ 里还没有这个 key 时才写入

    出参：
        无。副作用是修改进程环境变量。
    """
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class Config:
    """一次运行的全部外部配置。

    字段：
        root: 项目根目录（默认为 cwd），用于找 .env 和相对路径。
        workspace: 产物目录；expressions.json 与 llm_calls.jsonl 都在它下面。
        api_key: DeepSeek 密钥；为空则 Provider 不可用。
        base_url / model / timeout: OpenAI 兼容端点的三件套。
        offline: 为 True 时即使有 key 也不调用 LLM。
    """

    root: Path
    workspace: Path
    api_key: str
    base_url: str
    model: str
    timeout: int
    offline: bool

    @property
    def expressions_path(self) -> Path:
        """克隆成功后固化的表达式库；重启进程时从这里恢复能力。"""
        return self.workspace / "expressions.json"

    @property
    def llm_calls_path(self) -> Path:
        """LLM 调用记录（jsonl），排查/复现用。"""
        return self.workspace / "llm_calls.jsonl"

    def ensure(self) -> None:
        """幂等地把 workspace 建出来。

        入参：无（用 self 上的路径）。
        处理：mkdir(parents=True, exist_ok=True)，已存在不报错。
        出参：无。
        """
        self.workspace.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, root: Path | None = None, offline: bool = False) -> "Config":
        """构造 Config：先落 .env，再从环境变量里取值。

        入参：
            root: 项目根；None 表示用当前工作目录（解析成绝对路径）。
            offline: 命令行 --offline 传进来的强制离线开关。

        处理：
            1. 依次 load .env / .env.local（只补空缺，不覆盖已有变量）
            2. AGENTSCHEMA_WORKSPACE 存在时按"相对 root"解析
            3. AGENTSCHEMA_OFFLINE 接受 1/true/yes/on 四种写法
            4. 各字段做 strip，避免 .env 里多打的空格把 key 弄坏

        出参：
            Config 实例；此时还没建目录，建目录是 ensure() 的事。
        """
        root = (root or Path.cwd()).resolve()
        load_dotenv(root / ".env")
        load_dotenv(root / ".env.local")
        env_ws = os.environ.get("AGENTSCHEMA_WORKSPACE", "").strip()
        workspace = (root / env_ws).resolve() if env_ws else root / "workspace"
        offline_env = os.environ.get("AGENTSCHEMA_OFFLINE", "").strip().lower()
        return cls(
            root=root,
            workspace=workspace,
            api_key=os.environ.get("DEEPSEEK_API_KEY", "").strip(),
            # 去掉尾部斜杠，后面拼 "/chat/completions" 时才不会出现双斜杠
            base_url=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            # 空字符串也算没配，回落到默认模型
            model=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            timeout=int(os.environ.get("AGENTSCHEMA_TIMEOUT", "120")),
            offline=offline or offline_env in {"1", "true", "yes", "on"},
        )
