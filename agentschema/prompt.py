"""系统提示词：内置默认人设 + 项目根的覆盖文件。

流程位置：`create_agent()` 的第二步——装配前先拿到项目根，再决定这个会话
用哪份人设。提示词是风格的一部分（见 style.py），但它的内容是项目配置，
所以从根目录读，不写死在能力表里。

优先级：项目根 `SYSTEM_PROMPT.md` > 内置 `DEFAULT_SYSTEM_PROMPT`。
缺文件用默认；文件存在但读出来是空白算坏配置，装配期直接报错。
"""

from __future__ import annotations

from pathlib import Path

from .context_model import KernelError

# 项目根的覆盖文件名：存在即整份替换内置人设。
SYSTEM_PROMPT_FILE = "SYSTEM_PROMPT.md"

# 个人助手的默认人设：人设 + 工具调用纪律。
DEFAULT_SYSTEM_PROMPT = """你是 agentschema，一个中文优先的个人助手。
- 用与用户一致的语言对话；先理解意图，再决定是否需要工具。
- 需要执行操作时调用提供的工具，一次只调一个；工具失败就如实转述，不要编造结果。
- 不确定就向用户提问，不要猜。
"""


# 读项目根的系统提示词：文件不在就用内置默认。
#
# 入参：rootpath —— 项目根目录（已 resolve 的绝对路径）。
# 出参：提示词全文，保证去掉首尾空白后非空。
# 异常：KernelError —— 文件读不出来，或内容全是空白。
def load_system_prompt(rootpath: Path) -> str:
    """从项目根读系统提示词，没有就用内置默认。"""
    # 1. 文件不在：默认人设本身就是可用配置，不算错误
    path = rootpath / SYSTEM_PROMPT_FILE
    if not path.is_file():
        return DEFAULT_SYSTEM_PROMPT

    # 2. 读全文：读不动（编码错 / 权限）是坏配置，别悄悄退回默认值
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise KernelError(f"unreadable system prompt: {path} ({exc})") from exc

    # 3. 全空白等于没人设，装配期拦住，不留到会话里才发现
    if not text.strip():
        raise KernelError(f"empty system prompt: {path}")
    return text.strip()
