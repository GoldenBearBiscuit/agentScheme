"""全局日志与异常的统一入口（门面层）。

流程位置：横切所有模块。**只定接口，不绑实现**——当前实现是
"print 到 stderr + 极简级别过滤"，零依赖、看得见；将来要换
logging / 落盘 / jsonl，只改 `_emit()` 一个函数，调用点一行不动。

为什么要统一（而不是继续各写各的 print）：
    1. stdout 有两个身份：cli 模式下是表格 UI，serve 模式下是 JSON-RPC
       协议通道——日志只能走 stderr。散写的 print 早晚污染其中一个；
    2. `f"{type(exc).__name__}: {exc}"` 这行格式化在 gateway / server /
       cli 手写了 N 遍，已经开始漂移；
    3. 没有级别语义，"排查开关"就无从谈起（debug 默认不该显示）。

接口一览：
    log.debug / info / warning / error   —— 日志四级
    log.format_exc(exc)                  —— 统一异常格式化
    log.fail(exc, where=...)             —— 统一异常出口（格式化 + error 日志）

`fail()` 是异常侧的核心约定：**任何边界层接住异常，都调它一次**，
拿返回的单行文案去各自的出口（CLI 打 stderr、JSON-RPC 塞 message、
HTTP 塞 body）。将来要在异常上叠加动作（trace 关联、告警），
都加在 fail() 里，边界层不用改。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

# 级别 → 数值，只用于和阈值比较（数字对齐 logging 模块，换后端时语义不变）
_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}
_DEFAULT = "info"


# 取当前级别阈值（AGENTSCHEMA_LOG_LEVEL，默认 info）。
#
# 每次都现读环境变量——测试里改了能立即生效；换实现时可以改成一次性读取，
# 接口不变。出参：级别数值。
def _threshold() -> int:
    """当前日志级别阈值。"""
    raw = os.environ.get("AGENTSCHEMA_LOG_LEVEL", "").strip().lower()
    return _LEVELS.get(raw, _LEVELS[_DEFAULT])


# 唯一的后端挂点：现在 print 到 stderr，将来换 logging 只改这里。
#
# 入参：
#     level: "debug" / "info" / "warning" / "error"。
#     message: 主体文案（调用方保证已单行化）。
#     fields: 结构化附加字段（tool=、ms=、span= 这种），拼成 k=v 追加在行尾；
#         换 jsonl 后端时它们就是行里的键值，语义已经提前定好。
# 出参：无。
def _emit(level: str, message: str, fields: dict[str, Any]) -> None:
    """把一行日志打到 stderr。"""
    # 1. 结构化字段拼成后缀
    suffix = ""
    if fields:
        suffix = "  " + " ".join(f"{key}={value}" for key, value in fields.items())

    # 2. 一行输出，立即 flush——长驻进程的日志不 flush 等于没打
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {level:<7} {message}{suffix}", file=sys.stderr, flush=True)


def _log(level: str, message: str, **fields: Any) -> None:
    """级别过滤 + 转发（四个公开函数的公共实现）。"""
    if _LEVELS[level] >= _threshold():
        _emit(level, str(message).replace("\n", " "), fields)


def debug(message: str, **fields: Any) -> None:
    """细节轨迹：tracer 树、每次 LLM 重试、热加载代数。默认不显示。"""
    _log("debug", message, **fields)


def info(message: str, **fields: Any) -> None:
    """关键转换点：阶段开始/结束、修复触发、技能落盘。默认显示。"""
    _log("info", message, **fields)


def warning(message: str, **fields: Any) -> None:
    """降级但没死：启发式兜底、策略扫描拦截、派发失败。默认显示。"""
    _log("warning", message, **fields)


def error(message: str, **fields: Any) -> None:
    """失败：调用报错、边界层兜底。默认显示。"""
    _log("error", message, **fields)


# 当前是否显示 debug 级（边界层据此决定要不要附 traceback 这类"重武器"）。
# 出参：bool，阈值 ≤ 10 即 debug 档。
def debug_enabled() -> bool:
    """当前是否开着 debug 级。"""
    return _threshold() <= 10


# 把任意异常格式化成单行文案，替代各处手写的 f"{type(exc).__name__}: {exc}"。
#
# 入参：exc —— 任意异常（含内核异常、被调用的参考实现抛的）。
# 出参：单行字符串；message 为空的异常至少给类型名。
def format_exc(exc: BaseException) -> str:
    """异常 → 单行文案。"""
    # 换行压成空格：这行要进 CLI 一行输出和 JSON-RPC message
    text = f"{type(exc).__name__}: {exc}".strip().replace("\n", " ")
    return text or type(exc).__name__


# 统一异常出口：任何边界层接住异常后调这一个函数。
#
# 入参：
#     exc: 接住的异常，任意类型。
#     where: 错误冒出来的边界位置（"cli" / "server.dispatch" / "web" …），
#         只进日志帮人定位，不进返回文案。
#     **fields: 结构化上下文（tool=、method=…），随日志走。
# 出参：单行文案，调用方拿去自己的出口（CLI 打 stderr、JSON-RPC 塞 message、
#     HTTP 塞 body）。将来要在异常上叠加动作，都加在这里。
def fail(exc: BaseException, *, where: str = "", **fields: Any) -> str:
    """格式化 + 记一条 error 日志，返回单行文案。"""
    # 1. 统一格式；有 where 就缀在行尾帮定位
    text = format_exc(exc)
    located = f"{text}  @{where}" if where else text

    # 2. 落一条 error 级日志，原始文案照旧返回
    _log("error", located, **fields)
    return text
