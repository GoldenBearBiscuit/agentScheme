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

与 diag / ledger 的分工（三者并存，不要合并）：
    log    —— 运行日志：跑到哪了、哪里失败了。流式、给人看。
    diag   —— 基础设施诊断事件：结构化 jsonl，`report` 命令的素材。
    ledger —— 业务轨迹账本：intent/outcome 对，梯度挖掘的素材。

接口一览：
    log.debug / info / warning / error   —— 日志四级
    log.format_exc(exc)                  —— 统一异常格式化
    log.fail(exc, where=...)             —— 统一异常出口（格式化 + error 日志）

`fail()` 是异常侧的核心约定：**任何边界层接住异常，都调它一次**，
拿返回的单行文案去各自的出口（CLI 打 stderr、JSON-RPC 塞 message、
HTTP 塞 body）。将来要在异常上叠加动作（diag 镜像、trace 关联、
告警），都加在 fail() 里，边界层不用改。
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

# 级别 → 数值，只用于和阈值比较（数字对齐 logging 模块，换后端时语义不变）
_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}
_DEFAULT = "info"


def _threshold() -> int:
    """取当前级别阈值（AGENTSCHEMA_LOG_LEVEL，默认 info）。

    入参：无。
    处理：每次都现读环境变量——测试里改了能立即生效；换实现时
        可以改成一次性读取，接口不变。
    出参：级别数值。
    """
    raw = os.environ.get("AGENTSCHEMA_LOG_LEVEL", "").strip().lower()
    return _LEVELS.get(raw, _LEVELS[_DEFAULT])


def _emit(level: str, message: str, fields: dict[str, Any]) -> None:
    """唯一的后端挂点：现在 print 到 stderr，将来换 logging 只改这里。

    入参：
        level: "debug" / "info" / "warning" / "error"。
        message: 主体文案（调用方保证已单行化）。
        fields: 结构化附加字段（tool=、ms=、span= 这种）。
            print 实现里拼成 `k=v` 追加在行尾；换 jsonl 后端时它们
            就是行里的键值，语义已经提前定好。

    处理：`[HH:MM:SS] level  message  k=v k=v` 一行，stderr，立即 flush
        （长驻进程的日志不 flush 等于没打）。
    出参：无。
    """
    suffix = ""
    if fields:
        suffix = "  " + " ".join(f"{key}={value}" for key, value in fields.items())
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


def debug_enabled() -> bool:
    """当前是否显示 debug 级（边界层决定要不要附打印 traceback 等"重武器"）。

    入参：无。处理：阈值 ≤ 10 即 debug 档。出参：bool。
    """
    return _threshold() <= 10


def format_exc(exc: BaseException) -> str:
    """把任意异常格式化成单行文案，替代各处手写的 f"{type(exc).__name__}: {exc}"。

    入参：exc: 任意异常（含内核异常、被调用的参考实现抛的）。
    处理：`TypeName: message`；message 为空的异常至少给类型名；
        换行压成空格（这行文案要进 CLI 一行输出和 JSON-RPC message）。
    出参：单行字符串。
    """
    text = f"{type(exc).__name__}: {exc}".strip().replace("\n", " ")
    return text or type(exc).__name__


def fail(exc: BaseException, *, where: str = "", **fields: Any) -> str:
    """统一异常出口：任何边界层接住异常后调这一个函数。

    入参：
        exc: 接住的异常，任意类型。
        where: 错误冒出来的边界位置（"cli" / "server.dispatch" / "web" …），
            只进日志帮人定位，不进返回文案。
        **fields: 结构化上下文（tool=、method=…），随日志走。

    处理：
        1. format_exc 格式化（保证全项目一种格式）；
        2. error 级日志一条（带 where 定位）。
        将来的叠加动作（diag 镜像、trace_id 关联、告警）都加在这里，
        边界层不用改。

    出参：单行文案。调用方拿去自己的出口：
        cli         → 直接用（main 兜底已打印，通常无需再打）
        server      → 塞进 JSON-RPC error.message
        web         → 塞进 {"ok": False, "error": ...} 的 body
    """
    text = format_exc(exc)
    located = f"{text}  @{where}" if where else text
    _log("error", located, **fields)
    return text
