"""LLM 客户端：只用标准库 urllib 实现的 OpenAI 兼容调用（默认打 DeepSeek）。

流程位置：compile（证据→蓝图）、assimilate（蓝图→代码）、repair（改 run()）
三个阶段唯一的"智能"来源。没配 key 或 `--offline` 时本模块不发请求，
上层会退化成启发式规则，链路照样能跑通（只是没有真实现）。

为什么要自己解析 JSON 而不是用 SDK：项目目标之一是零依赖，
`pyproject.toml` 里 dependencies 是空的，装完就能跑。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from . import log
from .config import Config


class LLMError(RuntimeError):
    """一次调用彻底失败（重试耗尽、HTTP 4xx、返回体形状不对）。"""


class LLMUnavailable(LLMError):
    """压根没配置 LLM（无 key 或 offline）。上层据此切到启发式分支。"""


def _parse_json(text: str) -> dict[str, Any]:
    """从模型输出里抠出一个 JSON 对象。

    模型经常不老实：包 ```json 代码块、前后写「好的，结果如下」。
    这里不去猜语义，只做"取第一个 { 到最后一个 }"的粗暴截取，然后交给 json 解析。

    入参：
        text: 模型返回的原始 content 字符串。

    处理：
        1. 去掉首尾空白
        2. 若以 ``` 开头，取第一个代码块内容，并剥掉紧跟的 "json" 标记
        3. 定位首尾大括号做切片
        4. json.loads；失败时把片段塞进异常信息，便于排查是模型问题还是提示词问题

    出参：
        解析出的 dict。

    异常：
        LLMError——没有 JSON 对象，或 JSON 语法错误。
    """
    raw = (text or "").strip()
    if raw.startswith("```"):
        # count >= 2 才说明有闭合围栏，否则原样保留（模型可能只打了开头）
        raw = raw.split("```", 2)[1] if raw.count("```") >= 2 else raw
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMError(f"no JSON object in model output: {text[:300]!r}")
    candidate = raw[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMError(f"invalid JSON from model: {exc}; payload={candidate[:300]!r}") from exc


class LLM:
    """零依赖的 OpenAI 兼容客户端，顺带累计 token 用量。

    一个实例通常活一个命令的周期；`stats()` 里的数字目前只用于展示，
    ARCHITECTURE.md 里把它列进了"下一步：成本记账"。
    """

    def __init__(self, cfg: Config, tags: dict[str, Any] | None = None) -> None:
        """记住配置并清零计数器。

        入参：
            cfg: 已加载的 Config（提供 api_key / base_url / model / timeout / workspace）。
            tags: 实例级关联字段，随**每一次**对话落进 llm_calls.jsonl——
                典型是 {"phase": "repair", "span_id": "sp_...", "tool": "..."}，
                让"这次修复和 LLM 说了什么"可以被按 span 查出来。
        处理：只做赋值，不发请求、不校验 key。
        出参：实例。
        """
        self.cfg = cfg
        self.tags: dict[str, Any] = dict(tags or {})
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        # 最近一次调用的原始 usage（记忆层的 LLMBackend 需要它做 exec 观察值）
        self.last_usage: dict[str, Any] = {}

    @property
    def available(self) -> bool:
        """现在能不能真的调模型：有 key 且没被 --offline 关掉。"""
        return bool(self.cfg.api_key) and not self.cfg.offline

    def describe(self) -> str:
        """给命令行打印用的一行说明，例如 "deepseek-chat @ https://api.deepseek.com"。

        不可用时返回 "heuristic mode (原因)"，让用户一眼知道为什么结果是模板化的。
        """
        if not self.available:
            reason = "offline flag set" if self.cfg.offline else "DEEPSEEK_API_KEY missing"
            return f"heuristic mode ({reason})"
        return f"{self.cfg.model} @ {self.cfg.base_url}"

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """发一次 POST /chat/completions，带 3 次指数退避重试。

        入参：
            payload: 完整请求体（model / messages / temperature / response_format…）。

        处理：
            1. 序列化成 UTF-8 bytes，构造带 Bearer 头的 Request
            2. 最多试 3 次：
               - 429/5xx → 等 2s、4s、6s 后重试（限流和服务端抖动都是常态）
               - 其它 4xx（401 密钥错、400 参数错）→ 立刻抛，重试没有意义
               - 网络层异常（超时、连接断）→ 同样退避重试
            3. 全部失败后把最后一次的错误信息抛出去

        出参：
            解析后的响应 JSON（dict）。

        异常：
            LLMError——重试耗尽或不可重试的 HTTP 错误。
        """
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.cfg.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
            },
            method="POST",
        )
        last_error = "unknown transport error"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.cfg.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                # HTTPError 的 body 里有服务端的错误说明，尽量读出来带上
                body = ""
                try:
                    body = exc.read().decode("utf-8", "replace")[:600]
                except Exception:  # noqa: BLE001 - best effort diagnostics
                    pass
                last_error = f"HTTP {exc.code}: {body}"
                if exc.code in (429, 500, 502, 503, 504):
                    log.warning(f"llm retry in {2 * (attempt + 1)}s: {last_error}",
                                attempt=attempt + 1)
                    time.sleep(2 * (attempt + 1))
                    continue
                raise LLMError(last_error) from exc
            except Exception as exc:  # noqa: BLE001 - network layer
                last_error = repr(exc)
                log.warning(f"llm retry in {2 * (attempt + 1)}s: {last_error}",
                            attempt=attempt + 1)
                time.sleep(2 * (attempt + 1))
        raise LLMError(last_error)

    def _record_usage(self, data: dict[str, Any]) -> None:
        """从响应里把 usage 累加到实例计数器。

        入参：data: 响应 JSON。
        处理：calls +1，prompt/completion token 累加；字段缺失按 0 处理。
        出参：无。
        """
        usage = data.get("usage") or {}
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)

    def _journal(
        self,
        *,
        ok: bool,
        started: float,
        system: str,
        user: str,
        response: str = "",
        error: str = "",
        usage: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> None:
        """把一次完整对话追加进 workspace/llm_calls.jsonl（永不抛异常）。

        入参：
            ok / started / system / user: 调用结果与提示词原文。
            response / error: 二选一，成功的回答或失败的描述。
            usage: 服务端返回的 usage（失败时可能没有）。
            tags: 调用级关联字段，合并覆盖实例级 self.tags。

        处理：
            1. 组装一条：时间、phase/tool/span 关联、模型、耗时、token、
               三段文本各截断（system 400 / user 1000 / response 2000 字——
               jsonl 是给人排查和 UI 展示的，全文有需要时再加大）；
            2. 追加写盘；workspace 不存在或写失败一律静默——
               记录是增强，不能反过来弄死一次调用。

        出参：无。
        """
        merged = {**self.tags, **(tags or {})}
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "phase": merged.get("phase"),
            "tool": merged.get("tool"),
            "span_id": merged.get("span_id"),
            "model": self.cfg.model,
            "ok": ok,
            "ms": max(1, int((time.perf_counter() - started) * 1000)),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "system_head": system[:400],
            "user_head": user[:1000],
            "response_head": response[:2000],
            "error": error or None,
        }
        try:
            path = self.cfg.workspace / "llm_calls.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # 日志通道自己坏了只能闭嘴，见 log.py 的同款约定

    def complete_json(
        self, system: str, user: str, temperature: float = 0.2,
        tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """要一个 JSON 对象回来——本项目的所有 LLM 交互都走这一个入口。

        入参：
            system: 系统提示（角色 + 硬约束，例如"不许用 try/except 吞异常"）。
            user: 用户提示（证据、源码、失败栈之类）。
            temperature: 默认 0.2；转代码/结构化输出不需要发散。
            tags: 本次调用的关联字段（span_id / tool …），合并覆盖实例级。

        处理：
            1. available 为假直接抛 LLMUnavailable，让上层走启发式兜底
            2. 带上 response_format=json_object，减少模型乱写格式的概率
            3. 记 usage，取出 choices[0].message.content
            4. 交给 _parse_json 做容错解析
            5. 成功或失败都会写一条 llm_calls.jsonl（DeepSeek 的 API 是
               无状态的，平台只记用量不记内容——内容只能自己记）

        出参：
            解析后的 dict。

        异常：
            LLMUnavailable / LLMError，以及响应结构异常时的 LLMError。
        """
        if not self.available:
            raise LLMUnavailable("LLM not configured")
        started = time.perf_counter()
        data: dict[str, Any] = {}
        try:
            data = self._post(
                {
                    "model": self.cfg.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": temperature,
                    # 让服务端保证返回合法 JSON；模型不遵守时 _parse_json 再兜一层
                    "response_format": {"type": "json_object"},
                    "stream": False,
                }
            )
            self._record_usage(data)
            self.last_usage = data.get("usage") or {}
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                # 比如上游返回了 error 对象而不是 choices，这里要给出可诊断的原文
                raise LLMError(f"unexpected response shape: {json.dumps(data)[:400]}") from exc
            result = _parse_json(content)
            self._journal(
                ok=True, started=started, system=system, user=user, response=content,
                usage=data.get("usage"), tags=tags,
            )
            return result
        except LLMError as exc:
            self._journal(ok=False, started=started, system=system, user=user,
                          error=str(exc), tags=tags)
            raise

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.3,
        tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """无工具的一轮对话（记忆层的真后端用；OpenAI 兼容 messages 进/出）。

        入参：
            messages: 完整消息列表（system/user/assistant/tool）。
            temperature: 默认 0.3（事务性助手要稳，但不必完全确定）。
            tags: 本次调用的关联字段（phase="memory"、session_id）。

        处理：
            1. available 为假直接抛 LLMUnavailable；
            2. POST /chat/completions（不带 tools）；
            3. 记 usage（实例计数器 + self.last_usage）并 journal；
            4. 返回 choices[0].message 原样（role/content）。

        出参：assistant 消息 dict。

        异常：LLMUnavailable / LLMError。
        """
        if not self.available:
            raise LLMUnavailable("LLM not configured")
        started = time.perf_counter()
        system = str(messages[0].get("content") or "")[:400] if messages else ""
        user = str(messages[-1].get("content") or "")[:1000] if messages else ""
        try:
            data = self._post(
                {
                    "model": self.cfg.model,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": False,
                }
            )
            self._record_usage(data)
            self.last_usage = data.get("usage") or {}
            try:
                message = data["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMError(f"unexpected response shape: {json.dumps(data)[:400]}") from exc
            self._journal(
                ok=True, started=started, system=system, user=user,
                response=str(message.get("content") or ""), usage=data.get("usage"), tags=tags,
            )
            return message
        except LLMError as exc:
            self._journal(ok=False, started=started, system=system, user=user,
                          error=str(exc), tags=tags)
            raise

    def chat_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.3,
        tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """带工具表的对话轮（OpenAI function calling 协议，创造模式的引擎）。

        流程位置：agent.py 的对话循环每一步都调它——模型要么回文本
        （对话结束），要么回 tool_calls（agent 执行后把结果以 role:"tool"
        回填，再调本方法继续）。

        入参：
            messages: 完整对话历史（含 system；tool 结果也是其中一条）。
            tools: OpenAI tools 数组（type:function + parameters schema）。
            temperature: 创造模式默认 0.3——要一点灵活但不要发散。
            tags: 关联字段（phase="creator"、session_id），进 llm_calls.jsonl。

        处理：
            1. POST /chat/completions，带 tools + tool_choice=auto
            2. 返回 choices[0].message 原样（role/content/tool_calls）
            3. journal 记录这次对话：response 记 tool_calls 的名字与参数摘要

        出参：
            assistant 消息 dict；解析失败抛 LLMError。

        异常：
            LLMUnavailable / LLMError。
        """
        if not self.available:
            raise LLMUnavailable("LLM not configured")
        started = time.perf_counter()
        data: dict[str, Any] = {}
        try:
            data = self._post(
                {
                    "model": self.cfg.model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "temperature": temperature,
                    "stream": False,
                }
            )
            self._record_usage(data)
            self.last_usage = data.get("usage") or {}
            try:
                message = data["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMError(f"unexpected response shape: {json.dumps(data)[:400]}") from exc
            calls = message.get("tool_calls") or []
            summary = json.dumps(
                [
                    {"name": c.get("function", {}).get("name"),
                     "arguments": str(c.get("function", {}).get("arguments"))[:200]}
                    for c in calls
                ],
                ensure_ascii=False,
            ) if calls else str(message.get("content") or "")
            self._journal(
                ok=True, started=started, system=str(messages[0].get("content") or "")[:400],
                user=str(messages[-1].get("content") or "")[:1000],
                response=summary, usage=data.get("usage"), tags=tags,
            )
            return message
        except LLMError as exc:
            self._journal(ok=False, started=started,
                          system=str(messages[0].get("content") or "")[:400],
                          user=str(messages[-1].get("content") or "")[:1000],
                          error=str(exc), tags=tags)
            raise

    def stats(self) -> dict[str, int]:
        """本次运行的调用次数与 token 用量，用于命令行展示/后续成本记账。"""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
