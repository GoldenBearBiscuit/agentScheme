"""LLM 客户端：只用标准库 urllib 实现的 OpenAI 兼容调用（默认打 DeepSeek）。

没配 key 时本模块不发请求，上层会退化成启发式规则，链路照样能跑通
（只是没有真实现）。

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
    """压根没配置 LLM（没有 key）。上层据此切到启发式分支。"""


# 从 LLM 输出里抠出一个 JSON 对象。
#
# LLM 经常不老实：包 ```json 代码块、前后写「好的，结果如下」。
# 这里不去猜语义，只做"取第一个 { 到最后一个 }"的粗暴截取，再交给 json 解析。
#
# 入参：text —— LLM 返回的原始 content 字符串。
# 出参：解析出的 dict。
# 异常：LLMError —— 没有 JSON 对象，或 JSON 语法错误。
def _parse_json(text: str) -> dict[str, Any]:
    """从模型输出里抠出 JSON 对象。"""
    # 1. 去首尾空白
    raw = (text or "").strip()

    # 2. 剥掉 ``` 围栏和紧跟的 json 标记；count >= 2 才说明有闭合围栏
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1] if raw.count("```") >= 2 else raw
        if raw.lstrip().startswith("json"):
            raw = raw.lstrip()[4:]

    # 3. 定位首尾大括号做切片
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMError(f"no JSON object in model output: {text[:300]!r}")
    candidate = raw[start : end + 1]

    # 4. 解析；失败时把片段塞进异常，便于判断是模型问题还是提示词问题
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMError(f"invalid JSON from model: {exc}; payload={candidate[:300]!r}") from exc


# 零依赖的 OpenAI 兼容客户端，顺带累计 token 用量。
#
# 一个实例通常活一个命令的周期；stats() 里的数字目前只用于展示。
class LLM:
    """零依赖的 OpenAI 兼容客户端。"""

    # 记住配置并清零计数器。只做赋值，不发请求、不校验 key。
    #
    # 入参：
    #     cfg: 已加载的 Config（api_key / base_url / model / timeout / workspace）。
    #     tags: 实例级关联字段，随**每一次**对话落进 llm_calls.jsonl——
    #         典型是 {"phase": "clone", "span_id": "sp_..."}，
    #         让"这次调用和 LLM 说了什么"可以按 span 查出来。
    def __init__(self, cfg: Config, tags: dict[str, Any] | None = None) -> None:
        """记住配置并清零计数器。"""
        self.cfg = cfg
        self.tags: dict[str, Any] = dict(tags or {})
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        # 最近一次调用的原始 usage（记忆层的 LLMBackend 需要它做 exec 观察值）
        self.last_usage: dict[str, Any] = {}

    @property
    def available(self) -> bool:
        """现在能不能真的调LLM：有 key 就能。"""
        return bool(self.cfg.api_key)

    # 给命令行打印用的一行说明，例如 "deepseek-flash @ https://api.deepseek.com"。
    # 不可用时返回 "heuristic mode (原因)"，让人一眼知道为什么结果是模板化的。
    def describe(self) -> str:
        """模型与端点，或不可用的原因。"""
        if not self.available:
            return "heuristic mode (DEEPSEEK_API_KEY missing)"
        return f"{self.cfg.model} @ {self.cfg.base_url}"

    # 发一次 POST /chat/completions，带 3 次指数退避重试。
    #
    # 入参：payload —— 完整请求体（model / messages / temperature / response_format…）。
    # 出参：解析后的响应 JSON。
    # 异常：LLMError —— 重试耗尽，或碰到不可重试的 HTTP 错误。
    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /chat/completions，按 2s / 4s / 6s 退避重试。"""
        # 1. 序列化 + 构造带 Bearer 头的请求
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

        # 2. 最多试 3 次：限流与网络抖动退避重试，其它 4xx 立刻抛（重试没有意义）
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

        # 3. 三次都失败，把最后一次的错误抛出去
        raise LLMError(last_error)

    # 从响应里把 usage 累加到实例计数器。
    # 入参：data —— 响应 JSON；字段缺失按 0 处理。
    def _record_usage(self, data: dict[str, Any]) -> None:
        """把 usage 累加到实例计数器。"""
        usage = data.get("usage") or {}
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)

    # 把一次完整对话追加进 workspace/llm_calls.jsonl（永不抛异常）。
    #
    # 入参：
    #     ok / started / system / user: 调用结果与提示词原文。
    #     response / error: 二选一，成功的回答或失败的描述。
    #     usage: 服务端返回的 usage（失败时可能没有）。
    #     tags: 调用级关联字段，合并覆盖实例级 self.tags。
    # 出参：无。记录是增强，写不进去也不能反过来弄死一次调用。
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
        """追加一条对话记录到 llm_calls.jsonl。"""
        # 1. 组装记录：关联字段、耗时、token，三段文本各截断
        #    （jsonl 是给人排查和 UI 展示的，全文有需要时再加大）
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

        # 2. 追加写盘；目录不在或写失败一律静默——记录不该影响主流程
        try:
            path = self.cfg.workspace / "llm_calls.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # 日志通道自己坏了只能闭嘴

    # 要一个 JSON 对象回来——本项目的所有 LLM 交互都走这一个入口。
    #
    # 入参：
    #     system: 系统提示（角色 + 硬约束）。
    #     user: 用户提示（证据、源码、失败栈之类）。
    #     temperature: 默认 0.2；转代码 / 结构化输出不需要发散。
    #     tags: 本次调用的关联字段，合并覆盖实例级。
    # 出参：解析后的 dict。
    # 异常：LLMUnavailable / LLMError，以及响应结构异常时的 LLMError。
    def complete_json(
        self, system: str, user: str, temperature: float = 0.2,
        tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """调一次模型，返回解析好的 JSON 对象。"""
        # 1. 不可用就抛出去，让上层走启发式兜底，而不是在这里硬撑
        if not self.available:
            raise LLMUnavailable("LLM not configured")
        started = time.perf_counter()
        data: dict[str, Any] = {}
        try:
            # 2. 带 response_format=json_object，减少模型乱写格式的概率
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

            # 3. 记 usage，取正文
            self._record_usage(data)
            self.last_usage = data.get("usage") or {}
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                # 比如上游返回了 error 对象而不是 choices，这里要给出可诊断的原文
                raise LLMError(f"unexpected response shape: {json.dumps(data)[:400]}") from exc

            # 4. 容错解析，再把这次对话落盘
            result = _parse_json(content)
            self._journal(
                ok=True, started=started, system=system, user=user, response=content,
                usage=data.get("usage"), tags=tags,
            )
            return result
        except LLMError as exc:
            # 5. 失败也要落一条，否则排查时只看到成功记录
            self._journal(ok=False, started=started, system=system, user=user,
                          error=str(exc), tags=tags)
            raise

    # 无工具的一轮对话（OpenAI 兼容 messages 进 / 出）。
    #
    # 入参：
    #     messages: 完整消息列表（system / user / assistant / tool）。
    #     temperature: 默认 0.3（事务性助手要稳，但不必完全确定）。
    #     tags: 本次调用的关联字段（phase / session_id）。
    # 出参：assistant 消息 dict（role / content）。
    # 异常：LLMUnavailable / LLMError。
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.3,
        tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """不带 tools 的一轮对话。"""
        # 1. 不可用直接抛；顺手取首尾消息作为 journal 的摘要
        if not self.available:
            raise LLMUnavailable("LLM not configured")
        started = time.perf_counter()
        system = str(messages[0].get("content") or "")[:400] if messages else ""
        user = str(messages[-1].get("content") or "")[:1000] if messages else ""
        try:
            # 2. 发请求（不带 tools），记 usage
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

            # 3. 落盘后原样返回 assistant 消息
            self._journal(
                ok=True, started=started, system=system, user=user,
                response=str(message.get("content") or ""), usage=data.get("usage"), tags=tags,
            )
            return message
        except LLMError as exc:
            self._journal(ok=False, started=started, system=system, user=user,
                          error=str(exc), tags=tags)
            raise

    # 带工具表的对话轮（OpenAI function calling 协议）。
    #
    # 调用方每一步都调它：模型要么回文本（对话结束），要么回 tool_calls
    # （执行后把结果以 role:"tool" 回填，再调本方法继续）。
    #
    # 入参：
    #     messages: 完整对话历史（含 system；tool 结果也是其中一条）。
    #     tools: OpenAI tools 数组（type:function + parameters schema）。
    #     temperature: 默认 0.3——要一点灵活但不要发散。
    #     tags: 关联字段，进 llm_calls.jsonl。
    # 出参：assistant 消息 dict（role / content / tool_calls）。
    # 异常：LLMUnavailable / LLMError。
    def chat_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.3,
        tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """带 tools 的一轮对话，返回 assistant 消息。"""
        # 1. 不可用直接抛
        if not self.available:
            raise LLMUnavailable("LLM not configured")
        started = time.perf_counter()
        data: dict[str, Any] = {}
        try:
            # 2. 带 tools + tool_choice=auto 发请求
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

            # 3. journal 记工具名与参数摘要；没有 tool_calls 就记正文
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
