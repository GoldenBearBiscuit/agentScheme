from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentschema import (
    BehaviorStyle,
    KernelError,
    Style,
    create_agent,
)


class FakeProvider:
    """假模型：按预设顺序回文本或工具调用，记录每次请求。"""

    available = True
    model = "fake-model"

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat_tools(self, messages: list, tools: list) -> dict:
        self.calls.append({
            "messages": [dict(item) for item in messages],
            "tools": tools,
        })
        item = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(item, str):
            return {"role": "assistant", "content": item}
        return {"role": "assistant", "content": "", "tool_calls": item}


class UnavailableProvider:
    available = False


def tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


class KernelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="kernel-")
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def build(self, replacement=None, **kwargs):
        """用假 Provider 顶掉真实解析——替换点在测试这一侧，生产代码不知道测试存在。

        replacement 是顶进去的假 Provider；其余 kwargs 原样传给 create_agent。
        """
        with patch("agentschema.factory.resolve_provider", return_value=replacement):
            return create_agent(self.root, **kwargs)

    def agent(self, responses: list) -> tuple:
        """建一个带假 io 的 agent，返回 (agent, rendered)。"""
        rendered: list[str] = []
        with patch("agentschema.capabilities.io_render", side_effect=lambda content: rendered.append(content)):
            agent = self.build(FakeProvider(responses))
        return agent, rendered

    # ---------- 会话循环 ----------

    def test_loop_reads_input_until_quit(self) -> None:
        rendered: list[str] = []
        inputs = iter(["你好", "quit"])
        with patch("agentschema.capabilities.io_render", side_effect=lambda content: rendered.append(content)), \
             patch("agentschema.capabilities.io_input", side_effect=lambda: next(inputs)):
            agent = self.build(FakeProvider(["你好！有什么可以帮你？"]))
            context = agent.loop()

        self.assertIn("助手已启动", rendered[0])
        self.assertIn("你好！有什么可以帮你？", rendered[1])
        self.assertEqual(rendered[-1], "再见。")
        # 会话历史：system（人设 + 工具说明）+ user + assistant
        roles = [item["role"] for item in context.messages]
        self.assertEqual(roles[:2], ["system", "system"])
        self.assertIn("role", context.messages[2], "user 消息在历史里")

    def test_loop_exits_on_eof(self) -> None:
        with patch("agentschema.capabilities.io_render"), \
             patch("agentschema.capabilities.io_input", side_effect=EOFError):
            agent = self.build(FakeProvider(["unused"]))
            context = agent.loop()
        self.assertEqual(context.style, "assistant")

    # ---------- 单轮：直接回复 ----------

    def test_turn_direct_reply(self) -> None:
        agent, _ = self.agent(["你好！"])
        reply = agent.turn("在吗")
        self.assertEqual(reply, "你好！")
        roles = [item["role"] for item in agent.context.messages]
        self.assertEqual(roles, ["system", "system", "user", "assistant"])

    def test_turn_injects_system_prompt_and_tools(self) -> None:
        provider = FakeProvider(["ok"])
        agent, _ = self.agent(["ok"])
        agent.provider = provider
        agent.turn("hi")
        first = provider.calls[0]
        # 工具说明随请求带上；系统提示词在会话消息里
        tool_names = [item["function"]["name"] for item in first["tools"]]
        self.assertIn("text__normalize", tool_names)
        self.assertIn("io__render", tool_names)

    # ---------- 单轮：工具调用 ----------

    def test_turn_executes_tool_and_feeds_result_back(self) -> None:
        agent, rendered = self.agent([
            [tool_call("text__normalize", {"text": "  a   b  "})],
            "结果是 a b",
        ])
        reply = agent.turn("帮我规整一下文本")
        self.assertEqual(reply, "结果是 a b")
        # 工具结果以 role:"tool" 回填，载荷成功
        tool_message = next(item for item in agent.context.messages if item["role"] == "tool")
        payload = json.loads(tool_message["content"])
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"], "a b")
        self.assertIn("结果是 a b", rendered)

    def test_turn_tool_failure_becomes_payload_not_crash(self) -> None:
        agent, _ = self.agent([
            [tool_call("text__normalize", {"text": 42})],  # 契约不符
            "工具坏了，抱歉",
        ])
        reply = agent.turn("试个坏的")
        self.assertEqual(reply, "工具坏了，抱歉")
        tool_message = next(item for item in agent.context.messages if item["role"] == "tool")
        payload = json.loads(tool_message["content"])
        self.assertFalse(payload["ok"])
        self.assertIn("expected string", payload["error"])

    def test_turn_unknown_tool_becomes_payload_not_crash(self) -> None:
        agent, _ = self.agent([
            [tool_call("nope__missing", {})],
            "没这个工具",
        ])
        reply = agent.turn("调个不存在的")
        self.assertEqual(reply, "没这个工具")
        tool_message = next(item for item in agent.context.messages if item["role"] == "tool")
        self.assertFalse(json.loads(tool_message["content"])["ok"])

    def test_turn_stops_at_tool_round_limit(self) -> None:
        agent, rendered = self.agent([[tool_call("text__word_count", {"text": "a"})]])
        with patch.object(agent.functional, "tools_schema", return_value=[]):
            reply = agent.turn("无限调工具")
        self.assertIn("工具调用上限", reply)
        self.assertIn("工具调用上限", rendered[-1])

    # ---------- 装配：模型必选 ----------

    def test_provider_unavailable_is_assembly_error(self) -> None:
        with self.assertRaises(KernelError) as caught:
            self.build(UnavailableProvider(), provider="deepseek-flash")
        self.assertIn("provider unavailable", str(caught.exception))

    def test_unknown_style_is_rejected(self) -> None:
        with self.assertRaises(KernelError):
            self.build(FakeProvider(["x"]), style="nope")

    def test_default_style_is_assistant(self) -> None:
        agent, _ = self.agent(["x"])
        self.assertEqual(agent.style, "assistant")
        self.assertEqual(
            {item["style"] for item in agent.agents()},
            {"assistant"},
        )

    # ---------- 系统提示词 ----------

    def test_system_prompt_comes_from_project_root(self) -> None:
        (self.root / "SYSTEM_PROMPT.md").write_text("你是测试助手。\n", encoding="utf-8")
        agent, _ = self.agent(["ok"])
        agent.turn("hi")
        self.assertEqual(agent.context.messages[0]["content"], "你是测试助手。")

    def test_default_system_prompt_without_file(self) -> None:
        agent, _ = self.agent(["ok"])
        agent.turn("hi")
        self.assertIn("agentschema", agent.context.messages[0]["content"])

    def test_blank_system_prompt_file_is_assembly_error(self) -> None:
        (self.root / "SYSTEM_PROMPT.md").write_text("  \n\n", encoding="utf-8")
        with self.assertRaises(KernelError) as caught:
            self.build(FakeProvider(["x"]))
        self.assertIn("empty system prompt", str(caught.exception))

    # ---------- 注册与扩展 ----------

    def test_register_extra_style_binds_same_registry(self) -> None:
        agent, _ = self.agent(["x"])
        agent.register(Style(
            "minimal",
            "A quieter persona",
            system_prompt="你是极简助手。",
            behavior=BehaviorStyle(max_tool_rounds=2),
        ))
        self.assertEqual(
            {item["style"] for item in agent.agents()},
            {"assistant", "minimal"},
        )
        # 换到新风格对话：系统提示词来自它自己
        agent.style = "minimal"
        provider = FakeProvider(["ok"])
        agent.provider = provider
        agent.turn("hi")
        system_messages = [
            item["content"] for item in provider.calls[0]["messages"]
            if item["role"] == "system"
        ]
        self.assertIn("极简助手", system_messages[0])

    # ---------- 能力表 ----------

    def test_tools_schema_maps_ids_and_types(self) -> None:
        agent, _ = self.agent(["x"])
        schema = {item["function"]["name"]: item["function"] for item in agent.functional.tools_schema()}
        self.assertIn("text__normalize", schema)
        self.assertEqual(schema["text__normalize"]["parameters"]["required"], ["text"])
        self.assertEqual(schema["text__normalize"]["parameters"]["properties"]["text"], {"type": "string"})
        self.assertEqual(schema["io__input"]["parameters"]["properties"], {})

    # ---------- providers.json ----------

    def test_provider_table_comes_from_json(self) -> None:
        (self.root / "providers.json").write_text(
            json.dumps([{"name": "local-model", "vendor": "deepseek", "model": "local-model"}]),
            encoding="utf-8",
        )
        from agentschema.providers import resolve_provider
        from agentschema.config import Config

        provider = resolve_provider("local-model", Config.load(self.root))
        self.assertIn("local-model", provider.description)

    def test_bad_provider_file_is_rejected(self) -> None:
        (self.root / "providers.json").write_text('{"name": "x"}', encoding="utf-8")
        from agentschema.config import Config
        from agentschema.providers import resolve_provider

        with self.assertRaises(KernelError):
            resolve_provider("deepseek-flash", Config.load(self.root))

    def test_unknown_name_lists_known_models(self) -> None:
        (self.root / "providers.json").write_text(
            json.dumps([{"name": "alpha", "vendor": "deepseek"}]), encoding="utf-8"
        )
        from agentschema.config import Config
        from agentschema.providers import resolve_provider

        with self.assertRaises(KernelError) as caught:
            resolve_provider("beta", Config.load(self.root))
        self.assertIn("alpha", str(caught.exception))


    # ---------- 切面：观测三要素 ----------

    def test_aspect_sees_style_model_capability(self) -> None:
        seen: list[dict] = []

        class Recorder:
            def before(self, invocation: dict) -> None:
                seen.append(dict(invocation))

            def after(self, invocation, output, error, ms) -> None:
                pass

        agent, _ = self.agent([[tool_call("text__normalize", {"text": " a b "})], "done"])
        agent.functional.add_aspect(Recorder())
        agent.turn("跑一下")
        invocation = seen[0]
        # 三要素：哪个 LLM、什么风格、调了什么能力
        self.assertEqual(invocation["model"], "fake-model")
        self.assertEqual(invocation["style"], "assistant")
        self.assertEqual(invocation["capability_id"], "text.normalize")
        self.assertEqual(invocation["effect"], "read")

    def test_logging_aspect_writes_jsonl(self) -> None:
        agent, _ = self.agent([[tool_call("text__normalize", {"text": " a b "})], "done"])
        agent.turn("跑一下")
        log_path = self.root / "workspace" / "capability_calls.jsonl"
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        record = next(item for item in records if item["capability"] == "text.normalize")
        self.assertEqual(record["style"], "assistant")
        self.assertTrue(record["ok"])
        self.assertGreaterEqual(record["ms"], 1)

    def test_aspect_can_deny_invocation(self) -> None:
        from agentschema import KernelError

        class DenyNormalize:
            """只拒目标能力的策略；全拒策略会连引擎的 io.render 一起拦。"""

            def before(self, invocation: dict) -> None:
                if invocation["capability_id"] == "text.normalize":
                    raise KernelError(f"permission denied: {invocation['capability_id']}")

            def after(self, invocation, output, error, ms) -> None:
                if invocation["capability_id"] == "text.normalize":
                    raise AssertionError("denied call must not reach after")

        agent, _ = self.agent([[tool_call("text__normalize", {"text": "x"})], "收到"])
        agent.functional.add_aspect(DenyNormalize())
        agent.turn("试一下")
        tool_message = next(item for item in agent.context.messages if item["role"] == "tool")
        payload = json.loads(tool_message["content"])
        self.assertFalse(payload["ok"])
        self.assertIn("permission denied", payload["error"])


if __name__ == "__main__":
    unittest.main()
