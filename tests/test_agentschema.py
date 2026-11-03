from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentschema import (
    CapabilityRef,
    ExecutionPlan,
    ExecutionStep,
    TaskSpec,
    create_agent,
)


class FakeProvider:
    available = True

    def __init__(self, expression: str | list[str]):
        self.expressions = [expression] if isinstance(expression, str) else expression
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> dict:
        self.calls.append((system, user))
        expression = self.expressions[min(len(self.calls) - 1, len(self.expressions) - 1)]
        return {"expression": expression, "summary": "offline fake"}


class CaseProvider:
    available = True

    def complete_json(self, system: str, user: str) -> dict:
        return {"inputs": [{"value": 0}, {"value": 3}, {"value": 9}]}


class KernelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="kernel-")
        self.root = Path(self._tmp.name)
        package = self.root / self.root.name.replace("-", "_")
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "helpers.py").write_text("LIMIT = 6\n", encoding="utf-8")
        self.source = package / "legacy.py"
        self.source.write_text(
            "from .helpers import LIMIT\n\n"
            "def clamp(value):\n"
            "    return max(1, min(int(value), LIMIT))\n",
            encoding="utf-8",
        )
        self.cases = [
            {"value": 0},
            {"value": 3},
            {"value": 9},
            {"value": "4"},
            {"value": "bad"},
        ]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def agent(self, expression: str | list[str] = "max(1, min(int(value), 6))"):
        self.provider = FakeProvider(expression)
        return create_agent(self.root, providers={"deepseek": self.provider})

    def clone_task(self) -> TaskSpec:
        return TaskSpec(
            "clone-test",
            "clone",
            {
                "path": str(self.source),
                "symbol": "clamp",
                "inputs": self.cases,
                "capability_id": "legacy.clamp",
            },
        )

    def verification_task(self, expression: str) -> TaskSpec:
        return TaskSpec(
            "verify-test",
            "verification",
            {
                "path": str(self.source),
                "symbol": "clamp",
                "expression": expression,
                "inputs": self.cases,
            },
        )

    def test_styles_preserve_functional_output(self) -> None:
        agent = self.agent()
        for capability, args, expected in (
            ("text.normalize", {"text": "  same   function  "}, "same function"),
            ("text.word_count", {"text": "same function"}, 2),
        ):
            with self.subTest(capability=capability):
                task = TaskSpec("switch", capability, args)
                kiss = agent.run("kiss", task)
                pi = agent.run("pi", task)
                self.assertEqual(kiss.status, "ok")
                self.assertEqual(pi.status, "ok")
                self.assertEqual(kiss.output, expected)
                self.assertEqual(pi.output, expected)
                self.assertEqual([item["step"] for item in kiss.trace], ["act"])
                self.assertEqual(
                    [item["step"] for item in pi.trace],
                    ["prepare", "act", "observe"],
                )
                self.assertEqual(kiss.trace[0]["capability_id"], capability)
                self.assertEqual(pi.trace[1]["capability_id"], capability)

    def test_external_profile_uses_same_registry(self) -> None:
        agent = self.agent()
        agent.register(ExecutionPlan(
            "external",
            (ExecutionStep(
                "act",
                CapabilityRef("$task.capability_id"),
                args_from="$task.args",
            ),),
            "act",
        ))
        report = agent.run(
            "external",
            TaskSpec("external-task", "text.normalize", {"text": "  external   style "}),
        )
        self.assertEqual(report.output, "external style")
        self.assertIn("external", {item["profile"] for item in agent.agents()})

    def test_unavailable_provider_is_explicitly_unsupported(self) -> None:
        agent = create_agent(self.root, providers={})
        report = agent.run("clone", self.clone_task())
        self.assertEqual(report.status, "unsupported")
        self.assertIn("deepseek", report.error)

    def test_clone_uses_reference_outcomes_not_llm_expectations(self) -> None:
        agent = self.agent()
        report = agent.run("clone", self.clone_task())
        self.assertEqual(report.status, "ok", report.error)
        self.assertTrue(report.output["accepted"], report.output)
        self.assertTrue(report.output["registered"])
        self.assertEqual(report.output["harness"], "sqlite-commit-v1")
        self.assertEqual(report.output["provenance"]["symbol"], "clamp")
        self.assertEqual(
            [item["expected"] for item in report.output["cases"][:4]],
            [
                {"ok": True, "value": 1},
                {"ok": True, "value": 3},
                {"ok": True, "value": 6},
                {"ok": True, "value": 4},
            ],
        )
        self.assertEqual(report.output["cases"][4]["expected"]["error"], "ValueError")
        self.assertEqual(report.output["cases"][4]["actual"]["error"], "ValueError")
        self.assertIn("def clamp", self.provider.calls[0][1])
        self.assertEqual(len(self.provider.calls), 1)
        absorbed = agent.run(
            "kiss", TaskSpec("use-absorbed", "legacy.clamp", {"value": 9})
        )
        self.assertEqual(absorbed.status, "ok")
        self.assertEqual(absorbed.output, 6)
        reloaded = create_agent(self.root, providers={"deepseek": self.provider})
        restored = reloaded.run(
            "kiss", TaskSpec("restore-absorbed", "legacy.clamp", {"value": 0})
        )
        self.assertEqual(restored.output, 1)

    def test_clone_rejects_behavioral_mismatch(self) -> None:
        report = self.agent("min(int(value), 6)").run("clone", self.clone_task())
        self.assertEqual(report.status, "ok")
        self.assertFalse(report.output["accepted"])
        self.assertFalse(report.output["registered"])
        self.assertFalse(report.output["cases"][0]["match"])
        self.assertEqual(len(self.provider.calls), 2)

    def test_clone_repairs_once_from_oracle_feedback(self) -> None:
        report = self.agent([
            "min(int(value), 6)",
            "max(1, min(int(value), 6))",
        ]).run("clone", self.clone_task())
        self.assertTrue(report.output["accepted"], report.output)
        self.assertTrue(report.output["registered"])
        self.assertEqual(len(self.provider.calls), 2)
        self.assertIn("verification", self.provider.calls[1][1])

    def test_candidate_cannot_execute_python_or_touch_files(self) -> None:
        marker = self.root / "unsafe"
        expression = f'__import__("pathlib").Path({str(marker)!r}).touch()'
        report = self.agent(expression).run("clone", self.clone_task())
        self.assertEqual(report.status, "ok")
        self.assertFalse(report.output["accepted"])
        self.assertFalse(marker.exists())
        self.assertEqual(report.output["cases"][0]["actual"]["error"], "KernelError")

    def test_verification_strategy_checks_harness_sensitivity(self) -> None:
        agent = self.agent()
        accepted = agent.run(
            "verification",
            self.verification_task("max(1, min(int(value), 6))"),
        )
        self.assertTrue(accepted.output["accepted"], accepted.output)
        self.assertEqual(accepted.output["harness"], "sqlite-commit-v1")
        self.assertEqual(accepted.output["mutation"]["checked"], 6)
        self.assertEqual(accepted.output["mutation"]["survivors"], [])

        rejected = agent.run("verification", self.verification_task("None"))
        self.assertFalse(rejected.output["accepted"])

    def test_style_generates_inputs_but_not_expected_outcomes(self) -> None:
        agent = create_agent(self.root, providers={"deepseek": CaseProvider()})
        report = agent.run("test", TaskSpec(
            "generated-tests",
            "test",
            {
                "path": str(self.source),
                "symbol": "clamp",
                "expression": "max(1, min(int(value), 6))",
                "count": 3,
            },
        ))
        self.assertTrue(report.output["accepted"], report.output)
        self.assertEqual(len(report.output["cases"]), 3)
        self.assertEqual(report.output["cases"][0]["expected"]["value"], 1)

    def test_contract_failure_does_not_fake_success(self) -> None:
        report = self.agent().run(
            "kiss",
            TaskSpec("bad-contract", "text.normalize", {"text": 42}),
        )
        self.assertEqual(report.status, "failed")
        self.assertIn("expected string", report.error)


if __name__ == "__main__":
    unittest.main()
