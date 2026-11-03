"""CLI for agent registration, style switching, cloning, and verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import create_agent
from .model import KernelError, TaskSpec


def _json_object(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise KernelError("expected a JSON object")
    return value


def _json_cases(raw: str) -> list[dict[str, Any]]:
    value = json.loads(raw)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise KernelError("cases must be a JSON list of argument objects")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agentschema")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--offline", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("agents", help="list registered execution profiles")
    sub.add_parser("capabilities", help="list the static functional model")

    run = sub.add_parser("run", help="run one capability through a selected style")
    run.add_argument("--profile", required=True)
    run.add_argument("--capability", required=True)
    run.add_argument("--args", default="{}")

    clone = sub.add_parser("clone", help="clone one pure Python function with DeepSeek")
    clone.add_argument("--path", required=True)
    clone.add_argument("--symbol", required=True)
    clone.add_argument("--cases", required=True)
    clone.add_argument("--capability-id")

    verify = sub.add_parser("verify", help="run differential and mutation-sensitivity checks")
    verify.add_argument("--path", required=True)
    verify.add_argument("--symbol", required=True)
    verify.add_argument("--expression", required=True)
    verify.add_argument("--cases", required=True)

    test = sub.add_parser("test", help="generate DeepSeek test inputs and verify a candidate")
    test.add_argument("--path", required=True)
    test.add_argument("--symbol", required=True)
    test.add_argument("--expression", required=True)
    test.add_argument("--count", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        agent = create_agent(args.root, offline=args.offline)
        if args.command == "agents":
            result: Any = agent.agents()
        elif args.command == "capabilities":
            result = agent.functional.descriptors()
        else:
            if args.command == "run":
                task = TaskSpec(
                    f"cli-{args.profile}",
                    args.capability,
                    _json_object(args.args),
                )
                profile = args.profile
            elif args.command == "clone":
                task = TaskSpec(
                    "cli-clone",
                    "clone",
                    {
                        "path": args.path,
                        "symbol": args.symbol,
                        "inputs": _json_cases(args.cases),
                        "capability_id": args.capability_id or f"legacy.{args.symbol}",
                    },
                )
                profile = "clone"
            elif args.command == "verify":
                task = TaskSpec(
                    "cli-verification",
                    "verification",
                    {
                        "path": args.path,
                        "symbol": args.symbol,
                        "expression": args.expression,
                        "inputs": _json_cases(args.cases),
                    },
                )
                profile = "verification"
            else:
                task = TaskSpec(
                    "cli-test",
                    "test",
                    {
                        "path": args.path,
                        "symbol": args.symbol,
                        "expression": args.expression,
                        "count": args.count,
                    },
                )
                profile = "test"
            report = agent.run(profile, task)
            result = report.to_dict()
            if report.status != "ok":
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 1
            if isinstance(report.output, dict) and report.output.get("accepted") is False:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (KernelError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
