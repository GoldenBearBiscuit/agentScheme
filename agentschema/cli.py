"""CLI for agent registration, style listing, and the interactive assistant session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import create_agent
from .context_model import KernelError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agentschema")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--provider", default=None, help="model series name from providers.json")
    parser.add_argument("--functional", default=None,
                        help="functional model seed name (default: builtin)")
    parser.add_argument("--engine", default=None,
                        help="execution engine name (default: loop)")
    sub = parser.add_subparsers(dest="command", required=False)
    sub.add_parser("agents", help="list registered styles")
    sub.add_parser("capabilities", help="list the static functional model")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        agent = create_agent(
            args.root,
            provider=args.provider,
            functional=args.functional,
            engine=args.engine,
        )
        if args.command == "agents":
            print(json.dumps(agent.agents(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "capabilities":
            print(json.dumps(agent.functional.descriptors(), ensure_ascii=False, indent=2))
            return 0

        # 默认：进交互会话，一直跑到 quit / exit / Ctrl+C
        agent.loop()
        return 0
    except KeyboardInterrupt:
        print("\nbye.")
        return 0
    except (KernelError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
