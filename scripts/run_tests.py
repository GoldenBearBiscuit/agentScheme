"""测试引导：绕过本机 os.mkdir(mode=0o700) 的 ACL 限制后跑 unittest。

背景：本机用 `os.mkdir(path, 0o700)` 建的目录连当前用户都不可访问，
导致 tempfile.mkdtemp / TemporaryDirectory 系测试在 setUp 就 PermissionError。
这是环境 ACL 问题，不是项目代码问题。

用法：
    python scripts/run_tests.py              # discover tests/
    python scripts/run_tests.py tests.test_agentschema -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# 直接 `python scripts/run_tests.py` 时，sys.path[0] 是 scripts/；把项目根补回来。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MKDIR = os.mkdir


def _mkdir(path, mode=0o777, *args, **kwargs):  # noqa: ANN001 - 兼容 os.mkdir 签名
    """把 mode 固定成 0o777，绕开本机对 0o700 目录的 ACL 限制。"""
    return _MKDIR(path, 0o777, *args, **kwargs)


def main(argv: list[str]) -> int:
    """装好 ACL 兼容层后跑测试，返回 0/1 退出码。"""
    if sys.platform == "win32":
        os.mkdir = _mkdir  # type: ignore[assignment]

    verbosity = 2 if "-v" in argv else 1
    names = [arg for arg in argv if not arg.startswith("-")]
    loader = unittest.TestLoader()
    suite = (
        loader.loadTestsFromNames(names)
        if names
        else loader.discover("tests", top_level_dir=".")
    )
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
