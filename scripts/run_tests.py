"""测试入口
    执行命令后边的测试用例

用法：
    python scripts/run_tests.py                          # 跑 tests/ 下全部用例
    python scripts/run_tests.py -v                       # 详细输出
    python scripts/run_tests.py tests.test_agentschema -v # 只跑某个模块/用例
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv: list[str]) -> int:
    """跑测试并返回 0/1 退出码，方便 CI 判断。"""
    verbosity = 2 if "-v" in argv else 1
    names = [arg for arg in argv if not arg.startswith("-")]

    loader = unittest.TestLoader()
    suite = (
        loader.loadTestsFromNames(names)
        if names
        else loader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    )
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
