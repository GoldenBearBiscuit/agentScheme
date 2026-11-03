"""行为等价比较：差分验证的唯一判定函数。

从 mcp-assimilator 的 harness.compare 抽出（原文件还包含技能加载、评估等
流水线逻辑，本仓库只需要比较语义），实现与文档逐字保留。
"""

from __future__ import annotations

import math
from typing import Any


def compare(expected: Any, actual: Any, tol: float = 1e-9) -> bool:
    """递归比较两个返回值是否"行为等价"。

    入参：
        expected: 原实现跑出来的真值。
        actual: 我的实现跑出来的值。
        tol: 浮点相对容差（相对值，见下面）。

    处理：
        - bool 优先单独处理：Python 里 True == 1，不特判会把 true 和 1 判成相等
          （注意只把 bool 当 bool：True 与 1 必须判不等，否则真值语义被悄悄放宽）
        - 数字用**相对**误差 `|a-b| <= tol * max(1, |a|)`，所以 1e-12 和 0 算相等，
          而 1e6 量级的两个数允许更大的绝对误差
        - NaN 只和 NaN 相等（正常 == 会返回 False）
        - 无穷同号判相等、异号判不等（inf - inf = nan，不能落进容差比较）
        - dict 要求键集合完全相同，再逐键递归（多一个键就算不一致）
        - list/tuple 要求长度相同，逐元素递归
        - 其它类型退化成 ==

    出参：True 表示行为一致。
    """
    if isinstance(expected, bool) or isinstance(actual, bool):
        # 只有两边都是 bool 才可能相等：True vs 1、False vs 0 一律判不等
        return isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, float) and math.isnan(expected):
            return isinstance(actual, float) and math.isnan(actual)
        expected_value, actual_value = float(expected), float(actual)
        if expected_value == actual_value:
            # 同号 Inf 走这里（inf == inf 为 True）；NaN 已在上方返回，不会误命中
            return True
        # 异号 Inf 相减是 Inf，会骗过下面的相对容差（Inf <= Inf）；先挡掉非有限值
        if not (math.isfinite(expected_value) and math.isfinite(actual_value)):
            return False
        return abs(expected_value - actual_value) <= tol * max(1.0, abs(expected_value))
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False
        return all(compare(expected[key], actual[key], tol) for key in expected)
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False
        return all(compare(a, b, tol) for a, b in zip(expected, actual))
    return expected == actual
