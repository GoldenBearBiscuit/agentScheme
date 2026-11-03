# agentschema

三个模型（上下文 / 功能 / 执行）驱动的可验证 agent 内核。同一套静态能力库，可以用不同执行风格（kiss / pi）跑，也可以用 LLM 去"吸收"一个已有纯函数（clone / test / verification），但**期望值永远来自参考实现的真实运行结果，不来自 LLM**。

零第三方依赖（纯 stdlib），Python ≥ 3.10。

## 三个模型

| 模型 | 契约 | 职责 |
|---|---|---|
| 上下文模型 | `TaskSpec` / `ContextModel` / `RunReport` | 保存任务、阶段结果、证据、产物和运行报告 |
| 功能模型 | `FunctionalModel` / `Capability` / `CapabilityRef` | 注册可寻址函数及其输入、输出、来源和 Oracle；不绑定执行风格 |
| 执行模型 | `PlanModel` / `ExecutionStep` / `LoopEngine` / `MainAgent` | 注册和切换阶段计划、Provider 与结果投影 |

三个模型共享一个 `LoopEngine`。Tracing 横切三者；`pi` 与 `kiss` 对同一任务可以产生不同 Trace，但最终功能输出必须一致。

## 五个 Profile

| Profile | 阶段 | 边界 |
|---|---|---|
| `kiss` | `act` | 直接调用任务指定的共享能力 |
| `pi` | `prepare → act → observe` | 多步准备和观察，结果仍取 `act` |
| `clone` | `load → propose → verify → repair → verify_repair → register` | LLM 只生成受限表达式；参考实现实测结果才是期望值；首次拒绝后最多修正一次 |
| `test` | `load → cases → candidate → verify` | LLM 只生成输入用例，不生成期望值 |
| `verification` | `load → candidate → verify` | 差分验证 + 6 个负向变异，检查验证器能否拒绝错误实现 |

## 快速开始

```bash
python -m agentschema agents          # 列出 5 个执行 Profile
python -m agentschema capabilities    # 列出静态能力表
python -m agentschema run --profile pi --capability text.normalize --args '{"text":" a   b "}'

# 不需要 key 的验证路径：手工候选 + 差分/变异检查
python -m agentschema verify --path <你的纯函数文件> --symbol <函数名> \
  --expression 'max(1, min(int(value), 6))' --cases '[{"value":0},{"value":9}]'
```

`clone` / `test` 需要 `DEEPSEEK_API_KEY`；没有 key 时返回 `unsupported`，不会伪造离线克隆结果。加 `--offline` 可强制禁止真实调用。

## 目录

```
agentschema/
  model.py      三个模型的契约 + LoopEngine + MainAgent
  capabilities.py  静态能力表、5 个内置 Plan、表达式持久化（ExpressionStore）
  providers.py     DeepSeek Provider（执行模型的可插拔依赖）
  cli.py           命令行（agents / capabilities / run / clone / verify / test）
  compare.py       行为等价比较（差分验证的判定函数）
  config.py        .env + 环境变量 → Config
  llm.py           stdlib urllib 实现的 OpenAI 兼容客户端
  log.py           日志门面（默认 stderr）
tests/             10 个用例，stdlib unittest
scripts/           run_tests.py（本机 ACL 兼容引导）
```

运行产物写在 `workspace/`（`expressions.json` 存已固化的克隆能力，`llm_calls.jsonl` 存 LLM 调用记录），已在 `.gitignore` 中忽略。

## 测试

```bash
python scripts/run_tests.py -v        # 或 python -m unittest discover -s tests -v
```

## 边界（别当成 bug）

- **只支持可信旧模块中的同步纯 Python 函数**：加载参考实现会 import 旧模块；候选表达式不会作为 Python 模块执行，而是经过 AST 白名单后求值。
- `accepted` 只表示**给定用例上**行为一致，不表示已证明完整语义等价。
- 克隆成功的表达式写入 `workspace/expressions.json`，之后以共享能力形式恢复；被拒绝的候选不登记，也没有自动晋升。
- Provider 不可用时返回 `unsupported`，不降级成"假装成功"。

## 来源

代码从 `mcp-assimilator`（`D:\Project\mcp-assimilator`）的 `assimilator/kernel/` 切片迁出（HEAD `b6528d6`，2026-11-03）。为了让新仓库可独立运行，一并本地化了三个被依赖的最小部分：

| 原位置 | 现位置 | 处理 |
|---|---|---|
| `assimilator/harness.py::compare` | `agentschema/compare.py` | 只抽比较函数，实现与文档保留 |
| `assimilator/config.py` | `agentschema/config.py` | 砍掉 evidence/skills/registry/memory 等流水线路径，只留 LLM 与 workspace 配置 |
| `assimilator/llm.py`、`assimilator/log.py` | 同名文件 | 整文件迁入；日志环境变量前缀改为 `AGENTSCHEMA_` |

原仓库未做任何修改。
