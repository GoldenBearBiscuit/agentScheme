# agentschema

三个模型（上下文 / 功能 / 执行）驱动的会话式 agent 内核。`LoopEngine` 跑一个五环节会话循环（compose → req → parse → execute → render），默认风格是中文优先的个人助手：模型是内核标配，工具说明从能力表自动生成，输入和展示也是能力、可换实现。

零第三方依赖（纯 stdlib），Python ≥ 3.10。

## 三个模型

| 模型 | 契约 | 职责 |
|---|---|---|
| 上下文模型 | `ContextModel` | 保存会话消息历史、工具调用结果、证据链；记忆框架的选择收在域内部 |
| 功能模型 | `FunctionalModel` / `Capability` | 注册可寻址函数及其契约；工具说明（OpenAI tools schema）从这里生成；按「持久时长 × 影响范围」二维分层（演进方向） |
| 执行模型 | `Style`（`BehaviorStyle`） | 风格 = 系统提示词（人设）+ 行为参数（工具轮数上限等） |

三个模型之上是会话引擎：`MainAgent` 顶层只认 **style 和 provider，两者平级且都必选**——模型是内核标配，装配期没 key 直接报错，不留到会话里才炸。`LoopEngine.turn()` 跑单轮，`loop()` 是死循环，quit / exit / Ctrl+C 退出。

## 快速开始

```bash
cp .env.example .env                   # 填 DEEPSEEK_API_KEY
python -m agentschema                  # 进入助手会话（默认入口）
python -m agentschema agents           # 列出已注册风格（诊断命令）
python -m agentschema capabilities     # 列出能力表（诊断命令）
```

会话示例：

```
[assistant] 助手已启动，quit/exit 退出
> 帮我看看这句话几个词：hello world
「hello world」按空格分隔算 2 个词（hello、world）。
> quit
再见。
```

可用模型列在项目根的 `providers.json`（对象列表，可直接手改），`--provider <name>` 按名字选，默认 `deepseek-flash`：

```bash
python -m agentschema --provider deepseek-v4-pro
```

## 系统提示词

默认人设写在 `agentschema/prompt.py` 的 `DEFAULT_SYSTEM_PROMPT`。要换人设不必改代码：在项目根放一份 `SYSTEM_PROMPT.md`（也就是 `--root` 指向的目录），`create_agent()` 装配前会读它并整份替换默认值；没有这个文件就用内置默认。文件存在但读出来是空白会当场报错（`empty system prompt`），不会悄悄退回默认。

## 内置能力

| 能力 | 说明 |
|---|---|
| `text.normalize` / `text.word_count` | 文本处理（最小工具集示例） |
| `io.input` / `io.render` | 终端输入 / 输出——展示是能力，可换 TUI、Web 或测试假件 |

## 目录

```
agentschema/
  context_model.py     上下文模型（ContextModel：messages / steps / evidence）+ KernelError
  functional_model.py  功能模型（Capability / FunctionalModel / tools_schema）
  style.py             执行模型（BehaviorStyle / Style）
  loop.py              会话引擎（LoopEngine: turn / loop；MainAgent）
  factory.py           装配厂（create_agent：style + provider，都必选）
  capabilities.py      内置能力表与 assistant 风格
  prompt.py            系统提示词（默认人设 + 项目根 SYSTEM_PROMPT.md 覆盖）
  providers.py         DeepSeek Provider（内核标配模型，不可用即装配报错）
  cli.py               命令行（默认进会话；agents / capabilities 诊断）
  config.py            .env + 环境变量 → Config
  llm.py               stdlib urllib 实现的 OpenAI 兼容客户端（chat_tools 协议）
  log.py               日志门面（默认 stderr）
tests/             会话循环、装配、能力表用例，stdlib unittest
scripts/           run_tests.py（测试入口：sys.path 引导 + unittest 转发）
providers.json     模型系列表（对象列表，可直接手改）
```

运行产物写在 `workspace/`（`llm_calls.jsonl` 存 LLM 调用记录），已在 `.gitignore` 中忽略。

## 测试

```bash
python scripts/run_tests.py -v                    # 任意工作目录下可用，推荐
python -m unittest discover -s tests -v           # 等价命令，需在项目根执行
python scripts/run_tests.py tests.test_agentschema -v   # 只跑单个模块
```

测试同样走会话路径：FakeProvider 模拟 LLM 的回复或工具调用，io 能力在测试侧替换——没有"只跑一次"的特殊执行形态。

## 边界（别当成 bug）

- **模型必选**：没有 `DEEPSEEK_API_KEY` 时装配直接报错（`provider unavailable`），不会降级成"假装能聊"。
- 工具执行失败会转成结构化错误载荷回给模型，由它向用户如实转述，不炸会话。
- 单轮工具调用次数有上限（`BehaviorStyle.max_tool_rounds`，默认 8），超限如实提示。
- clone / verification 等质量方法论能力已移除，后续以工具形态回归。
