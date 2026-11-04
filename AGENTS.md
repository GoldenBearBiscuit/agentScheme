# AGENTS.md

agentschema：三个模型（上下文 / 功能 / 执行）驱动的会话式 agent 内核。零第三方依赖，Python ≥ 3.10。

## 常用命令

- 跑测试：`python scripts/run_tests.py -v`（任意工作目录可用；等价命令 `python -m unittest discover -s tests -v` 需在项目根）
- 进助手会话：`python -m agentschema`（默认入口；quit / exit / Ctrl+C 退出）
- 列已注册风格：`python -m agentschema agents`
- 换模型：`python -m agentschema --provider deepseek-v4-pro`（可选值见 `providers.json`；模型是内核标配，没配 key 装配期就报错）

## 编码风格

本仓库 Python 代码统一按下面的规则写。它约束的是**可读性从哪来**：来自顺序和命名，不来自抽象层次。

宁可要一个 30 行但五步清晰的函数，也不要五个各 6 行、得来回跳的小函数。

### 1. 一物一名

同一个概念在整条链路上只用一个名字：

```python
create_agent(style=...)            # 入参
    style = style or DEFAULT_STYLE # 同名收敛默认值，不新建变量
    selected = builtin[style]      # 用它查表
    MainAgent(selected, providers=providers)  # 交给下游；下游 self.style = selected.style
```

允许**覆盖同名入参**来补默认值：

```python
style = style or DEFAULT_STYLE     # ✅ 概念没变，名字不变
picked = style or DEFAULT_STYLE    # ❌ 概念没变，名字变了
```

不接受同义词。改名前先问：这是新概念，还是同一个东西换了个叫法？后者不改。

### 2. 函数体分步，步骤可数

每步一个编号注释，一个动作一个目的：

```python
# 1. 校验风格名：内置表里有才放行，报错时把已知名字一起给出（功能模型随风格返回）
# 2. 加载配置：整个装配只读一次 .env，Provider 复用这份
# 3. 解析 Provider：名字写错就在装配时炸，不留到 run 才暴露
# 4. 装配：顶层只有 style（含功能模型）和 providers
```

四步，扫一眼就知道这个函数有几件事。**不为了拆而拆**——拆了就得去别处找顺序。

### 3. 每步内部：校验 → 收集 → 处理

顺序固定，因为顺序本身在表达约束：坏输入不能产生副作用。

```python
def create_agent(rootpath=None, *, style=None, provider=None):
    """按 style 装配一个可运行的 agent。"""
    # 1. 校验：名字不在内置表里就当场出去
    style = style or DEFAULT_STYLE
    builtin = {item.style: item for item in builtin_styles()}
    if style not in builtin:
        known = ", ".join(sorted(builtin))
        raise KernelError(f"unknown style: {style} (known: {known})")
    selected = builtin[style]

    # 2. 收集：把这一步需要的东西取齐
    config = Config.load(rootpath)

    # 3. 处理：加工、装配、返回
    return MainAgent(selected, providers=providers)
```

两条推论：

- **校验前置**：写错的名字在装配期就炸，不留到 `run()` 才暴露。
- **不猜也不兜底**：`providers.json` 格式坏了直接报错，不悄悄退回默认值。

### 4. 契约注释在函数头，函数内只留一句

```python
# 发一次 POST /chat/completions，带 3 次指数退避重试。
#
# 入参：payload —— 完整请求体（model / messages / temperature…）。
# 出参：解析后的响应 JSON。
# 异常：LLMError —— 重试耗尽或不可重试的 HTTP 错误。
def _post(self, payload):
    """POST /chat/completions，按 2s / 4s / 6s 退避重试。"""
```

- 入参 / 出参 / 异常 → `#` 块，紧贴 `def`，中间不留空行
- 函数内 docstring 只留一句
- 步骤说明留在代码里，挨着它描述的那几行
- 单行就能说清的函数不用加头注，保持一句 docstring 即可

### 5. 返回形状固定

- 失败统一走异常，不用 `False` / `None` 混着表达"没成功"
- 空值用空容器：`{}` 而不是 `None`
- 唯一例外是**结果状态**（如 `RunReport.status == "unsupported"`），那是数据不是错误

### 模板

```python
# <一句话职责>。
#
# 入参：
#     a: ...
#     b: ...
# 出参：<类型，什么状态>。
# 异常：KernelError —— <什么情况>。
def do_something(a, *, b=None):
    """<一句话>。"""
    # 1. 校验：坏输入先出去
    ...
    # 2. 收集：取齐依赖
    ...
    # 3. 处理：加工并返回
    return ...
```

### 这套风格排斥什么

| 反例 | 原因 |
|---|---|
| `picked` / `default_style` 这类同义词 | 同一个概念两个名字，读的人要猜 |
| `str \| Any \| dict` 三态参数 | 为将来预留的灵活度，当下没人用 |
| 生产代码里给测试开的后门（模块级覆盖字典） | 测试替换点应该在测试侧（`patch` 函数） |
| `plans = [x]` 然后 `for plan in plans` | 单元素容器 + 循环，纯噪音 |
| 契约写在函数体内、步骤写在函数头外 | 位置错位，改代码时注释跟不上 |

## 自查清单

动完代码过一遍：

- [ ] 新出现的名字，是概念变了还是同一概念换叫法？
- [ ] 每个函数能数出几步？有没有哪步混了两件事？
- [ ] 校验是不是都在产生副作用之前？
- [ ] 多行契约注释在 `def` 上方，函数内只剩一句？
- [ ] 有没有留下给测试用的生产代码分支？

## 边界

- 只支持可信旧模块中的**同步纯 Python 函数**；候选表达式经 AST 白名单后求值，不作为模块执行。
- `accepted` 只表示**给定用例上**行为一致，不等于完整语义等价。
- Provider 不可用时返回 `unsupported`，不降级成"假装成功"。
