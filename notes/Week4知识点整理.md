# Week 4 知识点整理（Day 18 ~ Day 23）

> 🏷️ 所属模块：🦜 **LangChain**（langchain_core.tools + langchain.agents.create_agent；Agent 底层运行于 🕸️ LangGraph）
> 📅 整理日期：2026-10-05 | 状态：全部完成 ✅
>
> 主题：Tool 工具 + Agent 智能体（ReAct 循环）

---

## Day 18：自定义 Tool ✅

### Tool 三要素（面试必背 ⭐）

| 要素 | 来源 | 作用 |
|---|---|---|
| name | 函数名 | 工具的唯一标识 |
| description | docstring（函数注释） | **给模型看的"说明书"**，模型靠它决定什么时候调用这个工具 |
| 参数 schema | 类型注解自动生成 | 告诉模型该传什么参数 |

```python
@tool
def get_weather(city: str) -> str:
    """查询指定中国城市今天的天气。输入城市名，如 '北京'。"""  # ← 这行注释至关重要！
    ...
```

> **面试标准答案 —— description 为什么重要？**
> 模型选择工具时只能看到 name + description + schema，看不到函数实现。
> description 写得模糊 → 模型该调的时候不调、乱传参数。写好说明书 = Agent 质量的一半。

---

## Day 19-20：create_agent 创建智能体 ✅

### 新版标准入口（1.0）

```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[calculator, get_weather],
    system_prompt="你是一个助手，需要用工具时主动调用工具",
)
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```

> ⚠️ 旧教程的 `AgentExecutor`、`initialize_agent`、`create_react_agent` 全部已废弃！

### ReAct 循环（面试必考 ⭐，本次实验亲眼所见）

```
用户问题："北京天气怎么样？帮我算 123*456"
   ↓
🤔 思考：需要查天气 + 算数 → 决定调用 get_weather、calculator
   ↓
🔧 执行：工具返回 "晴，25°C" 和 "56088"
   ↓
🤔 再思考：信息够了 → 生成最终回答
   ↓
💬 "北京今天晴，25°C。123 × 456 = 56088。"
```

> **面试标准答案 —— ReAct 模式？**
> Reasoning（推理）+ Acting（行动）的循环：思考 → 选工具 → 执行 → 观察结果 → 再思考，
> 直到信息足够给出最终答案。Agent 与普通 Chain 的本质区别：**流程不固定，由模型自主决策**。

### Agent vs Chain（面试题）

| | Chain（LCEL） | Agent |
|---|---|---|
| 流程 | 程序员写死 | 模型自主决定 |
| 适合 | 流程确定的场景（RAG 问答） | 步骤不确定的场景（助手类） |
| 可控性 | 高 | 较低（需防死循环） |

---

## Day 21：生活助手 Agent（4 工具）✅

工具：calculator / get_weather / save_note / read_notes

### 验证结果（复合指令）

输入："帮我记一下：明天下午3点开产品会。另外上海天气怎么样？"
→ Agent 自主调用 save_note + get_weather，两件事都办了 ✅
→ 再问"看看记事本里有什么"，Agent 调 read_notes 读出了刚才的记录 ✅

### Day 22-23 面试题（已掌握）

- **Agent 死循环怎么办？** → 设置最大迭代次数（`recursion_limit`）、工具加超时、中间件监控
- **Tool Calling 是什么？** → 模型输出结构化的"我要调X工具、参数是Y"，由框架真正执行函数，结果回填给模型

---

## 📁 Week 4 代码文件清单

| 文件 | 内容 |
|---|---|
| `day18_tools.py` | @tool 自定义工具 + 三要素 |
| `day19_agent.py` | create_agent + ReAct 循环观察 |
| `day21_assistant.py` | 生活助手（4 工具，含文件读写） |
| `notes.txt` | 助手写入的笔记文件（运行产物） |

---

## ⏭️ 下一站：Week 5 —— LangGraph 高级编排（State/Node/Edge）
