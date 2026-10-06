# Week 5 知识点整理（Day 24 ~ Day 30）

> 📅 整理日期：2026-10-05 | 状态：全部完成 ✅
>
> 主题：LangGraph 高级编排 —— 把流程画成"图"

---

## Day 24-25：核心三要素（只记这三个）✅

| 要素 | 是什么 | 代码 |
|---|---|---|
| **State** | 贯穿全流程的共享数据 | `class State(TypedDict): topic: str` |
| **Node** | 函数：读 State → 干活 → 返回要更新的字段 | `def write_node(state): return {"draft": ...}` |
| **Edge** | 节点连线；条件边 = 按 State 决定走向 | `add_conditional_edges(...)` |

### 组装套路（必背）

```python
builder = StateGraph(State)
builder.add_node("write", write_node)
builder.add_edge(START, "write")
builder.add_edge("write", "review")
builder.add_conditional_edges("review", route_fn, {"write": "write", "end": END})
graph = builder.compile()
result = graph.invoke({...})
```

### 实验验证

写作节点产出初稿 → 审核节点打了 8 分 → 条件边判断"通过"→ 结束。
（如果低于 8 分会自动回到写作节点**循环重写**——这就是条件边的威力）

> **面试标准答案 —— Chain / Agent / LangGraph 的区别？**
> - **Chain（LCEL）**：流程程序员写死，最可控，适合固定流程（如 RAG）
> - **Agent**：流程模型全权决定，最灵活但不可控
> - **LangGraph**：折中方案——你画流程图（框架可控），模型在节点里干活（局部智能），
>   还支持持久化、人工介入、多 Agent，是生产环境复杂系统的主流选择

---

## Day 26：Checkpointer 持久化 ✅

```python
graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "user_001"}}  # 会话ID，不同id完全隔离
graph.invoke(state, config=config)
```

### 三大作用（面试加分）

1. **断点续跑**：流程中断后从存档恢复
2. **时间旅行**：`get_state_history(config)` 能看到每一步的 State 快照（实验里看到了 4 个历史快照）
3. **多会话隔离**：user_001 和 user_002 的状态互不干扰（实验验证：count 分别是 20 和 1010）

生产环境用 `SqliteSaver` / `PostgresSaver` 替代内存版。

---

## Day 27：Human-in-the-loop 人工介入 ✅

```python
from langgraph.types import interrupt, Command

def review_node(state):
    decision = interrupt({"msg": "请审核", "draft": state["draft"]})  # ← 流程在此暂停
    return {"approved": decision["approved"]}

# 人工审批后恢复：
graph.invoke(Command(resume={"approved": True}), config=config)
```

### 实验验证

AI 写好宣传语 → 流程暂停 → 模拟人工点"通过" → 流程继续 → 发布 ✅

> **面试点**：人工介入必须配 checkpointer（暂停时状态要存盘，恢复时才知道跑到哪了）。
> 典型场景：发文审核、转账确认、敏感操作审批。

---

## Day 28-30：本周项目 —— 自动写作 Agent ✅

### 流程图

```
START → 生成大纲 → 写正文 → 自我审核 →【条件边】
                                    ├─ 分数≥8 或 已写3稿 → 人工终审 → END
                                    └─ 不达标 → 回到写正文（循环重写）
```

### 关键设计

- **attempts 计数器防死循环**：最多重写 3 次，超过强制送人工（这是生产级思维！）
- **interrupt 人工终审**：AI 达标后仍要人点头才发布
- 实验结果：1 稿就拿了 9 分 → 人工通过 → 输出终稿 ✅

---

## 📁 Week 5 代码文件清单

| 文件 | 内容 |
|---|---|
| `day24_langgraph_basic.py` | State/Node/条件边 + 写作审核循环 |
| `day26_checkpointer.py` | 持久化 + 时间旅行 + 多会话隔离 |
| `day27_hitl.py` | interrupt 人工介入 |
| `day28_writing_agent.py` | 自动写作 Agent（综合实战） |

---

## ⏭️ 下一站：Week 6 —— LangSmith 调试 + 求职级实战项目
