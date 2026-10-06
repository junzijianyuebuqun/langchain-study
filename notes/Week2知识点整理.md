# Week 2 知识点整理（Day 6 ~ Day 10）

> 🏷️ 所属模块：🦜 **LangChain**（langchain_core: LCEL runnables / RunnableParallel / RunnablePassthrough / messages）
> 📅 整理日期：2026-10-05 | 状态：全部完成 ✅
>
> 主题：LCEL 管道深入 + 对话记忆 + 综合实战（面试模拟器）

---

## Day 6：LCEL 深入（核心范式）✅

### LCEL 数据流向（面试会让你画！）

```
dict → prompt(填充变量) → model(返回AIMessage) → parser(变成str)
```

> **面试标准答案 —— LCEL 是什么？**
> LangChain Expression Language，用 `|` 管道符声明式地组合组件。
> 1.0 版本用它取代了旧的 LLMChain/SequentialChain，优势：
> 写法统一、自动支持流式/异步/批量、可任意嵌套组合。

### 两个重要工具

| 工具 | 作用 | 场景 |
|---|---|---|
| `RunnableParallel` | 多条链**并行**执行，结果合成 dict | 同一输入同时生成笑话+诗 |
| `RunnablePassthrough` | 输入**原样透传** | 输入既要给链用、又要保留原文 |

```python
map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)
result = map_chain.invoke({"topic": "程序员"})
result["joke"]  # 笑话
result["poem"]  # 诗
```

---

## Day 7：对话记忆 Memory（面试高频 ⭐）✅

### 核心原理（已亲手验证）

> 🔥 **模型没有记忆！"记忆"= 每轮把完整历史消息列表发给模型。**

```python
messages = [SystemMessage(...)]
while True:
    messages.append(HumanMessage(content=用户输入))   # 记入"记忆本"
    response = model.invoke(messages)                  # 整本发给模型
    messages.append(AIMessage(content=response.content))  # 回答也记入
```

实验证据：第 1 轮告诉它"猫叫豆豆"，第 3 轮问它，准确回答"豆豆，3岁了"——因为 messages 列表已经累积到 7 条。

### 衍伸思考（面试加分）

- 记忆列表无限增长会怎样？→ token 爆炸 + 超上下文窗口
- 解决办法：滑动窗口（只留最近 N 轮）、摘要记忆（旧对话压缩成摘要）
- Week 5 学 LangGraph 的 checkpointer 后，记忆会被框架自动管理

---

## Day 8-9：综合实战 —— 多轮面试模拟器 ✅

### 项目结构（Week 1+2 全部知识点的合体）

```
SystemMessage 人设（面试官规则）
    ↓
循环 3 轮：
    HumanMessage(触发词) → model 出题 → AIMessage 记入
    候选人回答 → HumanMessage 记入        ← 记忆机制
    ↓
with_structured_output(Evaluation)       ← 结构化输出
    ↓
评价报告（得分 + 总评 + 建议）
```

### 关键设计模式（记住这个套路，以后所有多轮应用都一样）

1. **触发词技巧**：让 AI 主动出题时，需要给它一个 HumanMessage 触发，如"（请出题）"
2. **角色反转**：AI 的提问要作为 `AIMessage` 记入历史，人的回答才是 `HumanMessage`
3. **同一份历史，两种用法**：对话时用普通 model，评价时换 `with_structured_output` 的 model

### 验证结果

面试官成功由浅入深出了 3 题（LangChain 概念 → RAG 切分策略 → Agent 系统设计），
最后输出了结构化评价：表达 6/10、专业 5/10 + 3 条改进建议 ✅

---

## Day 10：复盘日

### Week 1+2 完整知识图谱

```
LangChain 1.0
├── 模型层：ChatDeepSeek（invoke/batch/stream/ainvoke）
├── 消息层：SystemMessage / HumanMessage / AIMessage
├── 模板层：ChatPromptTemplate（{占位符}）+ Few-shot
├── 解析层：StrOutputParser / with_structured_output(Pydantic)
├── 编排层：LCEL 管道 | + RunnableParallel + RunnablePassthrough
└── 记忆层：消息列表累积（原理）→ LangGraph checkpointer（Week 5 学）
```

### 本周面试题自测

- [x] LCEL 是什么？相比旧版 Chain 的优势？
- [x] 画出管道里数据的流向
- [x] RunnableParallel 和 RunnablePassthrough 的区别？
- [x] 记忆的本质？记忆列表过长怎么办？
- [x] 多轮对话程序的基本骨架怎么写？

---

## 🐛 本周踩坑记录（重要！）

| 坑 | 原因 | 解决 |
|---|---|---|
| `UnicodeEncodeError: 'ascii' codec` | 根目录 .env 没填真实 Key（还是中文占位符），header 编码失败 | 把 week01\.env 同步到根目录和 week02 |
| 管道输入交互脚本中文变问号 | PowerShell 管道 stdin 编码问题 | 交互脚本手动运行；验证用非交互 demo 版 |

> 💡 **经验**：`.env` 只留一份最好（放项目根目录），`load_dotenv()` 会自动向上查找。

---

## 📁 Week 2 代码文件清单

| 文件 | 内容 |
|---|---|
| `day06_lcel.py` | RunnableParallel / RunnablePassthrough |
| `day07_chatbot.py` | 交互式聊天机器人 |
| `day07_memory_demo.py` | 记忆原理演示 |
| `day08_interview_simulator.py` | 面试模拟器（交互版，可真人玩） |
| `day08_interview_demo.py` | 面试模拟器（演示版，已验证） |

---

## ⏭️ 下一站：Week 3 —— RAG 检索增强生成（⭐ 最重要）

```
文档 → 加载(Loader) → 切分(Splitter) → 向量化(Embedding) → 存入向量库(Chroma)
提问 → 问题向量化 → 相似度检索 → 相关片段拼进 Prompt → 模型生成答案
```

面试必考，务必吃透！
