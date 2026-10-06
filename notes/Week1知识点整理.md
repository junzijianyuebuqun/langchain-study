# Week 1 知识点整理（Day 1 ~ Day 4）

> 🏷️ 所属模块：🦜 **LangChain**（langchain_core: chat_models / messages / prompts / output_parsers + 集成包 langchain-deepseek）
> 📅 整理日期：2026-10-05 | 状态：Day 1~4 已学完 ✅
>
> 本文档持续更新，每学完一天就追加。

---

## Day 1：环境搭建 + 第一个程序

### 核心概念

| 概念 | 一句话解释 | 记忆口诀 |
|---|---|---|
| `ChatDeepSeek` | LangChain 对 DeepSeek 模型的统一封装 | 换模型只改这一行 |
| `invoke()` | 同步调用模型：输入消息 → 返回 AIMessage 对象 | "调用" |
| `response.content` | AIMessage 对象里的文字内容 | 要文字就取 .content |
| `temperature` | 随机性开关：0 = 固定输出，越大越发散 | 0 准确 / 1.5 创意 |

### 关键认知

- **temperature 实验结论**（亲手验证过）：
  - `temperature=0` → 3 次回答几乎一样（适合：代码生成、数据提取）
  - `temperature=1.5` → 3 次完全不同，但可能"说胡话"（适合：创意文案）

### 环境备忘

```
Python 环境：D:\langchain和LangGraph\langchain_study\python_3_12（Python 3.12.6）
运行命令：  cd D:\langchain和LangGraph\langchain_study
           .\python_3_12\python.exe -X utf8 week01\xxx.py
API Key：   放在 .env 文件里，代码用 load_dotenv() 加载
```

---

## Day 2：ChatModel 深入

### 三种消息类型（面试高频 ⭐）

| 类型 | 作用 | 例子 |
|---|---|---|
| `SystemMessage` | 设定角色和行为准则 | "你是一个严厉的面试官" |
| `HumanMessage` | 用户的输入 | "什么是 RAG？" |
| `AIMessage` | 模型的回复 | 模型的回答内容 |

### 四种调用方式

| 方式 | 特点 | 使用场景 |
|---|---|---|
| `invoke` | 同步，等全部生成完一次返回 | 后台任务 |
| `batch` | 一次传多个问题，并行处理 | 批量翻译/评测 |
| `stream` | 流式输出，逐字返回 | ⭐ 聊天打字机效果（面试必问） |
| `ainvoke` | 异步，不阻塞 | Web 高并发服务 |

### 关键认知（最重要！）

> 🔥 **模型本身没有记忆！** 每次调用都是独立的 HTTP 请求。
> "记忆" = 每轮对话时，把**历史消息列表**一起发给模型。
> （实验验证：把之前的 HumanMessage + AIMessage 塞进 messages 列表，模型就能"记得"你叫小明）

### 参数

- `max_tokens`：限制最大输出长度，超了直接被掐断
- `usage_metadata`：查看 token 用量 → `{'input_tokens': 7, 'output_tokens': 200}`
- **输出 token 比输入贵** → 控制输出长度 = 省钱

---

## Day 3：Prompt 提示词模板

### ChatPromptTemplate —— 占位符变量

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{role}"),   # {xxx} 是占位符
    ("human", "请解释什么是{topic}"),
])
chain = prompt | model
chain.invoke({"role": "老师", "topic": "向量数据库"})  # 调用时填值
```

- **价值**：一份模板无限复用，只换变量值
- 实验验证：`role=老师` vs `role=厨师`，同一问题回答风格完全不同

### Few-shot 少样本提示（面试高频 ⭐）

```
最终模板 = system 指令 + Few-shot 示例(2~5个) + 用户真正的问题
```

> **面试标准答案 —— Few-shot 为什么有效？**
> 大模型是"模式模仿"高手。给几个 input→output 示例，它会自动学会映射规律，
> 比纯文字指令**格式更稳定、输出更可控**。示例一般 2~5 个就够。

---

## Day 4：输出解析（面试高频 ⭐）

### 为什么需要输出解析？

模型默认输出**自然语言文本** → 程序没法直接用 → 需要变成**结构化数据**（JSON/对象）

### 两种方式

**方式 1：`with_structured_output`（要字段、要校验时用）**

```python
class MovieReview(BaseModel):           # Pydantic 定义结构
    title: str = Field(description="电影名称")
    rating: int = Field(description="评分，1-10")

structured_model = model.with_structured_output(MovieReview)
result = structured_model.invoke("评价《流浪地球》")
result.title   # 直接当 Python 对象用！
```

**方式 2：`StrOutputParser`（只要纯文本时用）**

```python
chain = prompt | model | StrOutputParser()
# 链的输出直接是字符串，不用再取 .content
# 关键作用：让链可以继续往下接（第一环的输出 → 第二环的输入）
```

### 面试标准答案 —— Pydantic 在这里的作用？

> 定义数据的结构和类型约束。模型输出的 JSON 会被**自动校验**，
> 字段缺失/类型错误会直接报错，保证程序拿到的数据一定合法。

---

## 🎯 Week 1 面试题自测（目前进度）

- [x] LangChain 解决什么问题？（统一不同模型接口 + 组件化抽象）
- [x] 三种消息类型的区别？
- [x] 流式输出的原理和应用场景？
- [x] temperature 的作用？怎么选值？
- [x] 模型为什么没有记忆？怎么实现记忆？
- [x] Few-shot 是什么？为什么有效？
- [x] 结构化输出的两种方式？Pydantic 的作用？
- [x] token 是什么？输入输出哪个贵？

---

## 📁 Week 1 代码文件清单

| 文件 | 内容 |
|---|---|
| `day01_hello.py` | 第一个程序 |
| `day01_temperature.py` | temperature 实验 |
| `day02_messages.py` | 三种消息类型 |
| `day02_invoke_methods.py` | 四种调用方式 |
| `day02_params.py` | max_tokens / token 用量 |
| `day03_prompt.py` | ChatPromptTemplate 基础 |
| `day03_fewshot.py` | Few-shot 示例 |
| `day03_homework_answer.py` | 简历优化助手（作业参考答案） |
| `day04_structured.py` | with_structured_output |
| `day04_parser.py` | StrOutputParser + 两环链 |

---

## ⏭️ 待学习（Day 5 ~ Day 7）

- Day 5：Week 1 复盘 + 菜品推荐器小测验 + 代码传 GitHub
- Day 6：LCEL 深入（RunnablePassthrough / RunnableParallel 并行）
- Day 7：对话记忆 Memory + 命令行聊天机器人

---

## Day 5：Week 1 综合测验 —— 菜品推荐器 ✅

**综合考点**：Prompt 模板 + 结构化输出 + LCEL 管道（三件套合体）

```python
class Dish(BaseModel):
    name: str = Field(description="菜名")
    difficulty: str = Field(description="难度：简单/中等/困难")

chain = prompt | model.with_structured_output(Dish)
result = chain.invoke({"taste": "微辣", "people": 2})
result.name  # "宫保鸡丁"
```

**心得**：三件套不是孤立的——模板负责"问得好"，结构化负责"答得能用"，管道负责"串起来"。

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

## 📁 代码文件清单（更新）

> ⚠️ 目录调整：Day 6、7 属于 Week 2 内容，已移到 `week02\` 文件夹

| 文件 | 位置 | 内容 |
|---|---|---|
| `day05_quiz.py` | week01\ | 菜品推荐器（三件套综合） |
| `day06_lcel.py` | week02\ | RunnableParallel / RunnablePassthrough |
| `day07_chatbot.py` | week02\ | 交互式聊天机器人（手动运行） |
| `day07_memory_demo.py` | week02\ | 记忆原理演示（已验证） |

---

## ⏭️ Week 1 毕业状态

- [x] Day 1 环境 + hello world + temperature
- [x] Day 2 三种消息 + 四种调用 + 参数
- [x] Day 3 Prompt 模板 + Few-shot
- [x] Day 4 结构化输出 + StrOutputParser
- [x] Day 5 综合测验（菜品推荐器）
- [x] Day 6 LCEL 深入
- [x] Day 7 记忆 + 聊天机器人

## ⏭️ 下一站：Week 2/3 预告

- Day 8-9：综合练习「多轮面试模拟器」
- Day 11 起：**RAG 检索增强生成**（⭐ 面试必考，最重要的一周）
  - Loader 加载文档 → Splitter 切分 → Embedding 向量化 → Chroma 向量库 → 检索问答
