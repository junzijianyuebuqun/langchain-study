# LangChain 零基础到求职 详细学习计划

> **学员画像**：Python 熟练 ✅ | 使用 DeepSeek API ✅ | 每天 3-4 小时 ✅ | 目标：转行 AI 应用开发岗 ✅
>
> **总周期**：约 6-7 周（45 天左右）
>
> **技术主线**：LangChain 1.0 + LangGraph + DeepSeek API（**不要学旧版 API**，如 `LLMChain`、`AgentExecutor` 已废弃）

---

## 📋 总体路线图

```
Week 1  → 环境搭建 + 核心三件套（ChatModel / Prompt / 输出解析）
Week 2  → LCEL 管道 + 流式输出 + 结构化输出
Week 3  → RAG 检索增强生成（重点！面试必考）
Week 4  → Tool 工具 + Agent 智能体
Week 5  → LangGraph 高级编排
Week 6  → LangSmith 调试 + 两个求职级实战项目
Week 7  → 面试知识点复盘 + 简历项目包装
Week 8  → 官方 Learn 教程补全（分类/提取/摘要/SQL/Agentic RAG/图数据库/评估）
```

---

## 🗓️ Week 1：环境搭建 + 核心三件套

### Day 1：环境搭建（预计 2 小时）

**Step 1.1 检查 Python 版本（5 分钟）**
```powershell
python --version
```
- 要求 **≥ 3.10**（LangChain 1.0 硬性要求），如果低于 3.10，去 python.org 下载 3.12 安装

**Step 1.2 创建项目文件夹和虚拟环境（10 分钟）**
```powershell
mkdir D:\kimi\langchain-learn
cd D:\kimi\langchain-learn
python -m venv .venv
.venv\Scripts\Activate.ps1
```
- 如果激活脚本报错（执行策略限制），运行：`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- 激活成功后命令行前面会出现 `(.venv)`

**Step 1.3 安装依赖（5 分钟）**
```powershell
pip install langchain langchain-deepseek langgraph langsmith python-dotenv
```

**Step 1.4 注册 DeepSeek 获取 API Key（15 分钟）**
1. 打开 https://platform.deepseek.com
2. 手机号注册 → 实名认证
3. 左侧菜单「API keys」→「创建 API key」→ 复制保存（**只显示一次**）
4. 充值 10 元即可（够你学完全部课程，DeepSeek 非常便宜）

**Step 1.5 配置环境变量（10 分钟）**
在项目文件夹创建 `.env` 文件：
```
DEEPSEEK_API_KEY=sk-你的key
```

**Step 1.6 验证环境 —— 运行第一个程序（30 分钟）**
创建 `day01_hello.py`：
```python
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 里的 API Key

from langchain_deepseek import ChatDeepSeek

model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
response = model.invoke("用一句话解释什么是 LangChain")
print(response.content)
```
运行：`python day01_hello.py`
- ✅ 看到中文回答 = 环境搭建成功
- ❌ 报 401 错误 = API Key 错了；报网络超时 = 检查网络

**Step 1.7 理解刚才发生了什么（30 分钟）**
- `ChatDeepSeek` 是什么：LangChain 对 DeepSeek 模型的统一封装
- `invoke()` 是什么：同步调用模型，输入字符串/消息，输出 AIMessage 对象
- `temperature` 是什么：0 = 输出固定，1 = 输出随机
- 动手实验：改 temperature 为 0 和 1.5，各问同一个问题 3 次，观察区别

> 🎯 **Day 1 验收标准**：能独立跑通 hello world，并说出 invoke 的作用

---

### Day 2：ChatModel 深入（预计 3 小时）

**Step 2.1 理解三种消息类型（1 小时）**
创建 `day02_messages.py`：
```python
from dotenv import load_dotenv
load_dotenv()
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatDeepSeek(model="deepseek-chat")

messages = [
    SystemMessage(content="你是一个严厉的面试官，只回答技术问题"),
    HumanMessage(content="什么是 RAG？"),
]
response = model.invoke(messages)
print(response.content)
```
- **必须背下来的概念**（面试高频）：
  - SystemMessage：设定角色和行为准则
  - HumanMessage：用户的输入
  - AIMessage：模型的回复
- 动手实验：把 SystemMessage 改成"用东北话回答"，观察效果

**Step 2.2 掌握 4 种调用方式（1 小时）**
```python
# 1. 同步调用
model.invoke("你好")

# 2. 批量调用（一次传多个问题，并行处理）
model.batch(["1+1=?", "2+2=?"])

# 3. 流式输出（打字机效果，面试必问！）
for chunk in model.stream("写一首关于程序员的短诗"):
    print(chunk.content, end="", flush=True)

# 4. 异步调用
import asyncio
async def main():
    r = await model.ainvoke("你好")
    print(r.content)
asyncio.run(main())
```
- 每种都亲手敲一遍运行，感受区别

**Step 2.3 常用参数实验（1 小时）**
- `temperature`、`max_tokens`、`top_p` 逐个实验
- 记录笔记：每个参数调大调小对输出的影响

> 🎯 **Day 2 验收标准**：能默写出三种消息类型和四种调用方式

---

### Day 3：Prompt 提示词模板（预计 3 小时）

**Step 3.1 ChatPromptTemplate 基础（1 小时）**
创建 `day03_prompt.py`：
```python
from dotenv import load_dotenv
load_dotenv()
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

model = ChatDeepSeek(model="deepseek-chat")

# 定义模板：{topic} 是占位符变量
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{role}，用通俗易懂的语言解释概念"),
    ("human", "请解释什么是{topic}")
])

# 把模板和模型用 | 连起来（这就是 LCEL，明天细讲）
chain = prompt | model
response = chain.invoke({"role": "老师", "topic": "向量数据库"})
print(response.content)
```

**Step 3.2 Few-shot 少样本提示（1 小时，面试高频）**
```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "开心", "output": "难过"},
    {"input": "高", "output": "矮"},
]
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])
few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个反义词转换器"),
    few_shot,
    ("human", "{input}"),
])
chain = final_prompt | model
print(chain.invoke({"input": "快"}).content)
```

**Step 3.3 练习作业（1 小时）**
写一个「简历优化助手」：输入一段项目描述，用 prompt 模板让模型输出优化后的版本（要求带上岗位类型变量）

> 🎯 **Day 3 验收标准**：理解占位符变量，能解释 Few-shot 为什么有效（给模型示例让它模仿格式）

---

### Day 4：输出解析（预计 3 小时，面试高频！）

**Step 4.1 为什么需要结构化输出（30 分钟）**
- 模型默认输出自然语言文本，程序没法直接用
- 实际开发中我们需要 JSON 格式的数据 → 这就是输出解析要解决的

**Step 4.2 with_structured_output（新版标准写法，1.5 小时）**
创建 `day04_structured.py`：
```python
from dotenv import load_dotenv
load_dotenv()
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

# 用 Pydantic 定义你想要的输出结构
class MovieReview(BaseModel):
    title: str = Field(description="电影名称")
    rating: int = Field(description="评分，1-10")
    summary: str = Field(description="100字以内的影评")

model = ChatDeepSeek(model="deepseek-chat")
structured_model = model.with_structured_output(MovieReview)

result = structured_model.invoke("评价一下电影《流浪地球》")
print(result.title, result.rating, result.summary)
```

**Step 4.3 StrOutputParser 与 LCEL 组合（1 小时）**
```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}")
chain = prompt | model | StrOutputParser()  # 输出直接是字符串，不再是 AIMessage
print(chain.invoke({"topic": "LangChain"}))
```

> 🎯 **Day 4 验收标准**：能解释 Pydantic 在这里的作用，知道结构化输出的两种场景

---

### Day 5：第一周复盘 + 小测验（预计 2-3 小时）

1. 不看笔记，从零写一个程序：用 prompt 模板 + 结构化输出，做一个「菜品推荐器」（输入口味偏好 → 输出 JSON：菜名、推荐理由、难度）
2. 口头自答这些面试题（写在笔记里）：
   - LangChain 解决了什么问题？（统一不同模型的接口，提供组件化抽象）
   - SystemMessage 和 HumanMessage 的区别？
   - 流式输出的原理和应用场景？
   - temperature 的作用？
3. 把 Week 1 所有代码整理到一个 GitHub 仓库（`git init` + 提交），**这是你简历上的第一个痕迹**

---

## 🗓️ Week 2：LCEL 管道 + 记忆

### Day 6：LCEL 表达式语言（核心范式，必须吃透）

**Step 6.1 理解管道符 `|`（1 小时）**
```python
# prompt | model | parser 这个链条叫 LCEL
# 数据流向：dict → prompt → PromptValue → model → AIMessage → parser → str
chain = prompt | model | StrOutputParser()
```
- 面试点：LCEL 是什么？（LangChain Expression Language，声明式组合组件的方式，1.0 版本取代了旧 Chain）

**Step 6.2 RunnablePassthrough 和并行（1.5 小时）**
```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# 并行执行多个链
joke_chain = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话") | model | StrOutputParser()
poem_chain = ChatPromptTemplate.from_template("写一首关于{topic}的诗") | model | StrOutputParser()

map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)
result = map_chain.invoke({"topic": "程序员"})
print(result["joke"])
print(result["poem"])
```

### Day 7：对话记忆 Memory（面试高频）

**Step 7.1 理解为什么模型"没有记忆"（30 分钟）**
- 每次调用都是独立的 HTTP 请求，模型本身不记东西
- "记忆" = 把历史消息一起发给模型

**Step 7.2 新版记忆方案（基于 LangGraph 的 checkpointer，1.5 小时）**
- 先了解概念即可，Week 5 深入；今天先手工实现记忆：维护一个 messages 列表，每轮 append

**Step 7.3 练习（1 小时）**
写一个命令行聊天机器人，支持多轮对话（循环 input + messages 累积）

### Day 8-9：综合练习 + 复盘
- 做一个「多轮对话的面试模拟器」：模型扮演面试官，记住你的历史回答，最后给出结构化评价（JSON：表达能力、专业度、改进建议）
- 这道题把 Week 1-2 所有知识点串起来了

### Day 10：休息/查漏补缺

---

## 🗓️ Week 3：RAG 检索增强生成（⭐ 面试必考，最重要的一周）

### Day 11：理解 RAG 原理（不写代码，先把原理吃透）

**必须能背下来的流程**（面试 100% 会问）：
```
文档 → 加载(Loader) → 切分(Splitter) → 向量化(Embedding) → 存入向量库(VectorStore)
用户提问 → 问题向量化 → 相似度检索 → 取出相关片段 → 拼进 Prompt → 模型生成答案
```
- 为什么需要 RAG？（模型知识有截止日期 + 不知道私有数据 + 幻觉问题）
- 为什么需要切分？（文档太长超过上下文窗口；片段越小检索越精准）
- Embedding 是什么？（把文本变成向量，语义相近的文本向量距离近）

### Day 12：文档加载与切分

```powershell
pip install langchain-community pypdf chromadb langchain-huggingface sentence-transformers
```

创建 `day12_rag_load.py`：
```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 加载（先用一个 txt 文件练习）
loader = TextLoader("test.txt", encoding="utf-8")
docs = loader.load()

# 2. 切分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每片 500 字
    chunk_overlap=50,    # 相邻片重叠 50 字（保证上下文连贯）
)
chunks = splitter.split_documents(docs)
print(f"切成了 {len(chunks)} 片")
print(chunks[0].page_content)
```
- 实验：调 chunk_size 为 100 和 1000，观察切分结果

### Day 13：Embedding + 向量库

创建 `day13_rag_vector.py`：
```python
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 用本地免费的中文 Embedding 模型（不用调 API，省钱）
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 向量化并存入 Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",  # 持久化到本地
)

# 检索测试
results = vectorstore.similarity_search("你的问题", k=3)
for r in results:
    print(r.page_content[:100])
```
- 注：首次运行会下载 embedding 模型（约 100MB），耐心等待

### Day 14：组装完整 RAG 链

创建 `day14_rag_full.py`：
```python
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

model = ChatDeepSeek(model="deepseek-chat")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
根据以下资料回答问题。如果资料里没有答案，就说"资料中没有提到"。

资料：
{context}

问题：{question}
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print(rag_chain.invoke("你的问题"))
```
- 把这行管道表达式拆开画在纸上，理解每一步的数据流（面试会让你手写）

### Day 15：RAG 优化 + 面试题（面试加分项）

了解这些优化手段（能说出名字和思路即可）：
- 检索优化：多路召回、重排序（Rerank）、混合检索（关键词+向量）
- 切分优化：按语义切分、父子文档
- 评估：答案准确率、检索命中率

自答面试题：
- RAG 完整流程？
- chunk_size 怎么选？太大太小的后果？
- 向量库有哪些？Chroma/FAISS/Milvus 区别？
- 如何减少幻觉？（prompt 约束 + 引用来源 + 检索质量控制）

### Day 16-17：本周项目 —— 个人知识库问答机器人
- 找 3-5 篇你感兴趣领域的 PDF/网页，做成知识库
- 功能要求：多轮对话 + 回答时标注引用来源 + "不知道就说不知道"
- 上传到 GitHub，写好 README（截图 + 架构图）

---

## 🗓️ Week 4：Tool 工具 + Agent 智能体

### Day 18：自定义 Tool

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，输入如 '3 * (4 + 5)'"""
    return str(eval(expression))

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    return f"{city}今天晴，25°C"  # 练习阶段先 mock

print(calculator.name)         # calculator
print(calculator.description)  # 描述很关键！模型靠它判断什么时候调用
```
- **面试点**：Tool 的三要素 = name + description + 参数 schema（description 写得好不好直接决定 Agent 质量）

### Day 19-20：create_agent 创建智能体（新版标准入口）

```python
from dotenv import load_dotenv
load_dotenv()
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent

model = ChatDeepSeek(model="deepseek-chat")

agent = create_agent(
    model=model,
    tools=[calculator, get_weather],
    system_prompt="你是一个助手，需要用工具时主动调用工具",
)

result = agent.invoke({"messages": [{"role": "user", "content": "北京天气怎么样？另外帮我算 123 * 456"}]})
for msg in result["messages"]:
    msg.pretty_print()  # 能看到完整的 思考→调工具→拿到结果→回答 过程
```
- **面试必考**：ReAct 模式 = Reasoning（思考）+ Acting（行动）循环：思考 → 选工具 → 执行 → 观察结果 → 再思考 → 直到给出最终答案

### Day 21：Agent 练习
做一个「生活助手 Agent」，至少 4 个工具：计算器、查天气（可接真实免费 API）、记事本（读写本地文件）、网页搜索（DuckDuckGo 免费）

### Day 22-23：复盘 + 面试题
- Agent 和 Chain 的区别？（Chain 流程固定；Agent 由模型自主决定调用什么工具、调几次）
- 什么是 Tool Calling / Function Calling？
- Agent 失控（死循环）怎么办？（设置最大迭代次数）

---

## 🗓️ Week 5：LangGraph 高级编排

### Day 24-25：LangGraph 核心三要素

**只记三样东西**：
- **State**：贯穿全流程的共享数据（TypedDict 定义）
- **Node**：一个函数，读 State → 干活 → 更新 State
- **Edge**：节点间的连线，条件边 = 根据 State 决定走哪条路

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    topic: str
    draft: str
    approved: bool

def write_node(state: State):
    # 调模型写初稿...
    return {"draft": "初稿内容"}

def review_node(state: State):
    # 调模型审核...
    return {"approved": True}

def should_continue(state: State):
    return "end" if state["approved"] else "write"  # 不通过就重写（循环！）

builder = StateGraph(State)
builder.add_node("write", write_node)
builder.add_node("review", review_node)
builder.add_edge(START, "write")
builder.add_edge("write", "review")
builder.add_conditional_edges("review", should_continue, {"write": "write", "end": END})
graph = builder.compile()

result = graph.invoke({"topic": "LangGraph", "draft": "", "approved": False})
```

### Day 26：持久化 Checkpointer（面试加分项）
- 概念：把每一步的 State 存下来，支持断点续跑、时间旅行、多会话隔离（thread_id）

### Day 27：Human-in-the-loop 人工介入
- 概念：流程跑到某个节点暂停，等人确认后继续（如：发文前人工审核）

### Day 28-30：本周项目 —— 自动写作 Agent
- 流程：生成大纲 → 分章节写作 → 自我审核 → 不达标自动重写（条件边循环）→ 人工确认（interrupt）→ 输出最终文章
- 这个项目能覆盖 LangGraph 80% 面试考点

---

## 🗓️ Week 6：LangSmith + 求职级实战项目

### Day 31：LangSmith 调试
1. 注册 https://smith.langchain.com（免费额度够用）
2. .env 里加三行：
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=你的key
LANGSMITH_PROJECT=my-project
```
3. 运行任何之前的代码，去网页看每一次模型调用的完整链路（输入/输出/耗时/token 数）
- 面试话术："我用 LangSmith 做链路追踪和效果评估"

### Day 32-38：⭐ 求职级项目（二选一或都做，写进简历）

**项目 A：企业级智能知识库问答系统**（对应 RAG 岗位）
- LangChain RAG + 多轮对话 + 来源引用 + LangSmith 追踪 + FastAPI 接口 + Streamlit 界面
- 亮点写法：检索优化（重排序）、幻觉控制策略、评估指标

**项目 B：多工具研究型 Agent**（对应 Agent 岗位）
- LangGraph 编排：搜索 → 阅读 → 总结 → 反思 → 补充搜索（循环）→ 生成报告
- 亮点写法：状态持久化、人工介入、失败重试、中间件

**项目完成标准**：
- README 有架构图（mermaid）、功能截图、快速启动步骤
- 代码有注释、有 .env.example（不要把真实 API Key 传上去！）

---

## 🗓️ Week 7：面试冲刺

### Day 39-41：高频面试题自测（全部要能脱口而出）
1. LangChain 是什么？解决什么问题？
2. LCEL 是什么？相比旧版 Chain 的优势？
3. 画出 RAG 完整流程图
4. Embedding 原理？余弦相似度是什么？
5. chunk_size 和 overlap 怎么选？
6. Agent 的 ReAct 循环是什么？
7. Tool 的三要素？
8. Chain vs Agent vs LangGraph 的区别和选用场景？
9. LangGraph 的 State / Node / Edge？
10. 如何做 RAG 效果评估？如何减少幻觉？
11. 流式输出的原理？
12. 多轮对话记忆怎么实现？

### Day 42-45：简历包装 + 模拟面试
- 简历项目描述模板："基于 LangChain 1.0 + DeepSeek + Chroma 构建企业知识库问答系统，实现文档加载/语义切分/向量检索/引用溯源全链路，通过重排序将检索准确率提升 X%，使用 LangSmith 进行全链路追踪与评估"
- 找人模拟面试或自问自答录音

---

## 📌 学习方法纪律（小白必读）

1. **每行代码必须亲手敲**，禁止复制粘贴 —— 肌肉记忆比看懂重要 10 倍
2. **每天结束写 3 行笔记**：今天学了什么 / 卡在哪 / 明天做什么
3. **报错先看错误最后一行**，看不懂就复制完整报错去问（DeepSeek 网页版免费）
4. **所有代码当天提交 GitHub** —— 绿色提交记录就是你的学习证明
5. 卡住超过 30 分钟就跳过做标记，不要死磕
6. 旧教程里看到 `LLMChain` / `AgentExecutor` / `initialize_agent` 直接关掉，全是废弃 API

## 📖 资料清单

| 资料 | 用途 |
|---|---|
| python.langchain.ac.cn（官方中文文档） | 查 API 的第一选择 |
| DeepSeek 网页版（chat.deepseek.com） | 免费问问题、看报错 |
| GitHub: LangChain-Chinese-Getting-Started-Guide | 按顺序练例子 |
| smith.langchain.com | 调试追踪平台 |

---

## 🗓️ Week 8：官方 Learn 页面教程补全（7 个高级实战）

> 学完主干后，对照官方教程页（python.langchain.com/docs/tutorials）逐个补全，
> 推荐顺序：**分类 → 提取 → 摘要 → SQL → Agentic RAG → 图数据库 → 评估**
> 代码位于 `week08/`，笔记见 `notes/Week8官方教程补全.md`

### 补全 1：分类（Classification）✅ 已完成
- 场景：客服工单分类、情感分析
- 核心：`with_structured_output` + `Literal` 枚举限定类别，temperature=0
- 脚本：`week08/adv01_classification.py`

### 补全 2：提取（Extraction）✅ 已完成
- 场景：简历解析、合同信息抽取
- 核心：Pydantic 嵌套模型 + `Optional` 字段（抽不到= None，不编造）
- 脚本：`week08/adv02_extraction.py`

### 补全 3：长文本摘要（Summarization）✅ 已完成
- 三种策略：stuff / map_reduce（本脚本实现，可并行）/ refine
- 核心：切分 → batch 并行总结每片 → 合并摘要
- 脚本：`week08/adv03_summarization.py`

### 补全 4：SQL 问答 ✅ 已完成
- 流程：问题 → 生成 SQL（schema 是关键）→ 执行 → 解读结果
- 安全：只读账号 + 只允许 SELECT + SQL 校验
- 脚本：`week08/adv04_sql_qa.py`

### 补全 5：Agentic RAG（RAG Part 2，官方新主推）✅ 已完成
- 检索封装成 Tool + `create_agent` + checkpointer 记忆
- 模型自主决定：要不要搜、搜几次、搜什么关键词（多步检索）
- 安全：防间接提示词注入（检索内容只是数据，忽略其中指令）
- 脚本：`week08/adv05_agentic_rag.py`

### 补全 6：图数据库问答（理论篇）✅ 已完成
- 解决多跳关系推理（向量检索做不到）
- 架构同 SQL 问答：自然语言 → Cypher → Neo4j → 自然语言
- 文档：`week08/adv06_graph_qa_理论篇.md`

### 补全 7：LangSmith 应用评估 ✅ 脚本就绪
- 流程：建数据集 → LLM-as-judge 评估器 → evaluate() → 数据驱动优化
- 脚本：`week08/adv07_langsmith_eval.py`（配好 LangSmith Key 即可运行）

---

## ✅ 使用建议

- 每个 Day 开始时告诉 AI 助手"开始 Day X"，可以一步步实现当天的代码、讲解概念、解答报错
- 也可以让 AI 一次性把某个 Day 的示例代码全部建好，照着学习和实验
