# Week 3 知识点整理（Day 11 ~ Day 17）

> 📅 整理日期：2026-10-05 | 状态：全部完成 ✅
>
> 主题：⭐ RAG 检索增强生成（面试必考，整个学习计划最重要的一周）

---

## Day 11：RAG 原理（必背流程图）

```
【建库阶段】文档 → 加载(Loader) → 切分(Splitter) → 向量化(Embedding) → 存入向量库(Chroma)
【问答阶段】提问 → 问题向量化 → 相似度检索 → 相关片段拼进 Prompt → 模型生成答案
```

> **面试标准答案 —— 为什么需要 RAG？**
> ① 模型知识有截止日期 ② 模型不知道企业私有数据 ③ 减少幻觉（让模型"看着资料回答"）

---

## Day 12：文档加载与切分 ✅

### 核心组件

| 组件 | 作用 | 代码 |
|---|---|---|
| `TextLoader` / `PyPDFLoader` | 文件 → Document 对象列表 | `TextLoader("a.txt", encoding="utf-8").load()` |
| `RecursiveCharacterTextSplitter` | 长文档切小块 | `chunk_size=200, chunk_overlap=30` |

- **Document 对象** = `page_content`(文本) + `metadata`(元数据)

> **面试标准答案 —— chunk_size 怎么选？**
> 太大：一片混多个主题，检索不精准；太小：语义不完整，一句话被切断。
> 经验值：中文 200~500 字，overlap 取 10%~15%（保证上下文连贯）。

### 实验结果

556 字的《公司请假制度》被切成 5 片，每片一个主题（请假类型/流程/注意事项）✅

---

## Day 13：Embedding + 向量库 ✅

### 核心概念

> **面试标准答案 —— Embedding 是什么？**
> 把文本变成向量（一串数字）。语义相近的文本，向量距离近。
> 所以能"按意思搜索"而不是"按关键词匹配"。

### 关键代码

```python
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")  # 本地免费中文模型
vectorstore = Chroma.from_documents(
    documents=chunks, embedding=embeddings,
    persist_directory="week03/chroma_db",  # 持久化到本地
)
results = vectorstore.similarity_search("年假有几天？", k=2)
```

### 语义搜索的威力（实验证据）

问"**工资**怎么算？"——文档里没有任何"工资"标题，但向量检索依然找到了
"病假期间工资按80%发放"这片。关键词搜索做不到，这就是语义理解 ✅

### 常见向量库对比（面试题）

| 向量库 | 特点 | 场景 |
|---|---|---|
| Chroma | 轻量、嵌入式、本地文件 | 学习/小型项目 |
| FAISS | Meta 出品，纯计算库，极快 | 中等规模、单机 |
| Milvus | 分布式、功能全 | 生产环境、大规模 |

---

## Day 14：组装完整 RAG 链（⭐ 面试会让你手写）✅

### 核心代码（必背）

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

### 数据流向（画在纸上！）

```
"年假有几天？"
   ├──→ retriever 检索 → 3 个相关片段 → format_docs 拼成文本 → {context}
   └──→ RunnablePassthrough 原样透传 ────────────────────────→ {question}
                              ↓
              prompt 填充 → model 生成 → parser 输出答案
```

### 防幻觉的关键写法

Prompt 里加一句：**"只根据资料回答，资料里没有就说'资料中没有提到'"**

实验验证：问"公司食堂几点开饭？"（文档里没有）→ 模型正确回答"资料中没有提到" ✅
没有这句话，模型就会开始编造。

---

## Day 16-17：本周项目 —— 知识库问答机器人 ✅

### 在 Day 14 基础上加"引用来源"功能

```python
rag_chain = RunnableParallel(
    answer=({...} | prompt | model | StrOutputParser()),  # 答案
    sources=retriever,                                     # 同时带出原文片段
)
result["answer"]   # 文字回答
result["sources"]  # 引用了哪几片文档（可追溯、可验证）
```

### 验证结果

- "年假有多少天？怎么休？" → 准确回答 + 列出 3 条引用来源 ✅
- "突发疾病怎么请假？" → 找到"紧急情况可电话告知，2天内补办手续" ✅
- "公司团建多久一次？" → 正确回答"资料中没有提到" ✅

---

## 🐛 本周踩坑记录

| 坑 | 原因 | 解决 |
|---|---|---|
| `table collections already exists` | `langchain_community.vectorstores.Chroma` 与新版 chromadb 不兼容 | 换官方新包：`pip install langchain-chroma`，`from langchain_chroma import Chroma` |
| HF 模型下载慢 | huggingface.co 国内访问慢 | 脚本开头加 `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"` |

---

## 📁 Week 3 代码文件清单

| 文件 | 内容 |
|---|---|
| `company_policy.txt` | 示例知识库文档（请假制度） |
| `day12_rag_load.py` | 文档加载 + 切分 |
| `day13_rag_vector.py` | Embedding + 入向量库 + 检索 |
| `day14_rag_full.py` | 完整 RAG 链（防幻觉） |
| `day16_kb_bot.py` | 知识库问答机器人（带来源引用） |
| `chroma_db/` | 持久化的向量库数据 |

---

## ⏭️ 下一站：Week 4 —— Tool 工具 + Agent 智能体
