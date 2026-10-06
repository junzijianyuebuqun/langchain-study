# LangChain 1.0 学习实战笔记

> 从零基础到求职的 LangChain / LangGraph 系统学习记录，基于 **LangChain 1.0 + LangGraph + DeepSeek API**，全程可运行代码 + 知识点文档。
>
> 技术栈：Python 3.12 · LangChain 1.3 · LangGraph 1.2 · DeepSeek · Chroma · sentence-transformers

## 📂 项目结构（按官网模块顺序：LangChain → LangGraph → LangSmith）

> 🏷️ 模块图例：🦜 = LangChain | 🕸️ = LangGraph | 📊 = LangSmith
> 详见 [notes/模块地图.md](notes/模块地图.md)

```
langchain_study/
├── plan/          # 45天详细学习计划
├── notes/         # Week1~8 知识点整理 + 面试冲刺 + 模块地图
├── week01/        # 🦜 核心三件套：ChatModel / Prompt / 输出解析
├── week02/        # 🦜 LCEL 管道 + 对话记忆 + 面试模拟器
├── week03/        # 🦜 RAG 全流程：加载→切分→Embedding→向量库→问答
├── week04/        # 🦜 Tool 工具 + create_agent 智能体（ReAct 循环）
├── week05/        # 🕸️ LangGraph：State/Node/Edge + 持久化 + 人工介入
├── week06/        # 📊 LangSmith + ⭐求职项目：知识库问答 + 研究型 Agent
├── week08/        # 🦜🕸️📊 官方 Learn 教程补全（7 个高级实战）
└── official_tutorials/  # 📘 与官网教程页一一对应的代码（读官网配套练习）
```

## ⭐ 核心项目

### 项目 A：企业知识库智能问答系统（week06/project_a_kb_qa.py）

- RAG 全链路：文档加载 → 递归切分 → BGE 中文向量化 → Chroma 相似度检索
- 多轮对话记忆（MessagesPlaceholder），支持上下文追问
- 防幻觉 Prompt 约束 + 来源引用，答案可追溯

### 项目 B：多工具研究型 Agent（week06/project_b_research_agent.py）

- LangGraph 状态机编排：搜索 → 总结 → 反思 → 补充搜索（条件循环）
- rounds 计数器防死循环，工具层可无缝切换 Tavily/SerpAPI

## 🚀 快速开始

```bash
# 1. 安装依赖（Python ≥ 3.10）
pip install langchain langchain-deepseek langchain-chroma langchain-community \
            langchain-huggingface sentence-transformers chromadb python-dotenv pypdf

# 2. 配置 API Key：复制 .env.example 为 .env，填入你的 DeepSeek Key
DEEPSEEK_API_KEY=sk-xxx

# 3. 运行示例（以 RAG 为例）
python week03/day14_rag_full.py
```

## 📖 学习笔记索引

| 文档 | 内容 |
|---|---|
| [Week1 知识点](notes/Week1知识点整理.md) | ChatModel / Prompt / 输出解析 |
| [Week2 知识点](notes/Week2知识点整理.md) | LCEL / 记忆 / 面试模拟器 |
| [Week3 知识点](notes/Week3知识点整理.md) | ⭐ RAG 检索增强生成 |
| [Week4 知识点](notes/Week4知识点整理.md) | Tool / Agent / ReAct |
| [Week5 知识点](notes/Week5知识点整理.md) | LangGraph 状态机 |
| [Week6 知识点](notes/Week6知识点整理.md) | LangSmith / 求职项目 |
| [Week7 面试冲刺](notes/Week7面试冲刺.md) | 12 道高频题 + 简历模板 |
| [Week8 官方教程补全](notes/Week8官方教程补全.md) | 分类/提取/摘要/SQL/Agentic RAG/图数据库/评估 |
