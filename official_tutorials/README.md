# 📖 官方教程对照学习区（official_tutorials）

> **使用方法**：左边打开官网教程页，右边跑对应脚本。编号与官网教程页顺序完全一致。
>
> 官网入口：https://python.langchain.com/docs/tutorials/ （中文镜像：https://python.langchain.ac.cn/docs/tutorials/）

## 🗺️ 官网 ↔ 本地脚本 对照表

> 🏷️ 模块图例：🦜 = LangChain | 🕸️ = LangGraph | 📊 = LangSmith
> 顺序与官网教程页完全一致（入门组 → 编排组 → 评估组）

### 入门组（Get started）`🦜 LangChain`

| # | 官网教程 | 本地脚本 | 所属模块 | 状态 |
|---|---|---|---|---|
| 01 | [聊天模型和提示词](https://python.langchain.com/docs/tutorials/llm_chain/) | `01_chat_models_and_prompts.py` | 🦜 langchain_core | ✅ 已验证 |
| 02 | [语义搜索](https://python.langchain.com/docs/tutorials/retrievers/) | `02_semantic_search.py` | 🦜 集成包 loaders/vectorstores | ✅ 已验证 |
| 03 | [分类](https://python.langchain.com/docs/tutorials/classification/) | `03_classification.py` | 🦜 structured_output | ✅ 已验证 |
| 04 | [提取](https://python.langchain.com/docs/tutorials/extraction/) | `04_extraction.py` | 🦜 structured_output | ✅ 已验证 |

### 编排组（Orchestration）`🦜 LangChain + 🕸️ LangGraph`

| # | 官网教程 | 本地脚本 | 所属模块 | 状态 |
|---|---|---|---|---|
| 05 | [聊天机器人](https://python.langchain.com/docs/tutorials/chatbot/) | `05_chatbots.py` | 🦜 messages/trim_messages | ✅ 已验证 |
| 06 | [智能体](https://python.langchain.com/docs/tutorials/agents/) | `06_agents.py` | 🦜 agents（底层🕸️） | ✅ 已验证 |
| 07 | [RAG 第一部分](https://python.langchain.com/docs/tutorials/rag/) | `07_rag_part1.py` | 🦜 agents + vectorstores | ✅ 已验证 |
| 08 | [RAG 第二部分](https://python.langchain.com/docs/tutorials/qa_chat_history/) | `08_rag_part2.py` | 🦜 agents + 🕸️ checkpointer | ✅ 已验证 |
| 09 | [基于 SQL 的问答](https://python.langchain.com/docs/tutorials/sql_qa/) | `09_sql_qa.py` | 🦜 prompts + SQL | ✅ 已验证 |
| 10 | [摘要](https://python.langchain.com/docs/tutorials/summarization/) | `10_summarization.py` | 🦜 text_splitters + LCEL | ✅ 已验证 |
| 11 | [基于图数据库的问答](https://python.langchain.com/docs/tutorials/graph/) | `11_graph_qa_理论篇.md` | 🦜 langchain-neo4j | ✅ 理论篇 |

### 评估组 `📊 LangSmith`

| # | 官网教程 | 本地脚本 | 所属模块 | 状态 |
|---|---|---|---|---|
| 12 | [评估你的 LLM 应用](https://docs.smith.langchain.com/evaluation) | `12_langsmith_evaluation.py` | 📊 evaluation | ✅ 配 Key 可跑 |

## 🔄 与官网代码的差异说明（只有这两类）

1. **模型替换**：官网用 `init_chat_model("gpt-xxx")` → 本地统一 `init_chat_model("deepseek-chat", model_provider="deepseek")`
2. **数据源替换**：官网下载 Nike 财报/博客 → 本地用自生成的 PDF/知识库（避免网络问题），**处理流程与官网完全一致**

## 📌 与主学习路线的关系

```
week01~07  = 主线课程（知识体系，按 Day 循序渐进）
week08     = 知识点笔记（每个教程的原理 + 面试考点）
official_tutorials = 官网对照代码（读官网时的配套练习）
```

建议读法：
1. 先按 week01~07 主线学完基础知识
2. 打开官网教程页 + 对应编号脚本，对照着读
3. 面试前过一遍 `notes/Week8官方教程补全.md` 的考点总结
