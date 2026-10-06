# 官网教程 11：基于图数据库的问答（Graph QA）—— 理论篇

> 官方地址：https://python.langchain.com/docs/tutorials/graph/
>
> ⚠️ 官方实现依赖 Neo4j 数据库服务，本地搭建成本高、考频低，本文件讲清原理。

## 解决的问题：多跳关系推理

向量检索擅长"语义相似"，但回答不了关系链问题：

```
"张伟的领导的领导的部门有多少人？"  ← 需要沿关系跳3次
```

图数据库天然存关系：`(张伟)-[:汇报给]->(王强)-[:汇报给]->(李总)`

## 官方架构（与 SQL 问答完全同构）

```
自然语言问题 → 模型生成 Cypher 查询 → Neo4j 执行 → 模型翻译成自然语言
```

```python
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="...")
chain = GraphCypherQAChain.from_llm(llm=model, graph=graph)
chain.invoke("张伟的领导管理哪个部门？")
```

## 面试 30 秒版本

> "图问答是 RAG 的补充：向量检索管语义相似，图数据库管多跳关系。
> 架构和 SQL 问答一致——自然语言 → 查询语言（Cypher）→ 执行 → 回答。
> 适合组织架构、供应链、风控关联等知识图谱场景。"
