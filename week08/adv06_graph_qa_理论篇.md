# 官方教程补全 6：图数据库问答（Graph QA）—— 理论篇

> 官方地址：/docs/tutorials/graph/
>
> ⚠️ 本教程不提供可运行脚本：官方实现依赖 **Neo4j 图数据库**（需要安装数据库服务），
> 本地搭建成本高、求职面试中考频低，**了解原理和架构即可**。

---

## 1. 它解决什么问题？

普通 RAG 擅长"找相似文本"，但不擅长**关系推理**：

```
❌ 普通 RAG 难回答："张伟的领导的领导的部门有多少人？"
   （需要沿关系链跳三次，向量检索做不到）

✅ 图数据库天生就是存"关系"的：
   (张伟)-[:汇报给]->(王强)-[:汇报给]->(李总)-[:管理]->(技术部)
```

## 2. 官方教程的架构（和 SQL 问答如出一辙）

```
用户自然语言问题
      ↓
模型生成 Cypher 查询语句（Neo4j 的查询语言，类似 SQL）
      ↓
在图数据库中执行
      ↓
模型把结果翻译成自然语言
```

对比 SQL 问答（adv04）：只是把"生成 SQL"换成"生成 Cypher"，思路完全一样！

## 3. 核心代码示意（了解即可）

```python
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="...")
chain = GraphCypherQAChain.from_llm(llm=model, graph=graph)
chain.invoke("张伟的领导的领导管理哪个部门？")
```

## 4. 面试怎么说？（30 秒版本）

> "图数据库问答是 RAG 的补充方案：向量检索擅长语义相似，但不擅长多跳关系推理。
> 官方方案是让模型生成 Cypher 语句查询 Neo4j，架构和 SQL 问答一致——
> 自然语言 → 查询语言 → 执行 → 自然语言回答。
> 适合知识图谱场景，比如组织架构、供应链关系、风控关联分析。"

## 5. 如果想实操

需要：安装 Neo4j Desktop（免费）→ `pip install langchain-neo4j` → 按官方教程跑。
建议在完成主线学习、且目标岗位涉及知识图谱时再投入。
