# Week 6 知识点整理（Day 31 ~ Day 38）

> 📅 整理日期：2026-10-05 | 状态：全部完成 ✅
>
> 主题：LangSmith 调试 + 两个求职级实战项目

---

## Day 31：LangSmith 链路追踪 ✅

### 配置方法（3 行环境变量）

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_你的key
LANGSMITH_PROJECT=my-project
```

配好后运行任何 LangChain 代码，自动上报到 smith.langchain.com 网页，
能看到：每一次模型调用的输入/输出、耗时、token 消耗、链的每一步。

> **面试话术**："我用 LangSmith 做全链路追踪和效果评估，
> 能快速定位是检索环节还是生成环节出了问题。"

---

## ⭐ 求职项目 A：企业知识库问答系统 ✅

### 技术架构

```
用户提问
  ↓
retriever 检索 Chroma 向量库（k=3）
  ↓
Prompt = system(资料+防幻觉规则) + MessagesPlaceholder(对话历史) + 问题
  ↓
RunnableParallel → answer（答案）+ sources（引用来源）
  ↓
history 列表累积本轮对话（实现多轮记忆）
```

### 核心亮点（写进简历）

1. **多轮对话记忆**：`MessagesPlaceholder("history")` 插入历史消息
   - 实验验证：第1轮问"病假要什么证明"，第2轮追问"那工资怎么发？"
     模型理解"那"指的还是病假 → 正确回答"病假工资按80%发放" ✅
2. **来源引用**：RunnableParallel 同时输出答案和检索片段，可追溯
3. **防幻觉**：prompt 约束"资料里没有就说没有"

### 生产化升级方向（面试可聊）

- FastAPI 包成 HTTP 接口 + Streamlit 做界面
- 检索优化：重排序（Rerank）、混合检索（关键词+向量）
- LangSmith 评估：准备测试集，量化答案准确率

---

## ⭐ 求职项目 B：多工具研究型 Agent ✅

### 流程图（LangGraph 状态机）

```
START → 生成搜索词 → 搜索 → 总结 → 反思 →【条件边】
                                      ├─ 资料不足且<2轮 → 回"生成搜索词"（循环）
                                      └─ 否则 → END 输出报告
```

### 核心亮点

1. **反思机制**：reflect 节点让模型自我评估"资料够不够"，不够就自动补充搜索
   - 实验验证：第1轮反思"资料不足"→ 自动换了英文关键词再搜一轮 ✅
2. **防死循环**：`rounds` 计数器，最多 2 轮强制出报告 ✅
3. **工具抽象**：搜索工具可无缝替换为 Tavily/SerpAPI 真实搜索

### 实验结果

2 轮搜索后输出研究报告："LangChain 1.0 于2025年发布，核心是 LCEL 编排和
create_agent 智能体入口，废弃了旧的 Chain API。" ✅

---

## 🎯 两个项目的选择建议

| 项目 | 对应岗位 | 核心考点 |
|---|---|---|
| A 知识库问答 | RAG 应用工程师 | 检索全链路、记忆、防幻觉 |
| B 研究型 Agent | Agent 开发工程师 | 状态机、反思循环、工具调用 |

时间够就都做，不够就深耕一个 + 能讲清另一个的原理。

---

## 📁 Week 6 代码文件清单

| 文件 | 内容 |
|---|---|
| `day31_langsmith.py` | LangSmith 配置检查 + 演示 |
| `project_a_kb_qa.py` | ⭐ 项目A：多轮知识库问答（带来源引用） |
| `project_b_research_agent.py` | ⭐ 项目B：研究型 Agent（反思循环） |

---

## ⏭️ 下一站：Week 7 —— 面试冲刺（高频题 + 简历包装）
