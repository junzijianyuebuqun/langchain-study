# ============================================
# 官方教程补全 3：长文本摘要（Summarization）
# 官方地址：/docs/tutorials/summarization/
# 核心问题：文本太长塞不进 prompt 怎么办？
# 三种策略：stuff(一次全塞) / map_reduce(分片总结再合并) / refine(逐片迭代)
# 本脚本实现最实用的 map_reduce
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

model = ChatDeepSeek(model="deepseek-chat")

# --------------------------------------------
# 准备一篇"长文章"（模拟：把请假制度重复拼接成长文）
# --------------------------------------------
with open("week03/company_policy.txt", encoding="utf-8") as f:
    long_text = f.read() * 3  # 重复3遍模拟长文档

# 1. 切分（Map 阶段的输入）
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents([Document(page_content=long_text)])
print(f"长文共 {len(long_text)} 字，切成 {len(chunks)} 片\n")

# --------------------------------------------
# 2. Map 阶段：每片单独总结
# --------------------------------------------
map_prompt = ChatPromptTemplate.from_template("用一句话概括以下内容：\n\n{text}")
map_chain = map_prompt | model | StrOutputParser()

print("【Map 阶段】逐片总结...")
partial_summaries = map_chain.batch([{"text": c.page_content} for c in chunks])
# batch = 并行处理所有片（Day 2 学的！）

# --------------------------------------------
# 3. Reduce 阶段：把所有小总结合并成最终摘要
# --------------------------------------------
reduce_prompt = ChatPromptTemplate.from_template(
    "以下是若干片段摘要，请合并成一份150字以内的完整摘要：\n\n{text}"
)
reduce_chain = reduce_prompt | model | StrOutputParser()

print("【Reduce 阶段】合并摘要...")
final = reduce_chain.invoke({"text": "\n".join(partial_summaries)})

print(f"\n📄 最终摘要：\n{final}")

# --------------------------------------------
# 三种策略对比（面试题）：
# stuff       → 快但文本长了会爆 token
# map_reduce  → 可并行、适合超长文本（本脚本），但调用次数多
# refine      → 逐片迭代更新摘要，上下文连贯，但慢且无法并行
# --------------------------------------------
