# ============================================================
# 官网教程 10：摘要（Summarization）
# 官方地址：https://python.langchain.com/docs/tutorials/summarization/
# 所属模块：LangChain（text_splitters + LCEL）
# 官方目标：为（可能很长的）文本生成摘要
#
# 【与官网的对应关系】
#   官网三种策略：stuff / map-reduce / refine
#   本脚本实现 stuff（短文本）+ map_reduce（长文本），与官网一一对应
# ============================================================
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

model = init_chat_model("deepseek-chat", model_provider="deepseek")

with open("week03/company_policy.txt", encoding="utf-8") as f:
    base_text = f.read()

# ============ 策略一：Stuff（官网：短文本一次塞入） ============
print("========== 策略一：Stuff ==========")
stuff_chain = (
    ChatPromptTemplate.from_template("为以下内容写一份100字摘要：\n\n{text}")
    | model | StrOutputParser()
)
print(stuff_chain.invoke({"text": base_text}))

# ============ 策略二：Map-Reduce（官网：长文本分片处理） ============
print("\n========== 策略二：Map-Reduce ==========")
long_text = base_text * 3  # 模拟长文
chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30).split_documents(
    [Document(page_content=long_text)]
)
print(f"长文 {len(long_text)} 字 → {len(chunks)} 片")

# Map：并行总结每片（官网 map prompt）
map_chain = (
    ChatPromptTemplate.from_template("用一句话概括：\n\n{text}")
    | model | StrOutputParser()
)
partial = map_chain.batch([{"text": c.page_content} for c in chunks])

# Reduce：合并（官网 reduce prompt）
reduce_chain = (
    ChatPromptTemplate.from_template("把以下片段摘要合并成150字内的完整摘要：\n\n{text}")
    | model | StrOutputParser()
)
print(reduce_chain.invoke({"text": "\n".join(partial)}))

# ============ 策略三：Refine（官网：逐片迭代，了解原理） ============
# 第1片生成初稿 → 每来一片就"结合已有摘要+新片段"更新摘要
# 优点：上下文连贯；缺点：串行慢、调用次数 = 片数
print("\n【策略三 Refine 原理】逐片迭代更新摘要，串行执行，本脚本略（见官网）")
