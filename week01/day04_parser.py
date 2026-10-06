# ============================================
# Day 4 - Step 4.3：StrOutputParser 输出解析器
# 目标：把 AIMessage 对象解析成纯字符串，让链的输出更干净
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatDeepSeek(model="deepseek-chat")
prompt = ChatPromptTemplate.from_template("用一句话介绍{topic}")

# --------------------------------------------
# 对比 1：不加解析器 → 拿到的是 AIMessage 对象，要自己取 .content
# --------------------------------------------
chain_raw = prompt | model
r1 = chain_raw.invoke({"topic": "LangChain"})
print("【不加解析器】返回类型：", type(r1).__name__)
print("内容要自己取：", r1.content)

# --------------------------------------------
# 对比 2：加 StrOutputParser → 链的输出直接就是字符串！
# --------------------------------------------
chain = prompt | model | StrOutputParser()
r2 = chain.invoke({"topic": "LangChain"})
print("\n【加解析器】返回类型：", type(r2).__name__)   # str
print("内容直接用：", r2)

# --------------------------------------------
# 为什么这很重要？（为 Day 6 LCEL 铺垫）
# 链可以继续往下接：prompt | model | parser | 下一个prompt | model ...
# 如果中间不解析成 str，下一环就没法把结果当文本用
# --------------------------------------------

# 连续两个链的示例：先介绍 → 再翻译成英文
translate_prompt = ChatPromptTemplate.from_template("把下面这句话翻译成英文：{text}")

full_chain = (
    {"text": prompt | model | StrOutputParser()}   # 第一环：生成中文介绍
    | translate_prompt                              # 第二环：翻译
    | model
    | StrOutputParser()
)

print("\n【两环链】中文介绍 → 自动翻译成英文：")
print(full_chain.invoke({"topic": "RAG技术"}))
