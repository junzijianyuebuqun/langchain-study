# ============================================
# Day 6：LCEL 深入 —— RunnableParallel 并行 + RunnablePassthrough
# 面试点：LCEL 是什么？数据在管道里怎么流动？
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

model = ChatDeepSeek(model="deepseek-chat")

# --------------------------------------------
# 回顾：LCEL 管道里数据的流动
#   dict → prompt(填充变量) → model(返回AIMessage) → parser(变成str)
# --------------------------------------------

# --------------------------------------------
# 1. RunnableParallel：多条链并行执行，结果合成一个 dict
# --------------------------------------------
joke_chain = ChatPromptTemplate.from_template("讲一个关于{topic}的冷笑话，一句话") | model | StrOutputParser()
poem_chain = ChatPromptTemplate.from_template("写一句关于{topic}的诗") | model | StrOutputParser()

map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

print("【并行执行】同一个 topic 同时生成笑话和诗：")
result = map_chain.invoke({"topic": "程序员"})
print("😄 笑话：", result["joke"])
print("📜 诗句：", result["poem"])

# --------------------------------------------
# 2. RunnablePassthrough：原样透传输入
#    场景：输入既要给 A 用，又要原封不动传给 B
# --------------------------------------------
print("\n【RunnablePassthrough】输入透传：")

analysis_prompt = ChatPromptTemplate.from_template(
    "用户输入的关键词是「{topic}」，围绕它写一句广告文案"
)

chain2 = RunnableParallel(
    original=RunnablePassthrough(),          # 原样保留输入
    ad=analysis_prompt | model | StrOutputParser(),  # 输入同时给链用
)

r = chain2.invoke({"topic": "保温杯"})
print("原始输入还在：", r["original"])   # {'topic': '保温杯'}
print("生成的文案：", r["ad"])
