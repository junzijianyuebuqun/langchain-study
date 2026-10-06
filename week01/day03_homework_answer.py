# ============================================
# Day 3 - Step 3.3 练习作业：简历优化助手（参考实现）
# ⚠️ 先自己写！写完再来看这个参考答案！
#
# 需求：输入一段项目描述 + 目标岗位
#       用 prompt 模板让模型输出优化后的简历描述
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

model = ChatDeepSeek(model="deepseek-chat")

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一位资深简历优化专家，擅长把平淡的项目描述改写得专业、有亮点。
优化要求：
1. 使用 STAR 法则（情境-任务-行动-结果）
2. 多用动词开头，量化成果（数字、百分比）
3. 突出与【{job}】岗位相关的技术关键词
4. 输出 3-5 条要点，每条一句话"""),
    ("human", "我的原始项目描述：{description}"),
])

chain = prompt | model

result = chain.invoke({
    "job": "AI应用开发工程师",
    "description": "我在公司做了一个问答系统，用户可以问问题，系统会回答，用的是大模型"
})

print(result.content)
