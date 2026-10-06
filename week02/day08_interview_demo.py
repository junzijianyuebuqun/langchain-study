# ============================================
# Day 8-9：面试模拟器（演示版，用于验证逻辑）
# 用预设回答模拟候选人，自动跑完整流程
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

model = ChatDeepSeek(model="deepseek-chat")

messages = [
    SystemMessage(content="""你是「AI应用开发工程师」岗位的资深面试官。
规则：
1. 每次只问一个问题，问题围绕 LangChain、RAG、Agent
2. 由浅入深：第1题基础概念，第2题原理理解，第3题实战场景
3. 只输出问题本身，不要评价对方的回答""")
]

# 预设的候选人回答（模拟 input()）
preset_answers = [
    "LangChain是一个统一大模型接口的框架，用LCEL管道把提示词、模型、输出解析器组合起来",
    "RAG就是先检索后生成：文档切分后做向量化存进向量库，提问时检索相关片段拼进prompt，解决模型知识过期和私有数据的问题",
    "我会用create_agent创建智能体，定义好工具的三要素，再用LangSmith做链路追踪，设置最大迭代次数防止死循环",
]

for round_num in range(1, 4):
    messages.append(HumanMessage(content="（面试开始，请出题）" if round_num == 1 else "（请出下一题）"))
    question = model.invoke(messages)
    messages.append(AIMessage(content=question.content))
    print(f"\n👔 面试官（第{round_num}题）：{question.content}")

    answer = preset_answers[round_num - 1]
    messages.append(HumanMessage(content=answer))
    print(f"🙋 候选人：{answer}")

class Evaluation(BaseModel):
    """面试评价"""
    expression: int = Field(description="表达能力得分，1-10")
    profession: int = Field(description="专业度得分，1-10")
    overall: str = Field(description="总体评价，100字以内")
    suggestions: list[str] = Field(description="改进建议，3条")

eval_model = model.with_structured_output(Evaluation)
messages.append(HumanMessage(content="（面试结束，请根据以上对话给出评价）"))
result = eval_model.invoke(messages)

print("\n" + "=" * 40)
print("📋 面试评价报告")
print("=" * 40)
print(f"表达能力：{'⭐' * result.expression} ({result.expression}/10)")
print(f"专业度：  {'⭐' * result.profession} ({result.profession}/10)")
print(f"总体评价：{result.overall}")
print("改进建议：")
for i, s in enumerate(result.suggestions, 1):
    print(f"  {i}. {s}")
