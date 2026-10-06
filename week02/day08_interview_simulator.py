# ============================================
# Day 8-9：综合练习 —— 多轮面试模拟器（交互版）
# 综合考点：三种消息 + 记忆(消息列表) + Prompt设计 + 结构化输出
#
# 玩法：模型扮演面试官，连续问你 3 个问题（记得你之前的回答）
#       3 轮结束后，输出 JSON 格式的结构化评价
#
# 运行：.\python_3_12\python.exe -X utf8 week02\day08_interview_simulator.py
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

model = ChatDeepSeek(model="deepseek-chat")

# --------------------------------------------
# 1. 面试官人设
# --------------------------------------------
messages = [
    SystemMessage(content="""你是「AI应用开发工程师」岗位的资深面试官。
规则：
1. 每次只问一个问题，问题围绕 LangChain、RAG、Agent
2. 由浅入深：第1题基础概念，第2题原理理解，第3题实战场景
3. 只输出问题本身，不要评价对方的回答""")
]

# --------------------------------------------
# 2. 三轮问答（记忆 = 消息列表不断累积）
# --------------------------------------------
for round_num in range(1, 4):
    # 触发面试官出题
    messages.append(HumanMessage(content="（面试开始，请出题）" if round_num == 1 else "（请出下一题）"))
    question = model.invoke(messages)
    messages.append(AIMessage(content=question.content))
    print(f"\n👔 面试官（第{round_num}题）：{question.content}")

    # 候选人回答
    answer = input("🙋 你：")
    messages.append(HumanMessage(content=answer))

# --------------------------------------------
# 3. 面试结束 → 结构化评价
# --------------------------------------------
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
