# ============================================
# Day 5：Week 1 综合测验 —— 菜品推荐器
# 综合考点：Prompt模板(占位符) + 结构化输出(Pydantic) + LCEL管道
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 1. 定义输出结构
class Dish(BaseModel):
    """菜品推荐"""
    name: str = Field(description="菜名")
    reason: str = Field(description="推荐理由，50字以内")
    difficulty: str = Field(description="制作难度：简单/中等/困难")
    time_minutes: int = Field(description="预计制作时间（分钟）")

# 2. 定义模板（两个占位符：口味偏好 + 用餐人数）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位资深中餐大厨，根据用户的口味偏好和用餐人数推荐一道菜"),
    ("human", "我的口味偏好：{taste}，用餐人数：{people}人"),
])

# 3. 组装链：模板 → 模型(带结构化输出)
model = ChatDeepSeek(model="deepseek-chat")
chain = prompt | model.with_structured_output(Dish)

# 4. 调用
result = chain.invoke({"taste": "微辣、喜欢下饭菜", "people": 2})

print(f"🍜 推荐菜品：{result.name}")
print(f"💡 推荐理由：{result.reason}")
print(f"📊 制作难度：{result.difficulty}")
print(f"⏱️  预计时间：{result.time_minutes} 分钟")
