# ============================================
# Day 4 - Step 4.2：with_structured_output 结构化输出（面试高频！）
# 目标：让模型直接返回 Pydantic 对象，而不是一大段自然语言
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

# --------------------------------------------
# 为什么需要这个？
# 模型默认输出是一大段文字，程序没法直接用
# 实际开发中：我们要的是"字段" → 存数据库、传给下一个接口
# --------------------------------------------

# 1. 用 Pydantic 定义你想要的输出结构（字段 + 类型 + 描述）
class MovieReview(BaseModel):
    """电影评价"""
    title: str = Field(description="电影名称")
    rating: int = Field(description="评分，1-10的整数")
    pros: list[str] = Field(description="优点列表，最多3条")
    cons: list[str] = Field(description="缺点列表，最多2条")
    summary: str = Field(description="100字以内的影评总结")

model = ChatDeepSeek(model="deepseek-chat")

# 2. 给模型"套上"结构 → 返回值自动变成 MovieReview 对象
structured_model = model.with_structured_output(MovieReview)

result = structured_model.invoke("评价一下电影《流浪地球》")

# 3. 像访问普通 Python 对象一样取字段！
print("类型：", type(result))          # <class '__main__.MovieReview'>
print("片名：", result.title)
print("评分：", result.rating, "/10")
print("优点：")
for p in result.pros:
    print("  ✅", p)
print("缺点：")
for c in result.cons:
    print("  ❌", c)
print("总结：", result.summary)

# --------------------------------------------
# 面试点：Pydantic 在这里的作用？
# 答：定义数据的结构和类型约束。模型输出的 JSON 会被自动校验，
#     字段缺失/类型错误会直接报错，保证程序拿到的数据一定是合法的
# --------------------------------------------
