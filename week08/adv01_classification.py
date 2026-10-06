# ============================================
# 官方教程补全 1：分类（Classification）
# 官方地址：/docs/tutorials/classification/
# 场景：把文本分到预定义类别（客服工单分类、情感分析、垃圾邮件识别）
# 核心技术：with_structured_output + Literal 限定枚举值
# ============================================
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 1. 定义分类结构：Literal 把取值限定死在枚举里（防止模型瞎编类别）
class TicketClassification(BaseModel):
    """客服工单分类结果"""
    category: Literal["账单问题", "技术故障", "产品咨询", "投诉建议"] = Field(
        description="工单所属类别"
    )
    sentiment: Literal["正面", "中性", "负面"] = Field(description="用户情绪")
    urgency: int = Field(description="紧急程度，1-5，5最紧急")

model = ChatDeepSeek(model="deepseek-chat", temperature=0)  # 分类任务用 0，要稳定

# 2. 官方教程推荐的 prompt 写法：明确告诉模型要做什么
prompt = ChatPromptTemplate.from_template(
    "对以下客服工单进行分类：\n\n{ticket}"
)

chain = prompt | model.with_structured_output(TicketClassification)

# 3. 测试三条工单
tickets = [
    "你们这个月多扣了我两百块钱！马上给我退了，不然我投诉到底！",
    "你好，请问企业版支持多少人同时使用？想了解一下报价",
    "APP更新之后一直闪退，根本打不开，工作都没法做了",
]

for t in tickets:
    r = chain.invoke({"ticket": t})
    print(f"📨 工单：{t[:25]}...")
    print(f"   类别={r.category} | 情绪={r.sentiment} | 紧急度={r.urgency}/5\n")
