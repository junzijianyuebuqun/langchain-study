# ============================================================
# 官网教程 04：提取（Extraction）
# 官方地址：https://python.langchain.com/docs/tutorials/extraction/
# 所属模块：LangChain（langchain_core: structured_output）
# 官方目标：从非结构化文本提取结构化数据
#
# 【与官网的对应关系】
#   官网示例：从文本提取人物信息（name/age/hair_color...）
#   保留官网的 schema 风格 + "只提取明确提到的信息" 原则
# ============================================================
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# --- 官网 schema 风格（字段全部 Optional，没提到就是 None） ---
class Person(BaseModel):
    """人物信息（与官网教程 schema 对应）"""
    name: Optional[str] = Field(default=None, description="人物姓名")
    hair_color: Optional[str] = Field(default=None, description="头发颜色，如果文本提到的话")
    height_in_meters: Optional[str] = Field(default=None, description="身高（米）")

model = init_chat_model("deepseek-chat", model_provider="deepseek", temperature=0)

# --- 官网 prompt：明确"只提取文本中写明的属性" ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是信息提取算法。只提取文本中明确提到的相关信息，"
               "如果不知道某个属性的值，返回 null，不要编造。"),
    ("human", "{text}"),
])

chain = prompt | model.with_structured_output(Person)

# --- 官网示例 1：信息不全的文本 ---
text1 = "马路上，一个高个子男人走了过来"
r1 = chain.invoke({"text": text1})
print(f"📄 {text1}")
print(f"   name={r1.name} hair_color={r1.hair_color} height={r1.height_in_meters}\n")

# --- 官网示例 2：长文本（官网用爱丽丝梦游仙境，这里用中文故事） ---
text2 = "小明今年刚转学来，他有一头标志性的红发，身高一米八，在班里很显眼"
r2 = chain.invoke({"text": text2})
print(f"📄 {text2}")
print(f"   name={r2.name} hair_color={r2.hair_color} height={r2.height_in_meters}\n")

# --- 官网进阶：提取多个实体（list 包裹一层） ---
class People(BaseModel):
    """文本中出现的所有人物"""
    people: list[Person]

chain_multi = prompt | model.with_structured_output(People)
text3 = "红发的小明和黑发的小红一起去了图书馆"
r3 = chain_multi.invoke({"text": text3})
print(f"📄 {text3}")
for p in r3.people:
    print(f"   人物：name={p.name} hair={p.hair_color} height={p.height_in_meters}")
