# ============================================
# 官方教程补全 2：提取（Extraction）
# 官方地址：/docs/tutorials/extraction/
# 场景：从非结构化文本中抽出结构化数据（简历解析、合同关键信息抽取）
# 核心技术：Pydantic 嵌套模型 + Optional 字段（抽不到就为None）
# ============================================
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 1. 定义要提取的结构（官方教程风格：嵌套 + 可选字段）
class Education(BaseModel):
    school: str = Field(description="学校名称")
    major: Optional[str] = Field(default=None, description="专业，没提到则为None")

class Person(BaseModel):
    """从文本中提取的人物信息"""
    name: str = Field(description="姓名")
    age: Optional[int] = Field(default=None, description="年龄，没提到则为None")
    skills: list[str] = Field(default_factory=list, description="技能列表")
    education: Optional[Education] = Field(default=None, description="教育经历")

model = ChatDeepSeek(model="deepseek-chat", temperature=0)

# 2. 官方教程关键点：prompt 里要明确"只提取文本中明确提到的信息"
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是信息提取专家。只提取文本中明确提到的信息，"
               "文本没有提到的字段留空（None），不要编造。"),
    ("human", "{text}"),
])

chain = prompt | model.with_structured_output(Person)

# 3. 测试：第二段故意缺年龄和教育，验证 Optional 机制
texts = [
    "张伟，28岁，精通Python和LangChain，毕业于浙江大学计算机专业",
    "李梅熟悉RAG技术和向量数据库",
]

for t in texts:
    r = chain.invoke({"text": t})
    print(f"📄 原文：{t}")
    print(f"   姓名={r.name} 年龄={r.age} 技能={r.skills} 教育={r.education}\n")
