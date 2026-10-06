# ============================================================
# 官网教程 03：分类（Classification）
# 官方地址：https://python.langchain.com/docs/tutorials/classification/
# 所属模块：LangChain（langchain_core: structured_output + prompts）
# 官方目标：用 结构化输出 把文本分类到预定义标签
#
# 【与官网的对应关系】
#   官网示例：按 攻击性/语言/情绪 给文本打标签
#   完全一致的技术：with_structured_output + Literal + few-shot
# ============================================================
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# --- 官网结构：定义分类 schema（Literal 限定枚举） ---
class Classification(BaseModel):
    aggression: int = Field(description="攻击性程度，1-10", ge=1, le=10)
    language: Literal["中文", "英文", "日文", "其他"] = Field(description="文本语言")
    sentiment: Literal["正面", "中性", "负面"] = Field(description="情感倾向")

model = init_chat_model("deepseek-chat", model_provider="deepseek", temperature=0)

# --- 官网结构：tagging prompt ---
tagging_prompt = ChatPromptTemplate.from_template(
    """从以下文本中提取所需的分类属性。

只提取 'Classification' 结构中提到的属性。

文本：
{input}
"""
)

# --- 官网结构：with_structured_output ---
chain = tagging_prompt | model.with_structured_output(Classification)

# 官网示例文本（攻击性高）：换中文场景
texts = [
    "你们的产品真是垃圾！气死我了，我要投诉！",
    "今天天气不错，心情很好",
    "I think this movie is pretty good",
]

for t in texts:
    r = chain.invoke({"input": t})
    print(f"📄 {t[:20]}...")
    print(f"   攻击性={r.aggression}/10 | 语言={r.language} | 情感={r.sentiment}\n")
