# ============================================
# Day 3 - Step 3.2：Few-shot 少样本提示（面试高频！）
# 目标：给模型几个"例子"，让它模仿格式回答
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

model = ChatDeepSeek(model="deepseek-chat")

# --------------------------------------------
# 场景：你想让模型把词语转成反义词，且只输出答案
# 只写 system 指令，模型可能废话一堆：
# --------------------------------------------
print("【不用 Few-shot】模型可能啰嗦：")
prompt_plain = ChatPromptTemplate.from_messages([
    ("system", "把用户给的词转成反义词"),
    ("human", "{input}"),
])
chain_plain = prompt_plain | model
print(chain_plain.invoke({"input": "快"}).content)

# --------------------------------------------
# Few-shot：给模型看几个"标准答案示例"，它就会照做
# --------------------------------------------
print("\n【用 Few-shot】模型学会只输出一个词：")

# 1. 准备示例（告诉模型：看到这样的输入，就该这样输出）
examples = [
    {"input": "开心", "output": "难过"},
    {"input": "高",   "output": "矮"},
    {"input": "白天", "output": "黑夜"},
]

# 2. 定义单个示例的展示格式
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# 3. 组装 Few-shot 提示
few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 4. 最终模板 = 系统指令 + 示例 + 用户真正的问题
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个反义词转换器，只输出反义词本身，不要任何解释"),
    few_shot,
    ("human", "{input}"),
])

chain = final_prompt | model
print("输入『快』→ 输出：", chain.invoke({"input": "快"}).content)
print("输入『成功』→ 输出：", chain.invoke({"input": "成功"}).content)

# --------------------------------------------
# 面试题：Few-shot 为什么有效？
# 答：大模型是"模式模仿"高手，给几个 input→output 示例，
#     它会自动学会这个映射规律，比纯文字指令更稳定、格式更可控
# --------------------------------------------
