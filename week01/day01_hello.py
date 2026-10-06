# ============================================
# Day 1 - Step 1.6：第一个 LangChain 程序
# 目标：调通 DeepSeek 模型，验证环境搭建成功
# ============================================
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 里的 API Key

from langchain_deepseek import ChatDeepSeek

# 创建模型对象：model 指定用哪个模型，temperature 控制输出随机性
model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

# invoke = 同步调用模型，输入一句话，返回 AIMessage 对象
response = model.invoke("用一句话解释什么是 LangChain")

# response.content 是模型的文字回答
print(response.content)
