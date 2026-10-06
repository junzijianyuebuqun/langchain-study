# ============================================
# Day 2 - Step 2.3：常用参数实验
# 目标：感受 temperature / max_tokens 对输出的影响
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek

# --------------------------------------------
# 实验 1：max_tokens —— 限制最大输出长度
# --------------------------------------------
print("【实验1】max_tokens=20（话说到一半就被掐断）：")
model_short = ChatDeepSeek(model="deepseek-chat", max_tokens=20)
r = model_short.invoke("介绍一下中国的长城")
print(r.content)

print("\n【对比】max_tokens=200（能说完）：")
model_long = ChatDeepSeek(model="deepseek-chat", max_tokens=200)
r2 = model_long.invoke("介绍一下中国的长城")
print(r2.content)

# --------------------------------------------
# 实验 2：观察 response 里的 token 使用量（和钱直接相关！）
# --------------------------------------------
print("\n【实验2】这次调用花了多少 token：")
print(r2.usage_metadata)
# 输出示例：{'input_tokens': 15, 'output_tokens': 150, 'total_tokens': 165}
# DeepSeek 按 token 计费：input 便宜，output 贵
