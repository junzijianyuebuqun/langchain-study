# ============================================
# Day 1 - Step 1.7：temperature 参数实验
# 目标：感受"随机性"参数对模型输出的影响
# 面试点：temperature 是什么？0 = 输出固定，越大越随机
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek

question = "给我起一个有创意的奶茶店名字"

# 实验 1：temperature=0（最保守，每次回答几乎一样）
print("=" * 50)
print("【temperature = 0】问 3 次，观察答案是否几乎相同")
print("=" * 50)
model_zero = ChatDeepSeek(model="deepseek-chat", temperature=0)
for i in range(3):
    print(f"第{i+1}次：{model_zero.invoke(question).content}")

# 实验 2：temperature=1.5（很随机，每次回答都不同）
print()
print("=" * 50)
print("【temperature = 1.5】问 3 次，观察答案差异")
print("=" * 50)
model_crazy = ChatDeepSeek(model="deepseek-chat", temperature=1.5)
for i in range(3):
    print(f"第{i+1}次：{model_crazy.invoke(question).content}")
