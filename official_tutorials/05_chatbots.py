# ============================================================
# 官网教程 05：聊天机器人（Chatbots）
# 官方地址：https://python.langchain.com/docs/tutorials/chatbot/
# 所属模块：LangChain（langchain_core: messages + trim_messages）
# 官方目标：构建带记忆的聊天机器人
#
# 【与官网的对应关系】
#   官网知识点1：模型本身无状态（无记忆）→ 本脚本实验1
#   官网知识点2：把历史消息传入（手动记忆）→ 本脚本实验2
#   官网知识点3：trim_messages 裁剪历史防 token 爆炸 → 本脚本实验3
# ============================================================
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages

model = init_chat_model("deepseek-chat", model_provider="deepseek")

# --- 官网实验1：模型没有记忆 ---
print("【实验1】模型没有记忆：")
model.invoke([HumanMessage("你好！我叫小明")])
r = model.invoke([HumanMessage("我叫什么名字？")])
print(f"  单独问'我叫什么' → {r.content[:40]}...\n")

# --- 官网实验2：传入历史消息 = 记忆 ---
print("【实验2】带上历史再问：")
messages = [
    HumanMessage("你好！我叫小明"),
    AIMessage("你好小明！很高兴认识你，有什么可以帮你的吗？"),
    HumanMessage("我叫什么名字？"),
]
r2 = model.invoke(messages)
print(f"  带历史问 → {r2.content}\n")

# --- 官网实验3：trim_messages 裁剪历史（防 token 爆炸，面试加分） ---
print("【实验3】trim_messages 裁剪：")
long_history = [
    SystemMessage("你是友好的助手"),
    HumanMessage("我叫小明"), AIMessage("你好小明"),
    HumanMessage("我喜欢吃火锅"), AIMessage("火锅确实美味"),
    HumanMessage("我养了一只猫"), AIMessage("猫咪真可爱"),
    HumanMessage("它叫豆豆"), AIMessage("豆豆这名字不错"),
]

# ⚠️ 注意：官网用 token_counter=model（OpenAI 支持），
# DeepSeek 未实现 token 计数接口 → 改用 len（按消息条数计），效果等价
trimmer = trim_messages(
    max_tokens=6,            # 最多保留 6 条消息（用 len 计数时单位=条数）
    strategy="last",         # 保留最新的
    token_counter=len,       # 按消息条数计；OpenAI 模型可换成 model 按 token 计
    include_system=True,     # system 消息始终保留
    start_on="human",        # 裁剪后必须以 human 消息开头
)
trimmed = trimmer.invoke(long_history)
print(f"  原始 {len(long_history)} 条 → 裁剪后 {len(trimmed)} 条")
for m in trimmed:
    print(f"    [{m.__class__.__name__}] {m.content[:20]}")

# --- 综合：带裁剪的多轮对话骨架（官网最终形态） ---
print("\n【综合】模拟多轮对话（自动裁剪历史）：")
history = [SystemMessage("你是友好的助手，回答一句话")]
for user_say in ["我叫小明", "我喜欢吃火锅", "我养了一只叫豆豆的猫", "我的猫叫什么？"]:
    history.append(HumanMessage(user_say))
    response = model.invoke(trimmer.invoke(history))
    history.append(AIMessage(response.content))
    print(f"  你：{user_say}\n  🤖：{response.content}")
