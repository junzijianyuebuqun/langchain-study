# ============================================================
# 官网教程 01：聊天模型和提示词（Chat models and prompts）
# 官方地址：https://python.langchain.com/docs/tutorials/llm_chain/
# 所属模块：LangChain（langchain_core: chat_models + prompts）
# 官方目标：用 提示词模板 + 聊天模型 构建第一个 LLM 应用
#
# 【与官网的对应关系】
#   官网用 init_chat_model + OpenAI → 这里换成 DeepSeek，其余结构一致
#   官网示例是"翻译应用" → 完全保留
# ============================================================
from dotenv import load_dotenv
load_dotenv()

# --- 官网写法：init_chat_model 统一初始化（换模型只改参数） ---
from langchain.chat_models import init_chat_model

model = init_chat_model("deepseek-chat", model_provider="deepseek")

# --- 官网第一部分：直接调模型（消息列表） ---
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("把以下中文翻译成英文："),
    HumanMessage("你好，世界！"),
]
print("【直接调用】", model.invoke(messages).content)

# --- 官网第二部分：流式输出 ---
print("\n【流式输出】", end="")
for token in model.stream(messages):
    print(token.content, end="", flush=True)

# --- 官网第三部分：Prompt 模板（参数化，这是本教程重点） ---
from langchain_core.prompts import ChatPromptTemplate

system_template = "把以下文本从{source_language}翻译成{target_language}"

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}"),
])

# 官网演示：模板单独使用（生成 PromptValue）
prompt_value = prompt_template.invoke({
    "source_language": "中文", "target_language": "日文", "text": "今天天气真好"
})
print("\n\n【模板生成的消息】", prompt_value.to_messages())

# 官网演示：模板 | 模型 组成链
chain = prompt_template | model
response = chain.invoke({
    "source_language": "中文", "target_language": "日文", "text": "今天天气真好"
})
print("【链式调用】", response.content)
