# ============================================
# Day 31：LangSmith 链路追踪配置
# 作用：在网页上看到每一次模型调用的完整链路（输入/输出/耗时/token数）
#
# 使用前提（需要你自己操作）：
#   1. 注册 https://smith.langchain.com
#   2. 拿到 API Key 填进 .env 的三行配置（见下）
#   3. 之后运行任何代码，都会自动上报追踪数据
#
# 本脚本：演示配置方式 + 检查配置是否生效
# ============================================
import os
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------
# .env 里需要加的三行（拿到 LangSmith Key 后取消注释填写）：
#   LANGSMITH_TRACING=true
#   LANGSMITH_API_KEY=lsv2_pt_你的key
#   LANGSMITH_PROJECT=langchain-study
# --------------------------------------------

tracing_on = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
has_key = bool(os.getenv("LANGSMITH_API_KEY"))
project = os.getenv("LANGSMITH_PROJECT", "未设置")

print("LangSmith 配置检查：")
print(f"  追踪开关 LANGSMITH_TRACING：{'✅ 已开启' if tracing_on else '❌ 未开启'}")
print(f"  API Key：{'✅ 已配置' if has_key else '❌ 未配置'}")
print(f"  项目名：{project}")

if tracing_on and has_key:
    # 跑一次调用，数据会自动出现在 smith.langchain.com 的项目面板里
    from langchain_deepseek import ChatDeepSeek
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    chain = (
        ChatPromptTemplate.from_template("用一句话介绍{topic}")
        | ChatDeepSeek(model="deepseek-chat")
        | StrOutputParser()
    )
    print("\n运行一次链调用（去 LangSmith 网页看追踪记录）：")
    print(chain.invoke({"topic": "LangSmith"}))
else:
    print("\n💡 配置好 LangSmith 后重跑本脚本，就能在网页上看到链路追踪了")
    print("   面试话术：『我用 LangSmith 做全链路追踪和效果评估』")
