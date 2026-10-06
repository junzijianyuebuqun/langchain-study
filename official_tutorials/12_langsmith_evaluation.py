# ============================================================
# 官网教程 12：评估你的 LLM 应用（LangSmith Evaluation）
# 官方地址：https://docs.smith.langchain.com/evaluation
# 所属模块：LangSmith（evaluation 评估）
# 官方目标：用数据集 + 评估器自动给应用打分
#
# ⚠️ 需要 .env 配置 LangSmith（LANGSMITH_TRACING/KEY）才能完整运行；
#    未配置时打印核心概念
# ============================================================
import os
from dotenv import load_dotenv
load_dotenv()

if not (os.getenv("LANGSMITH_TRACING", "").lower() == "true" and os.getenv("LANGSMITH_API_KEY")):
    print("未配置 LangSmith。官方评估教程核心概念：")
    print("""
1. Dataset（数据集）：一批"问题+标准答案"对，相当于题库
2. Evaluator（评估器）：打分规则，常用 LLM-as-judge（模型当裁判）
3. evaluate()：自动在题库上跑你的应用，逐题打分
4. 迭代：改参数 → 重跑 → 分数对比（数据驱动优化）
""")
    raise SystemExit(0)

# ============== 配置好 LangSmith 后运行以下完整流程 ==============
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 1. 建数据集（对应官网 create_dataset + create_examples）
dataset_name = "official-eval-demo"
if not client.has_dataset(dataset_name=dataset_name):
    ds = client.create_dataset(dataset_name)
    client.create_examples(dataset_id=ds.id, examples=[
        {"inputs": {"q": "年假有几天？"}, "outputs": {"a": "按工龄：1-10年5天，10-20年10天，20年以上15天"}},
        {"inputs": {"q": "病假工资怎么算？"}, "outputs": {"a": "按80%发放"}},
    ])
    print("✅ 数据集已创建")

# 2. 被测应用（简化 RAG）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

model = init_chat_model("deepseek-chat", model_provider="deepseek")
vs = Chroma(persist_directory="official_tutorials/chroma_rag_db",
            embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5"))

def rag_app(inputs: dict) -> dict:
    docs = vs.similarity_search(inputs["q"], k=3)
    ctx = "\n".join(d.page_content for d in docs)
    return {"output": model.invoke(f"根据资料回答：\n{ctx}\n\n问题：{inputs['q']}").content}

# 3. LLM-as-judge 评估器（对应官网 evaluator）
judge = init_chat_model("deepseek-chat", model_provider="deepseek", temperature=0)

def correctness(outputs: dict, reference_outputs: dict) -> dict:
    r = judge.invoke(f"两答案语义是否一致？只答1或0：\n系统：{outputs['output']}\n标准：{reference_outputs['a']}")
    return {"key": "correctness", "score": 1 if "1" in r.content[:3] else 0}

# 4. 跑评估（对应官网 evaluate）
evaluate(rag_app, data=dataset_name, evaluators=[correctness], experiment_prefix="rag-v1")
print("✅ 评估完成，去 smith.langchain.com 查看报告")
