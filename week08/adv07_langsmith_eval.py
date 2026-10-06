# ============================================
# 官方教程补全 7：LangSmith 应用评估（Evaluation）
# 官方地址：https://docs.smith.langchain.com/evaluation
# 场景：RAG/Agent 做得好与坏，不能凭感觉 → 用数据集自动评估打分
#
# ⚠️ 需要先配置 LangSmith（.env 里的 LANGSMITH_API_KEY），
#    未配置时脚本会打印学习要点后退出
# ============================================
import os
from dotenv import load_dotenv
load_dotenv()

if not (os.getenv("LANGSMITH_TRACING", "").lower() == "true" and os.getenv("LANGSMITH_API_KEY")):
    print("❌ 未配置 LangSmith，打印学习要点：")
    print("""
📖 官方评估教程的核心流程（面试必答）：

1. 建数据集（Dataset）：
   准备一批"问题 + 标准答案"对，作为考试的题库
   client.create_dataset(...) / client.create_examples(...)

2. 定义评估器（Evaluator）：
   给答案打分的规则，最常用"LLM-as-judge"：
   让另一个模型对比"系统答案 vs 标准答案"，输出 0-1 分

3. 跑评估（evaluate）：
   系统自动把你的 RAG/Agent 在题库上跑一遍，
   每题打分，输出准确率报告

4. 迭代优化：
   改了 chunk_size / prompt / k 值 → 重跑评估 → 分数对比
   → 数据驱动地证明"优化有效"（面试亮点！）

💡 配置好 LangSmith 后重跑本脚本即可看到完整演示。
""")
    raise SystemExit(0)

# ============== 以下需要 LangSmith 配置好后才能运行 ==============
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# 1. 建数据集：问题 + 标准答案（基于 week03 请假制度知识库）
dataset_name = "company-policy-qa"
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name, description="请假制度问答题库")
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {"inputs": {"q": "年假有几天？"}, "outputs": {"a": "工龄1-10年5天，10-20年10天，20年以上15天"}},
            {"inputs": {"q": "病假工资怎么算？"}, "outputs": {"a": "按80%发放"}},
            {"inputs": {"q": "请10天假谁审批？"}, "outputs": {"a": "部门经理和人力资源部双重审批"}},
        ],
    )
    print(f"✅ 数据集 {dataset_name} 已创建（3道题）")

# 2. 被测系统：简化版 RAG（复用 week03 的向量库）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vs = Chroma(persist_directory="week03/chroma_db", embedding_function=embeddings)
model = ChatDeepSeek(model="deepseek-chat")

def rag_app(inputs: dict) -> dict:
    docs = vs.similarity_search(inputs["q"], k=3)
    ctx = "\n".join(d.page_content for d in docs)
    r = model.invoke(f"根据资料回答，没有就说不知道：\n{ctx}\n\n问题：{inputs['q']}")
    return {"output": r.content}

# 3. LLM-as-judge 评估器
judge = ChatDeepSeek(model="deepseek-chat", temperature=0)

def correctness(outputs: dict, reference_outputs: dict) -> dict:
    r = judge.invoke(
        f"对比两个答案语义是否一致，只回答 1（一致）或 0（不一致）：\n"
        f"系统答案：{outputs['output']}\n标准答案：{reference_outputs['a']}"
    )
    score = 1 if "1" in r.content.strip()[:3] else 0
    return {"key": "correctness", "score": score}

# 4. 跑评估
results = evaluate(
    rag_app,
    data=dataset_name,
    evaluators=[correctness],
    experiment_prefix="rag-v1",
)
print("✅ 评估完成，去 LangSmith 网页查看每题得分和整体准确率")
