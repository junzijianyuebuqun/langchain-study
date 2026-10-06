# ============================================================
# 官网教程 09：基于 SQL 的问答（Question-Answering with SQL）
# 官方地址：https://python.langchain.com/docs/tutorials/sql_qa/
# 所属模块：LangChain（prompts + 自定义 SQL 执行）
# 官方目标：构建执行 SQL 查询来回答问题的系统
#
# 【与官网的对应关系】
#   官网用 Chinook 数据库 → 这里自建员工表（结构更直观）
#   官网核心：自然语言 → SQL → 执行 → 自然语言回答，完全一致
# ============================================================
import sqlite3
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 准备数据库（对应官网的 Chinook sample） ---
DB = "official_tutorials/company.db"
conn = sqlite3.connect(DB)
conn.executescript("""
DROP TABLE IF EXISTS employees;
CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER, hire_year INTEGER);
INSERT INTO employees (name, department, salary, hire_year) VALUES
('张伟', '技术部', 25000, 2020), ('李梅', '技术部', 22000, 2022),
('王芳', '市场部', 18000, 2019), ('刘强', '市场部', 20000, 2021),
('陈静', '人事部', 15000, 2023), ('杨帆', '技术部', 30000, 2018);
""")
conn.commit()

schema = """表 employees：id(工号), name(姓名), department(部门), salary(月薪), hire_year(入职年份)"""

model = init_chat_model("deepseek-chat", model_provider="deepseek", temperature=0)

# --- 官网核心步骤1：问题 → SQL（write_query） ---
write_query = (
    ChatPromptTemplate.from_template(
        "根据表结构把问题转成一条 SQLite 查询，只输出 SQL：\n{schema}\n\n问题：{question}"
    ) | model | StrOutputParser()
)

# --- 官网核心步骤2：执行 SQL（execute_query） ---
def execute_query(sql: str):
    return conn.execute(sql.strip()).fetchall()

# --- 官网核心步骤3：结果 → 自然语言（generate_answer） ---
generate_answer = (
    ChatPromptTemplate.from_template(
        "根据查询结果回答问题：\n问题：{question}\nSQL：{sql}\n结果：{result}"
    ) | model | StrOutputParser()
)

for q in ["工资最高的员工是谁？", "每个部门有多少人？", "入职满5年的员工平均月薪多少？"]:
    sql = write_query.invoke({"schema": schema, "question": q})
    rows = execute_query(sql)
    answer = generate_answer.invoke({"question": q, "sql": sql, "result": str(rows)})
    print(f"❓ {q}\n🔧 {sql}\n💬 {answer}\n")

conn.close()
# 官网安全提示：生产环境用只读账号 + 只许 SELECT
