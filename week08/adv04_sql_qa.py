# ============================================
# 官方教程补全 4：SQL 问答（Question-Answering with SQL）
# 官方地址：/docs/tutorials/sql_qa/
# 场景：用自然语言查数据库（"上个月销售额多少" → 自动生成SQL执行）
# 流程：问题 → 模型生成SQL → 执行SQL → 模型解读结果
# ============================================
import sqlite3
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------
# 1. 准备一个 SQLite 示例数据库（员工表）
# --------------------------------------------
DB = "week08/company.db"
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

# 2. 把表结构告诉模型（schema 是生成正确 SQL 的关键！）
schema = """
表名：employees（员工表）
字段：id(工号), name(姓名), department(部门), salary(月薪), hire_year(入职年份)
"""

model = ChatDeepSeek(model="deepseek-chat", temperature=0)

# --------------------------------------------
# 3. 第一步：问题 → SQL
# --------------------------------------------
sql_prompt = ChatPromptTemplate.from_template("""
基于以下数据库结构，把用户问题转成一条 SQLite 查询语句。
只输出 SQL 本身，不要任何解释，不要 markdown 代码块标记。

{schema}

问题：{question}
""")
sql_chain = sql_prompt | model | StrOutputParser()

# --------------------------------------------
# 4. 第二步：执行 SQL + 解读结果
# --------------------------------------------
answer_prompt = ChatPromptTemplate.from_template("""
根据以下信息回答用户问题，简洁明了：
问题：{question}
执行的SQL：{sql}
查询结果：{result}
""")
answer_chain = answer_prompt | model | StrOutputParser()

def ask(question: str):
    sql = sql_chain.invoke({"schema": schema, "question": question}).strip()
    print(f"\n❓ {question}")
    print(f"🔧 生成SQL：{sql}")
    try:
        rows = conn.execute(sql).fetchall()
    except Exception as e:
        print(f"⚠️ SQL执行失败：{e}")
        return
    print(f"📊 查询结果：{rows}")
    answer = answer_chain.invoke({"question": question, "sql": sql, "result": str(rows)})
    print(f"💬 回答：{answer}")

ask("技术部有哪些人？")
ask("哪个部门的平均月薪最高？")
ask("2020年之后入职的员工有几个？")

conn.close()

# ⚠️ 安全提示（面试加分）：生产环境绝不能让模型直接执行任意SQL！
# 要用只读账号 + 限制查询类型（只允许SELECT）+ SQL校验
