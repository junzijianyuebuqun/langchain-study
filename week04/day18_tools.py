# ============================================
# Day 18：自定义 Tool 工具
# 面试点：Tool 三要素 = name + description + 参数schema
#         description 写得好不好直接决定 Agent 会不会正确使用它！
# ============================================
from langchain_core.tools import tool

# --------------------------------------------
# @tool 装饰器：把普通函数变成 Agent 能用的工具
# 函数名 → 工具名；docstring → description（给模型看的说明书！）
# 类型注解 → 参数 schema
# --------------------------------------------

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入如 '3 * (4 + 5)'，返回计算结果。"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算出错：{e}"

@tool
def get_weather(city: str) -> str:
    """查询指定中国城市今天的天气。输入城市名，如 '北京'。"""
    # 学习阶段先 mock（假数据），真实项目接天气 API
    fake_data = {"北京": "晴，25°C", "上海": "多云，28°C", "广州": "雷阵雨，31°C"}
    return fake_data.get(city, f"{city}今天晴，26°C（模拟数据）")

# --------------------------------------------
# 观察工具的三要素
# --------------------------------------------
print("工具名：", calculator.name)
print("说明书：", calculator.description)
print("参数schema：", calculator.args)

# 工具也可以直接当函数调用测试
print("\n直接调用测试：")
print(calculator.invoke({"expression": "123 * 456"}))
print(get_weather.invoke({"city": "北京"}))
