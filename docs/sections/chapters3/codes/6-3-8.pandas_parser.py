import pprint
from typing import Any, Dict
import pandas as pd
from langchain.output_parsers import PandasDataFrameOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0)


# 仅出于文档目的。
def format_parser_output(parser_output: Dict[str, Any]) -> None:
    for key in parser_output.keys():
        parser_output[key] = parser_output[key].to_dict()
    return pprint.PrettyPrinter(width=4, compact=True).pprint(parser_output)


# 定义您希望的 Pandas DataFrame。
df = pd.DataFrame(
    {
        "num_legs": [2, 4, 8, 0],
        "num_wings": [2, 0, 0, 0],
        "num_specimen_seen": [10, 2, 1, 8],
    }
)  # 设置解析器 + 将指令注入提示模板。
parser = PandasDataFrameOutputParser(dataframe=df)
# 这里是执行列操作的一个例子。
df_query = "检索 num_wings 列。"
# 设置提示。
prompt = PromptTemplate(
    template="回答用户查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | model | parser
parser_output = chain.invoke({"query": df_query})
format_parser_output(parser_output)
# 这里是执行行操作的一个例子。
df_query = "检索第一行。"
# 设置提示。
prompt = PromptTemplate(
    template="回答用户查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | model | parser
parser_output = chain.invoke({"query": df_query})
format_parser_output(parser_output)


# 这里是执行随机 Pandas DataFrame 操作的一个例子，限制行数。
df_query = "从第 1 行到第 3 行检索 num_legs 列的平均值。"
# 设置提示。
prompt = PromptTemplate(
    template="回答用户查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | model | parser
parser_output = chain.invoke({"query": df_query})
print(parser_output)


# 这里是执行格式不正确的查询的一个例子。
df_query = "检索 num_fingers 列的平均值。"
# 设置提示。
prompt = PromptTemplate(
    template="回答用户查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | model | parser
parser_output = chain.invoke({"query": df_query})
