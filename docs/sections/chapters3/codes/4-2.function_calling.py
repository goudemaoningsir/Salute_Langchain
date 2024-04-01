from langchain_openai import ChatOpenAI, OpenAI

print("####################      step1 初始化模型     ####################")
chat = ChatOpenAI(model="gpt-3.5-turbo")

print("####################      step2  绑定函数       ####################")
from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
)  # 注意这里的文档字符串非常关键，因为它们将随类名一起传递给模型


class Multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


print("####################      step3  bind_tools       ####################")
llm_with_tools = chat.bind_tools([Multiply])
print(llm_with_tools.invoke("what's 3 * 12"))

print("####################      step4  工具解析器       ####################")
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser

tool_chain = llm_with_tools | JsonOutputToolsParser()
print(tool_chain.invoke("what's 3 * 12"))

print("####################      step5  Pydantic       ####################")
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

tool_chain = llm_with_tools | PydanticToolsParser(tools=[Multiply])
print(tool_chain.invoke("what's 3 * 12"))

print("####################      step6 tool_choice       ####################")
llm_with_tools = chat.bind_tools([Multiply], tool_choice="Multiply")
print(tool_chain.invoke("what's 3 * 12"))

print("####################      step7 tool_choice one    ####################")
llm_with_multiply = chat.bind_tools([Multiply], tool_choice="Multiply")
print(
    llm_with_multiply.invoke(
        "make up some numbers if you really want but I'm not forcing you"
    )
)

print("####################      step8 定义函数模式   ####################")
import json
from langchain_core.utils.function_calling import convert_to_openai_tool


def multiply(a: int, b: int) -> int:
    """Multiply two integers together.
    Args:
        a: First integer
        b: Second integer
    """
    return a * b


print(json.dumps(convert_to_openai_tool(multiply), indent=2))

print("####################      step9 Pydantic   ####################")
from langchain_core.pydantic_v1 import BaseModel, Field


class multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


print(json.dumps(convert_to_openai_tool(multiply), indent=2))

print("####################      step10 LangChain   ####################")
from typing import Any, Type

from langchain_core.tools import BaseTool


class MultiplySchema(BaseModel):
    """Multiply tool schema."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class Multiply(BaseTool):
    args_schema: Type[BaseModel] = MultiplySchema
    name: str = "multiply"
    description: str = "Multiply two integers together."

    def _run(self, a: int, b: int, **kwargs: Any) -> Any:
        return a * b


# Note: we're passing in a Multiply object not the class itself.
print(json.dumps(convert_to_openai_tool(Multiply()), indent=2))
