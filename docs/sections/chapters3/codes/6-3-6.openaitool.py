from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import ChatOpenAI


class Joke(BaseModel):
    """要告诉用户的笑话。"""

    setup: str = Field(description="设置笑话的问题")
    punchline: str = Field(description="解决笑话的答案")


model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).bind_tools([Joke])
print(model.kwargs["tools"])

print("========================JsonOutputToolsParser=================================")
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are helpful assistant"), ("user", "{input}")]
)
from langchain.output_parsers.openai_tools import JsonOutputToolsParser

parser = JsonOutputToolsParser()
chain = prompt | model | parser
print(chain.invoke({"input": "tell me a joke"}))

print("=====================JsonOutputKeyToolsParser==============================")
from typing import List

from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser

parser = JsonOutputKeyToolsParser(key_name="Joke")

chain = prompt | model | parser
print(chain.invoke({"input": "tell me a joke"}))
parser = JsonOutputKeyToolsParser(key_name="Joke", first_tool_only=True)
chain = prompt | model | parser
print(chain.invoke({"input": "告诉我一个笑话"}))

print("=====================PydanticToolsParser==============================")
from langchain.output_parsers.openai_tools import PydanticToolsParser


class Joke(BaseModel):
    """要告诉用户的笑话。"""

    setup: str = Field(description="设置笑话的问题")
    punchline: str = Field(description="解决笑话的答案")

    # 您可以使用 Pydantic 轻松添加自定义验证逻辑。
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "？":
            raise ValueError("问题格式不正确！")
        return field


parser = PydanticToolsParser(tools=[Joke])

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).bind_tools([Joke])
chain = prompt | model | parser
print(chain.invoke({"input": "告诉我一个笑话"}))
