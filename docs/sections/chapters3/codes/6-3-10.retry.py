from langchain.output_parsers import (
    OutputFixingParser,
    PydanticOutputParser,
)
from langchain.prompts import (
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAI

template = """
根据用户问题，提供应该采取的行动和行动输入。
{format_instructions}
问题：{query}
响应：
"""


class Action(BaseModel):
    action: str = Field(description="要采取的行动")
    action_input: str = Field(description="行动的输入")


parser = PydanticOutputParser(pydantic_object=Action)
prompt = PromptTemplate(
    template="回答用户查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
prompt_value = prompt.format_prompt(query="谁是莱昂纳多·迪卡普里奥的女朋友？")

bad_response = '{"action": "search"}'

from langchain.output_parsers import RetryOutputParser

retry_parser = RetryOutputParser.from_llm(parser=parser, llm=OpenAI(temperature=0))
retry_parser.parse_with_prompt(bad_response, prompt_value)
Action(action="search", action_input="莱昂纳多·迪卡普里奥的女朋友")
