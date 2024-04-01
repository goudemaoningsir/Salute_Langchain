from langchain.output_parsers.enum import EnumOutputParser
from enum import Enum


class Colors(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


parser = EnumOutputParser(enum=Colors)


from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template(
    """What color eyes does this person have?\n> Person: {person}\nInstructions: {instructions}""",
).partial(instructions=parser.get_format_instructions())

chain = prompt | ChatOpenAI() | parser

print(chain.invoke({"person": "Frank Sinatra"}))
