from langchain_community.utils.openai_functions import (
    convert_pydantic_to_openai_function,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import ChatOpenAI


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


openai_functions = [convert_pydantic_to_openai_function(Joke)]
model = ChatOpenAI(temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are helpful assistant"), ("user", "{input}")]
)

print("================ JsonOutputFunctionsParser ===============")
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

parser = JsonOutputFunctionsParser()
chain = prompt | model.bind(functions=openai_functions) | parser
print(chain.invoke({"input": "tell me a joke"}))
for s in chain.stream({"input": "tell me a joke"}):
    print(s)

print("================ JsonKeyOutputFunctionsParser ===============")
from typing import List

from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser


class Jokes(BaseModel):
    """Jokes to tell user."""

    joke: List[Joke]
    funniness_level: int


parser = JsonKeyOutputFunctionsParser(key_name="joke")

openai_functions = [convert_pydantic_to_openai_function(Jokes)]
chain = prompt | model.bind(functions=openai_functions) | parser
print(chain.invoke({"input": "tell me two jokes"}))

for s in chain.stream({"input": "tell me two jokes"}):
    print(s)

print("================ PydanticOutputFunctionsParser ===============")
from langchain.output_parsers.openai_functions import PydanticOutputFunctionsParser


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # You can add custom validation logic easily with Pydantic.
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field


parser = PydanticOutputFunctionsParser(pydantic_schema=Joke)
openai_functions = [convert_pydantic_to_openai_function(Joke)]
chain = prompt | model.bind(functions=openai_functions) | parser
print(chain.invoke({"input": "tell me a joke"}))
