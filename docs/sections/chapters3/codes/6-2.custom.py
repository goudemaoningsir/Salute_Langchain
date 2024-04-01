from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")

from langchain_core.messages import AIMessage, AIMessageChunk


def parse(ai_message: AIMessage) -> str:
    """解析AI消息。"""
    return ai_message.content.swapcase()


chain = model | parse

print(chain.invoke("hello"))

for chunk in chain.stream("tell me about yourself in one sentence"):
    print(chunk, end="|", flush=True)

from typing import Iterable
from langchain_core.runnables import RunnableGenerator


def streaming_parse(chunks: Iterable[AIMessageChunk]) -> Iterable[str]:
    for chunk in chunks:
        yield chunk.content.swapcase()


streaming_parse = RunnableGenerator(streaming_parse)

chain = model | streaming_parse

print(chain.invoke("hello"))

for chunk in chain.stream("tell me about yourself in one sentence"):
    print(chunk, end="|", flush=True)