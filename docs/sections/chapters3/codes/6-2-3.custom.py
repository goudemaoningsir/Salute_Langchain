from typing import List
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")


class StrInvertCase(BaseGenerationOutputParser[str]):
    """一个示例解析器，用于反转消息中字符的大小写。
    这只是一个出于演示目的和保持示例尽可能简单的示例解析。
    """

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> str:
        """将一系列模型生成解析为特定格式。
        参数:
            result: 要解析的生成列表。假设生成是针对单一模型输入的不同候选输出。
            许多解析器假设只传递了一个生成给我们。
            partial: 是否允许部分结果。这对于支持流的解析器使用
        """
        if len(result) != 1:
            raise NotImplementedError(
                "This output parser can only be used with a single generation."
            )
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            # 说这个只与聊天生成一起工作
            raise OutputParserException(
                "This output parser can only be used with a chat generation."
            )
        return generation.message.content.swapcase()


chain = model | StrInvertCase

print(chain.invoke("Tell me a short sentence about yourself"))
